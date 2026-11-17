"""A live NFT2 stream with events, as libarc's arc_nft2_stream_* (c/src/arc_nft2.c): finals per closed utterance (Arc's
segmenter, Nft2Finals.decode), and with live=True the open utterance decoded as it arrives (stable log-mel frames -> encoder
chunks with carried caches -> the greedy RNNT with carried state; nothing is decoded twice), with events:

    ls = arc.live_stream(names=names, context={'dom': 'space'}, live=True)    # = LiveStream(arc, ...)
    ls.feed_reference(tts_pcm, t)       # what the device plays (t: stream time of its first sample); optional
    out = ls.feed(pcm)                  # -> dict(utterances=[closed utterances], events=[...])
    ls.set_names(names)                 # hot list: the open utterance's bias restarts on it, finals use it from now
    ls.end_turn()                       # close the open utterance now
    ls.flush()                          # the end of the stream

events (dicts, in order): {'type': 'partial', 't', 'utt', 'text', 'n_stable', 'start', 'end'} (n_stable leading words no later
partial changes), {'type': 'entity', 'kind': 'name' | 'almost' | 'quantity', 'text', 'entity', 'kb_index', 'p', ...} (in the
stable words; a quantity once a word follows it), {'type': 'eot', 'p'} (nft2_post.eot_prob on the open utterance's words and the
pause so far, every feed; and at each final), {'type': 'barge_in', 'p'} (speech that is not the playback while the device
plays: echo.barge_in), {'type': 'final', 'text', 'kind': None | 'playback' | 'dropped', 'p': playback_leak}.
Utterances get 'playback_leak' / 'playback' when a reference was fed. Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0;
see LICENSE.
"""
import numpy as np
import torch

from .echo import EchoCanceller, barge_in
from .nft2_post import ALMOST, _wk, almost, eot_prob, kb_forms, common_words
from .quantity import quantities

SR, FR, HOP, NFFT = 16000, 1600, 160, 512


class _Live:
    """the open utterance's incremental decode"""

    def __init__(self, arc, a, utt):
        bb = arc.bb
        self.arc, self.a, self.utt = arc, int(a), int(utt)
        self.enc = bb.enc
        c = self.enc.cfg
        L, d = int(c['n_layers']), self.enc.d_model
        self.cch = np.zeros((1, L, int(c['last_channel_cache_size']), d), np.float32)
        self.ctm = np.zeros((1, L, d, int(c['conv_cache'])), np.float32)
        self.clen = np.zeros(1, np.int64)
        self.pre = np.zeros((self.enc.pre_n, 80), np.float32)
        self.nmel = 0
        self.mel = np.zeros((0, 80), np.float32)
        self.t_enc = 0
        self.st = None
        self.ids, self.frames, self.tok = [], [], []
        self.sent_text = None
        self.ent = set()

    def step(self, audio, trie, alpha):
        """new stable log-mel frames of audio[a:] -> encoder chunks -> RNNT; True when tokens were added"""
        enc = self.enc
        avail = len(audio) - self.a
        stable = min((avail - NFFT // 2) // HOP + 1 if avail >= NFFT // 2 else 0, avail // HOP)
        have = self.nmel + len(self.mel)
        if stable > have:
            st = (have - 2) * HOP if have >= 2 else 0
            off = 2 if have >= 2 else have
            y = np.ascontiguousarray(audio[self.a + st:], np.float32)
            M, fl = enc.mel(torch.as_tensor(y)[None], torch.tensor([len(y)]))
            M = M[0, :int(fl[0])].numpy()
            add = min(stable - have, M.shape[0] - off)
            if add > 0:
                self.mel = np.concatenate([self.mel, M[off:off + add]], 0)
        n0 = len(self.ids)
        while True:
            first = self.nmel == 0 and enc.first_n > 0
            nn = enc.first_n if first else enc.chunk
            if len(self.mel) < nn:
                break
            new = self.mel[:nn]
            x = new if first else np.concatenate([self.pre, new], 0)
            self.pre = np.concatenate([self.pre, x], 0)[-enc.pre_n:]
            r = (enc.enc0 if first else enc.enc).run(enc.outs[:5], {'audio_signal': np.ascontiguousarray(x.T[None]),
                                                                   'length': np.array([x.shape[0]], np.int64),
                                                                   'cache_last_channel': self.cch, 'cache_last_time': self.ctm,
                                                                   'cache_last_channel_len': self.clen})
            out, olen, self.cch, self.ctm, self.clen = r[:5]
            k = int(olen[0])
            self.mel = self.mel[nn:]
            self.nmel += nn
            if k > 0:
                ids, frames, self.st = self.arc.bb.rnnt.greedy(out[0, :, :k].T, st=self.st, trie=trie, alpha=alpha)
                self.ids += ids
                self.frames += [f + self.t_enc for f in frames]
                self.tok += self.st.get('tok') or []
                self.t_enc += k
        return len(self.ids) != n0

    def words(self):
        return self.arc.bb.pieces.words(self.ids, self.frames, self.a / SR, stats=self.tok)


class LiveStream:
    def __init__(self, arc, names=None, context=None, domain='assistant', live=True, almost_names=False, nbest=0,
                 playback_threshold=0.7, playback_drop=False, echo_decode=False, bias_names=None, guard=None, room=True):
        from .transcribe import UttSegmenter
        self.arc = arc
        self.domain = domain
        self.fin = arc.nft2_finals(names, bias_names, context=context)
        if nbest:
            self.fin.c['nbest'] = int(nbest)
        if guard is not None:
            self.fin.c['guard'] = bool(guard)
        self.seg = UttSegmenter(self.fin.tr.c)
        self.live_on = bool(live)
        self.almost_on = bool(almost_names)
        self.thr = float(playback_threshold or 0.7)
        self.drop = bool(playback_drop)
        self.echo_decode = bool(echo_decode)
        self.audio = np.zeros(0, np.float32)
        self.nf = 0
        self.n_out = 0
        self.lv = None
        self.barge = False
        self.ec = None
        self.utts = []
        self.names = arc._kb(names) if names else []
        self.common = common_words(arc.bb.pieces)
        self.pad = float(self.fin.c.get('edge_pad_s', 0.15))
        self.room_on = bool(room) and getattr(arc, 'room_model', None) is not None

    # ---------------------------------------------------------------- inputs
    def feed_reference(self, pcm, t=None):
        if self.ec is None:
            self.ec = EchoCanceller()
            if len(self.audio):
                self.ec.feed_mic(self.audio)
        self.ec.feed_ref(pcm, t)

    def set_names(self, names):
        self.fin.set_names(names)
        self.names = self.arc._kb(names) if names else []
        if self.lv is not None:
            self.lv.st = dict(self.lv.st or {}, b=None) if self.lv.st else None

    def feed(self, pcm):
        from .api import frame_db
        x = np.asarray(pcm, np.float32).reshape(-1)
        self.audio = np.concatenate([self.audio, x])
        ev, utts = [], []
        if self.ec is not None:
            self.ec.feed_mic(x)
        while (self.nf + 1) * FR <= len(self.audio):
            for f0, f1, _ in self.seg.feed(self.nf, frame_db(self.audio[self.nf * FR:(self.nf + 1) * FR])):
                self._close(f0, f1, ev, utts)
            self.nf += 1
        self._live(ev)
        return dict(utterances=utts, events=ev)

    def end_turn(self):
        ev, utts = [], []
        for f0, f1, _ in self.seg.flush('end_turn'):
            self._close(f0, f1, ev, utts)
        self.lv = None
        return dict(utterances=utts, events=ev)

    def flush(self):
        from .api import frame_db
        ev, utts = [], []
        if len(self.audio) > self.nf * FR:
            for f0, f1, _ in self.seg.feed(self.nf, frame_db(self.audio[self.nf * FR:])):
                self._close(f0, f1, ev, utts)
            self.nf += 1
        for f0, f1, _ in self.seg.flush('end'):
            self._close(f0, f1, ev, utts)
        self.lv = None
        return dict(utterances=utts, events=ev, transcript=list(self.utts))

    # ---------------------------------------------------------------- finals
    def _close(self, f0, f1, ev, utts):
        a = max(0, int(f0 * FR - self.pad * SR))
        b = min(len(self.audio), int((f1 + 1) * FR + self.pad * SR))
        if b - a < 800:
            self.lv = None
            return
        y = self.audio[a:b]
        if self.ec is not None and self.echo_decode:
            r = self.ec.residual()
            y = np.array(y)
            m = min(b, len(r))
            if m > a:
                y[:m - a] = r[a:m]
        u = self.fin.decode(y, t0=a / SR, f0=f0, f1=f1, tail=True)
        idx = self.n_out
        self.n_out += 1
        kind = None
        if self.ec is not None:
            lk = self.ec.leak(f0 / 10, (f1 + 1) / 10)
            u['playback_leak'] = lk
            u['playback'] = lk is not None and lk >= self.thr
            if u['playback']:
                kind = 'dropped' if self.drop else 'playback'
        t = len(self.audio) / SR
        ev.append(dict(type='final', t=round(t, 3), utt=idx, text=u['text'], kind=kind, p=u.get('playback_leak'),
                       start=u['start'], end=u['end']))
        if self.live_on:
            ev.append(dict(type='eot', t=round(t, 3), utt=idx, p=eot_prob(u['words'], float(self.fin.tr.c.get('pause_s', 0.4)))))
        self.lv = None
        self.barge = False
        if kind == 'dropped':
            return
        from .api import utt_out
        o = utt_out(u)
        if self.room_on:
            ref = self.ec.ref[a:b] if self.ec is not None and self.ec.any_ref else None
            o['room'] = self.arc.room(self.audio[a:b], ref, frames=False, t0=a / SR)
        utts.append(o)
        self.utts.append(o)

    # ---------------------------------------------------------------- live
    def _live(self, ev):
        cur = self.seg.cur
        if not cur:
            return
        f_open = cur[0][0]
        t = round(len(self.audio) / SR, 3)
        if self.ec is not None and not self.barge and barge_in(self.ec, f_open / 10, self.nf / 10):
            self.barge = True
            lk = self.ec.leak(max(f_open / 10, self.nf / 10 - 0.3), self.nf / 10)
            ev.append(dict(type='barge_in', t=t, utt=self.n_out, p=round(1 - lk, 4), start=f_open / 10, end=self.nf / 10))
        if not self.live_on:
            return
        if self.lv is None:
            self.lv = _Live(self.arc, max(0, int(f_open * FR - self.pad * SR)), self.n_out)
        lv = self.lv
        changed = lv.step(self.audio, self.fin.trie, float(self.fin.c['alpha']))
        ws = lv.words()
        text = ' '.join(w['text'] for w in ws)
        if changed and text != lv.sent_text:
            lv.sent_text = text
            ev.append(dict(type='partial', t=t, utt=lv.utt, text=text, n_stable=max(0, len(ws) - 1),
                           start=ws[0]['start'] if ws else 0.0, end=ws[-1]['end'] if ws else 0.0, words=ws))
            self._entities(ws[:max(0, len(ws) - 1)], t, ev)
        if ws:
            low = 0
            for x in reversed(cur):
                if not x[2]:
                    break
                low += 1
            ev.append(dict(type='eot', t=t, utt=lv.utt, p=eot_prob(ws, low / 10)))

    def _entities(self, ws, t, ev):
        """list spellings (the longest at each position), sound-alikes and complete quantities in the stable words"""
        from .nft2 import Pieces
        lv = self.lv
        n = len(ws)
        if not n:
            return
        k = [_wk(w['text']) for w in ws]
        forms = []
        for i, e in enumerate(self.names):
            for f in [e.get('name', '')] + [str(a) for a in e.get('aliases') or []]:
                fk = [_wk(x) for x in Pieces.norm(str(f)).split()]
                if f and fk:
                    forms.append((i, fk))
        taken = set()
        i = 0
        while i < n:
            best = (0, -1)
            for kb, fk in forms:
                if len(fk) > best[0] and i + len(fk) <= n and all(fk[j] and fk[j] == k[i + j] for j in range(len(fk))):
                    best = (len(fk), kb)
            if best[1] < 0:
                i += 1
                continue
            L, kb = best
            taken.update(range(i, i + L))
            key = ('name', i, i + L, kb)
            if key not in lv.ent:
                lv.ent.add(key)
                ev.append(dict(type='entity', t=t, utt=lv.utt, kind='name', text=' '.join(w['text'] for w in ws[i:i + L]),
                               entity=self.names[kb]['name'], kb_index=kb, p=1.0, start=ws[i]['start'], end=ws[i + L - 1]['end']))
            i += L
        if self.names:
            for h in almost(ws, self.names, taken, None, self.common):
                taken.update(range(h['word_begin'], h['word_end']))
                key = ('almost', h['word_begin'], h['word_end'], h['kb_index'])
                if key in lv.ent:
                    continue
                lv.ent.add(key)
                ev.append(dict(type='entity', t=t, utt=lv.utt, kind='almost', text=h['heard'], entity=h['candidate'],
                               kb_index=h['kb_index'], p=h['p'], start=ws[h['word_begin']]['start'], end=ws[h['word_end'] - 1]['end']))
        tags = ['ENTITY_NAME' if j in taken else None for j in range(n)]
        for q in quantities([w['text'] for w in ws], tags):
            if q['word_end'] >= n:
                continue
            key = ('qty', q['word_begin'], q['word_end'])
            if key in lv.ent:
                continue
            lv.ent.add(key)
            ev.append(dict(type='entity', t=t, utt=lv.utt, kind='quantity', text=q['text'], entity=q['written'], kb_index=-1,
                           p=1.0, qtype=q['type'], value=q['value'], unit=q['unit'], relative=q['relative'],
                           start=ws[q['word_begin']]['start'], end=ws[q['word_end'] - 1]['end']))
