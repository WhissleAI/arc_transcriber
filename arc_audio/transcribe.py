import re, time
import numpy as np
import torch
SR, FR = (16000, 1600)
DEFAULT = dict(pause_s=0.4, min_s=1.0, max_s=15.0, cut_lookback_s=3.0, min_voiced_s=0.3, pad_s=0.15, decode='reencode', tokens_per_s=7.0, partial_rollback_tokens=6, partial_lock_s=2.0)

def _lkey(w):
    return re.sub("[^\\w']", '', w.lower())

def loopfix(text, K=2):
    w = [x for x in text.split() if _lkey(x) or not re.fullmatch('[-–—]+', x)]
    out = []
    i = 0
    while i < len(w):
        done = False
        for n in (1, 2, 3, 4):
            u = [_lkey(x) for x in w[i:i + n]]
            if len(u) < n or not all(u):
                continue
            r = 1
            while [_lkey(x) for x in w[i + r * n:i + (r + 1) * n]] == u:
                r += 1
            if r > K + 1:
                out += w[i:i + K * n]
                i += r * n
                done = True
                break
        if not done:
            out.append(w[i])
            i += 1
    return ' '.join(out)

def _db(a):
    return float(20 * np.log10(np.sqrt(np.mean(a * a)) + 1e-06)) if len(a) else -120.0

class UttSegmenter:

    def __init__(self, cfg=None):
        c = dict(DEFAULT, **cfg or {})
        self.c = c
        self.pause_n = int(round(c['pause_s'] * 10))
        self.min_n = int(round(c['min_s'] * 10))
        self.max_n = int(round(c['max_s'] * 10))
        self.look = int(round(c['cut_lookback_s'] * 10))
        self.min_voiced = int(round(c['min_voiced_s'] * 10))
        self.hist = []
        self.cur = []
        self.low_run = 0

    def _quiet(self, db):
        floor = float(np.percentile(self.hist, 10)) if len(self.hist) >= 20 else -60.0
        return db < max(-50.0, floor + 4.0) or db < -55.0

    def _emit(self, frames, reason):
        while frames and frames[-1][2]:
            frames = frames[:-1]
        while frames and frames[0][2]:
            frames = frames[1:]
        if sum((1 for x in frames if not x[2])) < self.min_voiced:
            return []
        return [(frames[0][0], frames[-1][0], reason)]

    def feed(self, f, db):
        self.hist.append(db)
        self.hist = self.hist[-300:]
        q = self._quiet(db)
        if not self.cur and q:
            return []
        self.cur.append((f, db, q))
        self.low_run = self.low_run + 1 if q else 0
        n = len(self.cur)
        if self.low_run >= self.pause_n and n - self.low_run >= self.min_n:
            out = self._emit(self.cur, 'pause')
            self.cur = []
            self.low_run = 0
            return out
        if n >= self.max_n:
            tail = self.cur[-self.look:]
            k = min(range(len(tail)), key=lambda i: tail[i][1])
            cut = n - len(tail) + k
            out = self._emit(self.cur[:cut + 1], 'max_length')
            self.cur = self.cur[cut + 1:]
            self.low_run = 0
            for x in self.cur:
                self.low_run = self.low_run + 1 if x[2] else 0
            return out
        return []

    def open_span(self):
        v = [x[0] for x in self.cur if not x[2]]
        return (self.cur[0][0], v[-1]) if v else None

    def flush(self):
        out = self._emit(self.cur, 'end') if self.cur else []
        self.cur = []
        self.low_run = 0
        return out

class Transcriber:

    def __init__(self, moon, tok, dev, dtype, cfg=None):
        self.m, self.tok, self.dev, self.dtype = (moon, tok, dev, dtype)
        self.c = dict(DEFAULT, **cfg or {})
        self.stats = dict(utterances=0, audio_s=0.0, decode_ms=0.0)
        from .tagdec import is_tagged_tokenizer
        self.tagged = bool(self.c.get('tagged', False)) and tok is not None and is_tagged_tokenizer(tok)
        self.last_ids = []

    def _max_len(self, n_samples):
        return 1 + int(float(self.c.get('len_mult', 1.0)) * (4 + self.c['tokens_per_s'] * n_samples / SR))

    def tagged_spans(self, text, ids=None):
        if not self.tagged:
            return []
        from .tagdec import readout, map_spans
        words, spans = readout(self.tok, self.last_ids if ids is None else ids)
        return map_spans(words, spans, (text or '').split())

    def set_bias(self, trie, types, bcfg):
        self.trie, self.btypes, self.bcfg = (trie, list(types), bcfg)

    def _procs(self, span_types):
        if getattr(self, 'trie', None) is None:
            return None
        from transformers import LogitsProcessorList
        from .bias import BiasProcessor, weights_for
        w = weights_for(set(span_types or ()) | set(getattr(self, 'always_types', ())), self.btypes, self.bcfg['inside'], self.bcfg['outside'])
        return LogitsProcessorList([BiasProcessor(self.trie, w, self.bcfg['start_bonus'], self.bcfg['cont_bonus'])])

    def _gen(self, kw, n_samples, span_types, prefix=None, return_ids=False, final=False):
        procs = self._procs(span_types)
        nb = int(self.c.get('final_beams', 1)) if final and (not prefix) and (not return_ids) else 1
        if nb > 1:
            from transformers import LogitsProcessorList
            from .bias import RowBias
            extra = {'logits_processor': LogitsProcessorList([RowBias(procs[0])])} if procs else {}
            out = self.m.generate(**kw, max_length=self._max_len(n_samples), do_sample=False, num_beams=nb, **extra)
            self.last_ids = [int(t) for t in out[0].tolist()]
            return self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()
        if prefix:
            st = self.m.config.decoder_start_token_id
            kw = dict(kw, decoder_input_ids=torch.tensor([[st] + list(prefix)], device=self.dev, dtype=torch.long))
        out = self.m.generate(**kw, max_length=max(self._max_len(n_samples), 2 + len(prefix or ())), do_sample=False, **{'logits_processor': procs} if procs else {})
        self.last_ids = [int(t) for t in out[0].tolist()]
        txt = self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()
        if return_ids:
            sp = set(self.tok.all_special_ids)
            return (txt, [int(t) for t in out[0].tolist() if int(t) not in sp])
        return txt

    @torch.no_grad()
    def decode_audio(self, wav, span_types=None, final=False):
        from .moon import generate_kwargs
        t = time.time()
        if len(wav) < 3200:
            wav = np.pad(wav, (0, 3200 - len(wav)))
        x = torch.as_tensor(wav, device=self.dev)[None].to(self.dtype)
        txt = self._gen(generate_kwargs(self.m, x), len(wav), span_types, final=final)
        self.stats['decode_ms'] += (time.time() - t) * 1000
        self.stats['utterances'] += 1
        self.stats['audio_s'] += len(wav) / SR
        return txt

    @torch.no_grad()
    def partial_states(self, h, n_samples, span_types=None, prev_ids=None, rollback=None):
        rb = self.c.get('partial_rollback_tokens', 4) if rollback is None else rollback
        pre = list(prev_ids or [])[:-rb] if prev_ids and len(prev_ids) > rb else None
        h = torch.as_tensor(np.asarray(h, np.float32), device=self.dev)[None].to(self.dtype)
        from .moon import states_output
        return self._gen(dict(encoder_outputs=states_output(self.m, h)), n_samples, span_types, prefix=pre, return_ids=True)

    @torch.no_grad()
    def decode_states(self, h, n_samples, span_types=None):
        from .moon import states_output
        t = time.time()
        h = torch.as_tensor(np.asarray(h, np.float32), device=self.dev)[None].to(self.dtype)
        txt = self._gen(dict(encoder_outputs=states_output(self.m, h)), n_samples, span_types)
        self.stats['decode_ms'] += (time.time() - t) * 1000
        self.stats['utterances'] += 1
        self.stats['audio_s'] += n_samples / SR
        return txt

    def utterance(self, f0, f1, reason, get_audio, get_states=None, span_types=None):
        pad = int(self.c['pad_s'] * SR)
        a, b = (f0 * FR, (f1 + 1) * FR)
        if self.c['decode'] in ('states', 'slice') and get_states is not None:
            h = get_states(max(0, a - pad), b + pad)
            txt = self.decode_states(h, b - a + 2 * pad, span_types) if len(h) else ''
        else:
            txt = self.decode_audio(get_audio(max(0, a - pad), b + pad), span_types, final=True)
        raw = txt
        if self.c.get('loopfix', False):
            txt = loopfix(txt)
        u = dict(start=round(f0 / 10, 1), end=round((f1 + 1) / 10, 1), f0=f0, f1=f1, text=txt, reason=reason, final=True)
        if self.tagged:
            from .tagdec import map_spans
            sp = self.tagged_spans(raw)
            u['tag_spans'] = map_spans(raw.split(), sp, txt.split()) if txt != raw else sp
        return u

    def partial(self, f0, n_total, get_audio, get_states=None, span_types=None):
        t = time.time()
        pad = int(self.c['pad_s'] * SR)
        ms0, n0, a0 = (self.stats['decode_ms'], self.stats['utterances'], self.stats['audio_s'])
        a = max(0, f0 * FR - pad)
        if self.c['decode'] in ('states', 'slice') and get_states is not None:
            h = get_states(a, n_total)
            txt = self.decode_states(h, n_total - a, span_types) if len(h) else ''
        else:
            txt = self.decode_audio(get_audio(a, n_total), span_types)
        self.stats['decode_ms'], self.stats['utterances'], self.stats['audio_s'] = (ms0, n0, a0)
        self.stats['partial_decodes'] = self.stats.get('partial_decodes', 0) + 1
        self.stats['partial_ms'] = self.stats.get('partial_ms', 0.0) + (time.time() - t) * 1000
        return txt

def norm_words(s):
    return re.sub("[^a-z0-9' ]+", ' ', (s or '').lower().replace('-', ' ')).split()

def align_spans(utts, spans):
    for u in utts:
        u['spans'] = []
    j = 0
    for si, s in enumerate(spans):
        mid = (s['start'] + s['end']) / 2
        while j < len(utts) and utts[j]['end'] < mid:
            j += 1
        k = next((i for i in (j - 1, j, j + 1) if 0 <= i < len(utts) and utts[i]['start'] - 0.3 <= mid <= utts[i]['end'] + 0.3), None)
        s['utt'] = k
        if k is None:
            continue
        utts[k]['spans'].append(si)
        uw, sw = (norm_words(utts[k]['text']), norm_words(s.get('text')))
        off = None
        if sw:
            key = sw[-1:] if s.get('method') == 'roster' else sw
            for i in range(len(uw) - len(key) + 1):
                if uw[i:i + len(key)] == key:
                    off = i
                    break
        s['in_text'] = off is not None
        if off is not None:
            s['text_offset'] = off
    return (utts, spans)
