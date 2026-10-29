import json, os, time
import numpy as np
import torch
from .ontology import Table, FINE_DOMAINS, domain_index
from .model import ArcNet
from .engine import Engine
from .words import WordStore
from .audio import load_audio, SR
from . import mst as MST
FR = 1600
STRIDE, RF = (320, 3920)

def _frame_of_enc(g):
    return int(np.ceil((g * STRIDE + RF) / SR * 10 - 1e-09)) - 1

def pool10(enc):
    n = len(enc)
    if n == 0:
        return np.zeros((0, 640), np.float32)
    f = np.array([_frame_of_enc(g) for g in range(n)])
    N10 = int(f.max()) + 1
    s = np.zeros((N10, enc.shape[1]), np.float32)
    c = np.zeros(N10, np.float32)
    E = enc.astype(np.float32)
    np.add.at(s, f, E)
    np.add.at(c, f, 1)
    mx = np.full((N10, enc.shape[1]), -np.inf, np.float32)
    np.maximum.at(mx, f, E)
    for i in range(N10):
        if c[i] == 0:
            s[i] = s[i - 1] if i else 0
            c[i] = 1
            mx[i] = mx[i - 1] if i else 0
    return np.concatenate([s / c[:, None], mx], 1).astype(np.float16).astype(np.float32)

class Pooler:

    def __init__(self):
        self.g = 0
        self.cur_f = None
        self.buf = []
        self.prev = None
        self.next_f = 0

    def _emit_upto(self, f_stop, out):
        while self.next_f < f_stop:
            if self.cur_f == self.next_f and self.buf:
                E = np.stack(self.buf)
                x = np.concatenate([E.mean(0), E.max(0)]).astype(np.float16).astype(np.float32)
                self.buf = []
            else:
                x = self.prev if self.prev is not None else np.zeros(640, np.float32)
            self.prev = x
            out.append(x)
            self.next_f += 1

    def push(self, enc):
        out = []
        for e in np.asarray(enc, np.float32):
            f = _frame_of_enc(self.g)
            self._emit_upto(f, out)
            self.cur_f = f
            self.buf.append(e)
            if _frame_of_enc(self.g + 1) != f:
                self._emit_upto(f + 1, out)
            self.g += 1
        return out

    def flush(self):
        out = []
        if self.buf:
            self._emit_upto(self.cur_f + 1, out)
        return out

def frame_db(a):
    return -120.0 if len(a) == 0 else float(20 * np.log10(np.sqrt(np.mean(a * a)) + 1e-06))

class Arc:

    def __init__(self, path, device=None, threads=None, load_decoder=True):
        device = device or os.environ.get('ARC_DEVICE', 'cpu')
        if threads:
            torch.set_num_threads(int(threads))
        self.path = path
        self.dev = torch.device(device)
        self.cfg = json.load(open(os.path.join(path, 'config.json')))
        self.table = Table(os.path.join(path, 'tags.json'))
        self.net = ArcNet(self.table, self.cfg['model'])
        from .weights import load_state, load_moonshine
        sd = load_state(path, 'model')
        miss, unexp = self.net.load_state_dict(sd, strict=False)
        miss = [k for k in miss if k not in ('trunk.delay_lut', 'head.valid') and (not k.startswith('head.cols_'))]
        if miss or unexp:
            raise RuntimeError(f'weights mismatch: missing {miss[:5]} unexpected {unexp[:5]}')
        self.net.to(self.dev).eval()
        from transformers import AutoTokenizer
        self.backbone = (self.cfg.get('backbone') or {}).get('type', 'edge')
        mp = os.path.join(path, self.cfg['encoder']['subfolder'])
        self.moon = load_moonshine(mp, decoder_only=self.backbone == 'parakeet').to(self.dev).float().eval()
        self.tok = AutoTokenizer.from_pretrained(mp) if load_decoder else None
        self.ev_head = self.dd_head = None
        self.bb = None
        if self.backbone == 'parakeet':
            from . import server as S
            bc = self.cfg['backbone']
            self.heads, hsd = S.load_heads(os.path.join(path, bc.get('adapters', 'adapters.safetensors')), self.dev)
            self.heads.pos_off = int(bc.get('pos_off', 2))
            self.bb = S.Backbone(S.fetch_backbone(bc), tap=int(bc.get('tap', 12)), device=self.dev, half=bc.get('half'))
            self.moon.model.encoder = S.Shim(self.bb, self.heads, self.moon.model.decoder.pos_emb)
            self.enc = None
        else:
            self.enc = self.moon.model.encoder
            hp = os.path.join(path, 'heads.safetensors')
            hsd = None
            if os.path.exists(hp):
                from safetensors.torch import load_file
                hsd = load_file(hp)
        self._load_aux(hsd)
        self._moons = {}
        self._mp = mp
        self.pieces = None
        if getattr(self.net, 'word', None) is not None:
            from .wordch import Pieces
            self.pieces = Pieces(os.path.join(mp, 'tokenizer.json'))
        self.domains = list(FINE_DOMAINS)
    def _load_aux(self, sd):
        if not sd:
            return
        from .events import EventHead, DirectedHead
        ev = {k[len('events.'):]: v for k, v in sd.items() if k.startswith('events.')}
        if ev:
            self.ev_head = EventHead(k=ev['c2.weight'].shape[0])
            self.ev_head.load_state_dict(ev)
            self.ev_head.to(self.dev).eval()
        dd = {k[len('device_directed.'):]: v for k, v in sd.items() if k.startswith('device_directed.')}
        if dd:
            self.dd_head = DirectedHead()
            self.dd_head.load_state_dict(dd)
            self.dd_head.to(self.dev).eval()

    def event_tracker(self, events='default'):
        if self.ev_head is None or events in (None, False, 'none', 'off'):
            return None
        from .events import EventTracker, LABELS
        ec = self.cfg.get('audio_events') or {}
        st = ec.get('status') or {}
        if events == 'all':
            on = set(LABELS)
        elif events == 'default':
            on = {l for l in LABELS if st.get(l, 'experimental') in ('stable', 'beta')}
        else:
            on = set(events)
        return EventTracker(self.ev_head, ec.get('thresholds') or {}, st, on)

    def directed(self, X):
        if self.dd_head is None or X is None or len(X) == 0:
            return None
        p = self.dd_head.prob(torch.as_tensor(np.asarray(X, np.float32), device=self.dev))
        thr = float((self.cfg.get('device_directed') or {}).get('threshold', 0.5))
        return dict(p=round(p, 4), directed=bool(p >= thr))

    @torch.no_grad()
    def _feats(self, wav, tap=False):
        if self.backbone == 'parakeet':
            from .server import ServerFeatures
            bc = self.cfg['backbone']
            src = ServerFeatures(self.bb, self.heads, chunk_s=float(bc.get('chunk_s', 30.0)), ctx_s=float(bc.get('context_s', 4.0)), tap=tap)
            xs, ts = src.feed(wav)
            x2, t2 = src.flush()
            X = np.stack(xs + x2).astype(np.float32) if xs or x2 else np.zeros((0, 640), np.float32)
            T = np.stack(ts + t2).astype(np.float32) if tap and (ts or t2) else None
            return (X.astype(np.float16).astype(np.float32), X, T)
        if not tap:
            x = pool10(MST.encode_offline(self.enc, np.asarray(wav, np.float32)))
            return (x, x, None)
        H, Ht = MST.encode_offline(self.enc, np.asarray(wav, np.float32), tap=int((self.cfg.get('audio_events') or {}).get('tap', 2)))
        x = pool10(H)
        return (x, x, pool10(Ht))

    def set_context(self, names=None, terms=None, topic=None, previous_text=None, auto_prev=False, **ctx):
        self._ctx = context_dict(names, terms, topic, previous_text, ctx)
        self._ctx_auto = bool(auto_prev)

    DECODER_FILES = {'v1e': 'decoder_v1e.safetensors', 'release': 'decoder_release.safetensors'}

    def default_decoder(self):
        return (self.cfg.get('transcription') or {}).get('decoder', 'tagged')

    def resolve_decoder(self, decoder=None):
        d = decoder or self.default_decoder()
        if d not in ('tagged', 'v1e'):
            raise ValueError('decoder must be tagged | v1e')
        from .tagdec import is_tagged_tokenizer
        if d == 'tagged' and (not (self.tok is not None and is_tagged_tokenizer(self.tok))):
            d = 'v1e'
        return d

    def moon_for(self, mode, decoder=None):
        if mode == 'moonshine_release' and (self.cfg.get('alignment') or {}).get('decoder') == 'main':
            return self.moon
        if mode == 'moonshine_release':
            key = 'release'
        elif mode == 'builtin' and self.resolve_decoder(decoder) == 'v1e' and os.path.exists(os.path.join(self._mp, self.DECODER_FILES['v1e'])):
            key = 'v1e'
        else:
            return self.moon
        if key not in self._moons:
            import copy
            from safetensors.torch import load_file
            m = copy.deepcopy(self.moon)
            m.load_state_dict(load_file(os.path.join(self._mp, self.DECODER_FILES[key])), strict=False)
            m.model.encoder = self.moon.model.encoder
            self._moons[key] = m.eval()
        return self._moons[key]

    @property
    def _moon_rel(self):
        return self._moons.get('release')
    HF_REPO = 'WhissleAI/arc-v1.1'
    BACKBONES = {'mini': 'mini', 'edge': 'mini', 'full': 'full', 'parakeet': 'full', 'server': 'full'}
    LEGACY = {'mini': 'edge', 'full': 'server'}
    MODEL_FILES = ['config.json', 'tags.json', '*.safetensors', '*/*', 'LICENSE', 'NOTICE']

    @classmethod
    def from_pretrained(cls, repo_or_dir='WhissleAI/arc-v1.1', backbone='mini', device=None, revision=None, threads=None, **kw):
        if backbone not in cls.BACKBONES:
            raise ValueError('backbone must be mini | full (aliases: edge = mini, parakeet / server = full)')
        sub = cls.BACKBONES[backbone]
        subs = [sub, cls.LEGACY[sub]]
        path = repo_or_dir
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            if revision is None and repo_or_dir == cls.HF_REPO:
                from . import __version__
                revision = f'v{__version__}'
            path = snapshot_download(repo_or_dir, revision=revision, allow_patterns=[f'{d}/{p}' for d in subs for p in cls.MODEL_FILES] + ['LICENSE', 'NOTICE'])
        for d in subs:
            if os.path.exists(os.path.join(path, d, 'config.json')):
                path = os.path.join(path, d)
                break
        return cls(path, device=device, threads=threads, **kw)

    def delay_frames(self, delay_s):
        d = int(round(float(delay_s) * 10))
        if d not in self.net.trunk.delays:
            raise ValueError(f'delay_s must be one of {[x / 10 for x in self.net.trunk.delays]}')
        return d

    @torch.no_grad()
    def features(self, wav):
        return self._feats(wav)[0]

    def tags(self, domain=None, include_experimental=False):
        from .engine import Readout
        R = Readout(self.net, self.cfg, domain_index(domain or 'unknown'), include_experimental)
        return [dict(self.table.describe(self.table.ids[i]), status=s) for i, s in sorted(R.status.items())]

    @staticmethod
    def _kb(names):
        out = []
        for c in names or []:
            out.append(dict(name=c) if isinstance(c, str) else dict(c))
        for c in out:
            c.setdefault('kind', 'other')
        return out
    TRANSCRIPT_CFG = {'builtin': dict(final_beams=4, loopfix=True, loop_guard=True), 'moonshine_release': dict(final_beams=1, loopfix=False)}
    DECODER_CFG = {'tagged': dict(tagged=True, len_mult=1.3), 'v1e': dict(tagged=False, len_mult=1.0)}

    def _transcriber(self, cfg=None, mode='builtin', decoder=None):
        from .transcribe import Transcriber
        c = dict(self.cfg.get('transcription', {}), **self.TRANSCRIPT_CFG.get(mode, {}))
        if mode == 'builtin':
            d = self.resolve_decoder(decoder)
            c.update((self.cfg.get('transcription', {}).get('decoders') or {}).get(d, self.DECODER_CFG[d]))
        else:
            c.update(tagged=False, len_mult=1.0)
        c.update(cfg or {})
        return Transcriber(self.moon_for(mode, decoder), self.tok, self.dev, torch.float32, c)

    def name_source(self, name_source=None, transcript='builtin', decoder=None, spans=None):
        if name_source in ('head', 'decoder', 'word'):
            if name_source == 'word' and getattr(self.net, 'word', None) is None:
                raise ValueError("name_source='word' needs the word channel (weights with a word channel)")
            return name_source
        if name_source not in (None, 'auto'):
            raise ValueError('name_source must be auto | decoder | head | word')
        if transcript == 'builtin' and self.resolve_decoder(decoder) == 'tagged':
            return 'decoder'
        if transcript == 'external' and spans:
            return 'decoder'
        return 'head'

    def _align(self, wav_seg, text, t0, fast=False):
        from .align import align_text
        cfg = self.cfg.get('alignment')
        if fast:
            cfg = dict(cfg or {}, _fast_dtw=True)
        return align_text(self.moon_for('moonshine_release'), self.tok, self.dev, torch.float32, wav_seg, text, t0, cfg, hop_s=self.hop_s)

    @property
    def hop_s(self):
        return 0.08 if self.backbone == 'parakeet' else None

    def transcribe(self, wav, names=None, bias_names=False, cfg=None, mode='builtin', decoder=None, context=None, auto_prev=False):
        from .transcribe import UttSegmenter
        wav = load_audio(wav) if not isinstance(wav, np.ndarray) else wav
        tr = self._transcriber(cfg, mode, decoder)
        self._set_bias(tr, names, bias_names)
        if context is not None or getattr(self, '_ctx', None):
            c = context if context is not None else self._ctx
            tr.set_context(c.get('ctx') if isinstance(c, dict) and 'ctx' in c else c, auto_prev=auto_prev or getattr(self, '_ctx_auto', False))
        elif auto_prev:
            tr.set_context(None, auto_prev=True)
        seg = UttSegmenter(tr.c)
        utts = []
        n = (len(wav) + FR - 1) // FR
        closed = []
        for f in range(n):
            closed += seg.feed(f, frame_db(wav[f * FR:(f + 1) * FR]))
        closed += seg.flush()
        get = lambda a, b: wav[max(0, a):min(len(wav), b)]
        for f0, f1, why in closed:
            u = tr.utterance(f0, f1, why, get)
            u['words'] = self._words_for(u, wav)
            u['source'] = mode
            utts.append(u)
        return utts

    def _words_for(self, u, wav):
        pad = int(self.cfg.get('transcription', {}).get('pad_s', 0.15) * SR)
        a = max(0, int(u['f0'] * FR) - pad)
        b = min(len(wav), int((u['f1'] + 1) * FR) + pad)
        W = self._align(wav[a:b], u['text'], a / SR) if u['text'] else []
        return tag_words(u, [dict(text=w['w'], start=w['start'], end=w['end'], is_final=True) for w in W])

    def _set_bias(self, tr, names, on):
        if not on or not names:
            return
        from .bias import Trie, DEFAULT
        kb = self._kb(names)
        tr.set_bias(Trie(self.tok, {'player': bias_forms(kb)}), ['player'], dict(DEFAULT))
        tr.always_types = ['player']

    @torch.no_grad()
    def predict(self, audio, domain='unknown', delay_s=0.3, names=None, transcript='builtin', words=None, include_experimental=False, calibrated=True, diarization='auto', rescore=True, bias_names=None, return_frames=True, chunk_frames=3000, name_text=None, decoder=None, spans=None, name_source=None, word_channel=True, word_arrival='rule', word_source=None, fallback='auto', strict=False, text_layer=None, events='default', context=None, auto_prev=False, kb_snap=True):
        t_start = time.time()
        wav = load_audio(audio)
        dom_i = domain_index(domain)
        d = self.delay_frames(delay_s)
        ws = WordStore()
        utts = []
        nsrc = self.name_source(name_source, transcript, decoder, spans)
        if transcript in ('builtin', 'moonshine_release'):
            utts = self.transcribe(wav, names, default_bias(bias_names, transcript), mode=transcript, decoder=decoder, context=context, auto_prev=auto_prev)
            for u in utts:
                ws.add(u['words'], 'moonshine')
        elif transcript == 'parakeet':
            from .parakeet import ParakeetFinals
            pk = ParakeetFinals(device=str(self.dev))
            utts = pk.transcribe_offline(wav, self)
            for u in utts:
                ws.add(u['words'], 'parakeet')
        elif transcript == 'external':
            if spans:
                from .tagdec import annotate_times
                words = annotate_times([dict(w, is_final=True) for w in words or []], spans)
            ws.add(words or [], 'external')
        elif transcript != 'none':
            raise ValueError('transcript must be builtin | moonshine_release | parakeet | external | none')
        ws.mark_covered(len(wav) / SR + 10)
        evt = self.event_tracker(events)
        x, xe, xt = self._feats(wav, tap=evt is not None)
        N = len(x)
        xs = np.concatenate([x[d:], np.repeat(x[-1:], min(d, N), 0)]) if N else x
        eng = Engine(self.net, self.cfg, dom_i, d, ws, kb=self._kb(names), include_experimental=include_experimental, calibrated=calibrated, diarization=diarization, rescore=rescore, name_text=name_text, name_source=nsrc, word_channel=word_channel, pieces=self.pieces, word_arrival=word_arrival, word_source=word_source or ('builtin' if transcript in ('builtin', 'moonshine_release') else 'external'), text_layer=text_layer)
        st = None
        dom = torch.tensor([dom_i], device=self.dev)
        dl = torch.tensor([d], device=self.dev)
        for s in range(0, N, chunk_frames):
            xb = torch.as_tensor(xs[s:s + chunk_frames], device=self.dev)
            h, st = self.net.trunk(xb[None], dom, dl, st)
            eng.feed(h[0], xb)
        eng.finish()
        out = self._result(eng, utts, ws, domain, d, return_frames)
        self._post(out, utts, names, xe, xt, evt, kb_snap, eng, return_frames)
        if self.fallback_mode(fallback, strict, transcript, nsrc) == 'builtin':
            fb = self.predict(wav, domain=domain, delay_s=delay_s, names=names, transcript='builtin', include_experimental=include_experimental, calibrated=calibrated, diarization=diarization, rescore=rescore, bias_names=bias_names, return_frames=False, chunk_frames=chunk_frames, decoder=decoder, word_channel=word_channel, fallback=None, events=None, kb_snap=False)
            from .textlayer import fallback_names
            add = fallback_names(out['names'], fb['names'])
            for r in add:
                r['id'] = eng._new_id('name')
            out['names'] = sorted(out['names'] + add, key=lambda r: (r['start'], r['end']))
            out['fallback'] = 'builtin'
        if transcript == 'external' and not utts:
            from .tidy import group_words, utterance_spans
            out['transcript'] = group_words([dict(w) for w in ws.finals], utterance_spans(wav, self.cfg.get('transcription')))
            if self.dd_head is not None:
                for u in out['transcript']:
                    u['device_directed'] = self.directed(xe[int(u['start'] * 10):int(u['end'] * 10) + 1])
        tidy_out(out, [w for u in out['transcript'] for w in u.get('words') or []])
        out['timing'] = dict(audio_s=round(len(wav) / SR, 2), wall_s=round(time.time() - t_start, 2))
        return out

    def _post(self, out, utts, names, xe, xt, evt, kb_snap, eng, return_frames):
        if evt is not None and xt is not None:
            evt.push(list(xe), list(xt))
            evt.step(final=True)
            out['audio_events'] = evt.out
            if return_frames:
                from .events import LABELS
                out.setdefault('frames', {})['audio_events'] = dict(labels=LABELS, probs=evt.frames())
        if self.dd_head is not None:
            for u, t in zip(utts, out['transcript']):
                t['device_directed'] = self.directed(xe[u['f0']:u['f1'] + 1])
        if kb_snap and names:
            from .kbsnap import KB, link, add_spans
            kb = KB(self._kb(names))
            for r in out['names']:
                link(r, kb)
            add = []
            for t in out['transcript']:
                add += add_spans(t.get('words') or [], out['names'] + add, kb, lambda: eng._new_id('name'))
            if add:
                out['names'] = sorted(out['names'] + add, key=lambda r: (r['start'], r['end']))

    def fallback_mode(self, fallback='auto', strict=False, transcript='external', name_source='word'):
        if strict or fallback in (None, False, 'none', 'off'):
            return None
        if fallback not in ('auto', 'builtin', True):
            raise ValueError('fallback must be auto | builtin | None')
        if transcript != 'external' or name_source != 'word':
            if fallback in ('builtin', True) and transcript != 'external':
                raise ValueError("fallback='builtin' applies to transcript='external'")
            return None
        return 'builtin'

    def _result(self, eng, utts, ws, domain, d, return_frames):
        R = eng.R
        out = dict(domain=domain, delay_s=d / 10, frame_hz=10, n_frames=eng.f, **eng.out, speakers=eng.speaker_summary(), item=eng.item_summary(), name_source='word' if eng.war else eng.nsrc)
        out['transcript'] = [utt_out(u) for u in utts] if utts else [dict(start=w['start'], end=w['end'], text=w['w']) for w in ws.finals]
        out['transcript_rewritten'] = rewrite(out['transcript'], out['names']) if utts else None
        if return_frames:
            out['frames'] = dict(tags={q: R.out_ids[q] for q in R.out_ids if R.out_ids[q]}, probs={q: np.concatenate(v) if v else np.zeros((0, len(R.out_ids.get(q, [])))) for q, v in eng.frames.items() if q in R.out_ids and R.out_ids[q]}, bio=np.concatenate(eng.frames['bio']) if eng.frames['bio'] else None, bio_tags=['bio.cough', 'bio.throat_clear', 'bio.breath'], speaker_index=np.concatenate(eng.frames['speaker_index']) if eng.frames['speaker_index'] else None)
        return out

    def stream(self, domain='unknown', delay_s=0.3, names=None, transcript='builtin', include_experimental=False, calibrated=True, diarization='auto', rescore=True, bias_names=None, partials=True, word_wait_s=None, partial_every_s=0.4, partial_window_s=7.0, background=True, worker_threads=None, live=True, name_text=None, decoder=None, name_source=None, early_commit=None, interim_source=None, word_channel=True, fallback='auto', strict=False, text_layer=None, events='default', context=None, auto_prev=False, kb_snap=True):
        return Stream(self, domain, delay_s, names, transcript, include_experimental, calibrated, diarization, rescore, bias_names, partials, word_wait_s, partial_every_s=partial_every_s, partial_window_s=partial_window_s, background=background, worker_threads=worker_threads, live=live, name_text=name_text, decoder=decoder, name_source=name_source, early_commit=early_commit, interim_source=interim_source, word_channel=word_channel, fallback=fallback, strict=strict, text_layer=text_layer, events=events, context=context, auto_prev=auto_prev, kb_snap=kb_snap)

    def early_policy(self, domain='unknown', transcript='builtin', early_commit=None, interim_source=None):
        if early_commit is False or transcript in ('none',):
            return None
        if isinstance(early_commit, dict):
            return early_commit
        ec = self.cfg.get('early_commit') or {}
        if early_commit is None and (not ec.get('enabled', False)):
            return None
        src = 'builtin' if transcript in ('builtin', 'moonshine_release') else transcript if transcript in ('parakeet',) else interim_source if interim_source in (ec.get('sources') or {}) else 'external'
        tab = (ec.get('sources') or {}).get(src) or {}
        out = {}
        for kind in ('name', 'number', 'audio'):
            t = tab.get(kind) or {}
            out[kind] = t.get(domain, t.get('unknown'))
        return out if any((v is not None for v in out.values())) else None

def extend_number(r, words):
    from .tidy import _numeric, clean_word
    from . import textlayer as TL
    W = sorted(words, key=lambda w: w['start'])
    if not W:
        return r
    txt = lambda w: str(w.get('text', w.get('w', '')))
    idx = [k for k, w in enumerate(W) if (w['start'] + w['end']) / 2 >= r['start'] - 0.02 and (w['start'] + w['end']) / 2 <= r['end'] + 0.02]
    if not idx:
        return r
    i, j = (idx[0], idx[-1])
    grew = False
    while i > 0 and _numeric(txt(W[i - 1])) and W[i]['start'] - W[i - 1]['end'] < 0.6 and clean_word(txt(W[i])).lower() in ('hundred', 'thousand', 'million', 'billion', 'and', 'point', 'to') + tuple((x for x in ('twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'))) and (not txt(W[i - 1]).rstrip().endswith(('.', ',', '?', '!'))):
        i -= 1
        grew = True
    if grew:
        ws = [txt(w) for w in W[i:j + 1]]
        r['heard'] = ' '.join(ws)
        r['start'] = round(min(r['start'], W[i]['start']), 3)
        vt, _ = TL.text_value(r.get('kind'), ws)
        if vt is not None and vt.get('v') is not None:
            r.update(value=vt.get('v'), value2=vt.get('v2'), state='read_text')
    return r

def tidy_out(out, words):
    from .tidy import tidy_name, tidy_number, keep_number, dedupe_names
    out['names'] = dedupe_names([tidy_name(r) for r in out.get('names') or []])
    out['numbers'] = [tidy_number(extend_number(r, words)) for r in out.get('numbers') or [] if keep_number(r)]
    return out

KB_KIND = {'person': 'person', 'team': 'team', 'organization': 'org', 'org': 'org', 'place': 'place', 'law': 'law', 'product': 'product'}

def context_dict(names=None, terms=None, topic=None, previous_text=None, extra=None):
    c = dict(extra or {})
    kb, ty = (list(c.pop('kb', []) or []), list(c.pop('kb_type', []) or []))
    ty = ty + [''] * (len(kb) - len(ty))
    for n in names or []:
        d = dict(name=n) if isinstance(n, str) else dict(n)
        kb.append(d['name'])
        ty.append(KB_KIND.get(d.get('kind', ''), 'term'))
    for t in terms or []:
        kb.append(str(t))
        ty.append('term')
    if kb:
        c['kb'], c['kb_type'] = (kb, ty)
    if topic:
        c['dom'] = str(topic)
    if previous_text:
        c['prev'] = str(previous_text)
    return c

def tag_words(u, events):
    if not u.get('tag_spans') or not events:
        return events
    from .tagdec import map_spans, annotate
    return annotate(events, map_spans(u['text'].split(), u['tag_spans'], [e['text'] for e in events]))

def utt_out(u, full=False):
    r = dict(u) if full else dict(start=u['start'], end=u['end'], text=u['text'], words=u.get('words', []), source=u.get('source', 'moonshine'))
    if 'tag_spans' in u:
        from .tagdec import spans_of_words
        r['tags'] = [dict(tag=s['tag'], start=round(float(s['start']), 3), end=round(float(s['end']), 3), text=' '.join((k[1] for k in s['keys']))) for s in spans_of_words([dict(w=w['text'], start=w['start'], end=w['end'], tag=w.get('tag'), tag_begin=w.get('tag_begin')) for w in u.get('words', [])])]
    return r

def bias_forms(kb):
    out = set()
    for c in kb:
        n = ' '.join(str(c.get('name', '')).split())
        if not n:
            continue
        out.add(n)
        parts = [p for p in n.split() if p.lower().strip('.') not in ('jr', 'sr', 'ii', 'iii', 'iv')] or [n]
        if len(parts) > 1:
            out.add(parts[-1])
        out.update((' '.join(str(a).split()) for a in c.get('aliases') or [] if str(a).strip()))
    return sorted(out)

def default_bias(bias_names, transcript):
    return transcript == 'builtin' if bias_names is None else bool(bias_names)

def rewrite(transcript, names):
    import re
    nz = lambda t: re.sub("[^a-z0-9']", '', t.lower())
    out = []
    links = [n for n in names if n.get('state') == 'linked' and n.get('words')]
    for u in transcript:
        ws = [dict(w) for w in u.get('words', [])]
        for n in links:
            idx = [k for k, w in enumerate(ws) if any((abs(w['start'] - x['start']) < 0.001 and w['text'] == x['w'] for x in n['words']))]
            toks = {nz(t) for t in n['entity'].split()}
            idx = [k for k in idx if nz(ws[k]['text']) in toks or len(idx) == 1 or True]
            if not idx or any((NUMWORD_RE(ws[k]['text']) for k in idx)):
                continue
            lo, hi = (min(idx), max(idx))
            while lo > 0 and nz(ws[lo - 1]['text']) in toks and ws[lo - 1]['text']:
                lo -= 1
            while hi + 1 < len(ws) and nz(ws[hi + 1]['text']) in toks and ws[hi + 1]['text']:
                hi += 1
            if any((NUMWORD_RE(ws[k]['text']) for k in range(lo, hi + 1))):
                continue
            src = ' '.join((ws[k]['text'] for k in range(lo, hi + 1)))
            ws[lo]['text'] = n['entity'] + (src[-1] if src[-1:] in ',.?!' else '')
            ws[lo]['rewritten_from'] = src
            for k in range(lo + 1, hi + 1):
                ws[k]['text'] = ''
        text = ' '.join((w['text'] for w in ws if w['text'])) if ws else u.get('text', '')
        out.append(dict(u, text=text, words=[w for w in ws if w['text']]))
    return out

def NUMWORD_RE(t):
    from .words import NUMWORD
    import re
    return bool(NUMWORD.match(re.sub('[^\\w:.\\-]', '', t)))

class _Worker:

    def __init__(self, run_job, threads):
        import collections, threading
        self.run_job, self.threads = (run_job, threads)
        self.q = collections.deque()
        self.cv = threading.Condition()
        self.busy = False
        self.stop = False
        self.partial_pending = False
        self.err = None
        self.t = threading.Thread(target=self._loop, name='arc-transcript', daemon=True)
        self.t.start()

    def submit(self, job):
        with self.cv:
            if job[0] == 'partial':
                self.partial_pending = True
            self.q.append(job)
            self.cv.notify_all()

    def _loop(self):
        if self.threads:
            torch.set_num_threads(int(self.threads))
        while True:
            with self.cv:
                while not self.q and (not self.stop):
                    self.cv.wait()
                if not self.q:
                    return
                k = next((i for i, j in enumerate(self.q) if j[0] != 'partial'), 0)
                job = self.q[k]
                del self.q[k]
                self.busy = True
            try:
                with torch.no_grad():
                    self.run_job(job)
            except BaseException:
                import traceback
                self.err = traceback.format_exc()
            finally:
                with self.cv:
                    self.busy = False
                    if job[0] == 'partial':
                        self.partial_pending = False
                    self.cv.notify_all()

    def wait(self):
        with self.cv:
            while (self.q or self.busy) and self.t.is_alive():
                self.cv.wait(0.5)

    def close(self):
        with self.cv:
            self.stop = True
            self.cv.notify_all()

def _slice(buf, lo, a, b):
    a = max(a, lo)
    b = min(b, lo + len(buf))
    return buf[a - lo:b - lo] if b > a else np.zeros(0, np.float32)

class Stream:

    def __init__(self, arc, domain, delay_s, names, transcript, include_experimental, calibrated, diarization, rescore, bias_names, partials, word_wait_s, partial_every_s=0.4, partial_window_s=7.0, background=True, worker_threads=None, live=True, name_text=None, decoder=None, name_source=None, early_commit=None, interim_source=None, word_channel=True, fallback=None, strict=False, text_layer=None, events='default', context=None, auto_prev=False, kb_snap=True):
        self.a = arc
        self.dom_i = domain_index(domain)
        self.domain = domain
        self.d = arc.delay_frames(delay_s)
        self.early = arc.early_policy(domain, transcript, early_commit, interim_source)
        self.mode = transcript
        self.decoder = decoder
        self.nsrc = arc.name_source(name_source, transcript, decoder)
        self.words = WordStore()
        ww = dict({'builtin': 17.0, 'moonshine_release': 17.0, 'external': 6.0, 'verbit': 12.0}, **(arc.cfg.get('words') or {}).get('word_wait_s') or {})
        key = transcript if transcript in ('builtin', 'moonshine_release') else interim_source if interim_source in ww else 'external'
        wait = word_wait_s if word_wait_s is not None else float(ww.get(key, 6.0))
        self.eng = Engine(arc.net, arc.cfg, self.dom_i, self.d, self.words, kb=Arc._kb(names), include_experimental=include_experimental, calibrated=calibrated, diarization=diarization, rescore=rescore, word_wait_s=wait, provisional='audio' if transcript == 'none' else 'words', name_text=name_text, name_source=self.nsrc, early=self.early, word_channel=word_channel, pieces=arc.pieces, word_arrival='live', word_source='builtin' if transcript in ('builtin', 'moonshine_release') else 'external', text_layer=text_layer)
        self.evt = arc.event_tracker(events)
        self.src = None
        self.menc = None
        if arc.backbone == 'parakeet':
            from .server import ServerFeatures
            bc = arc.cfg['backbone']
            self.src = ServerFeatures(arc.bb, arc.heads, chunk_s=float(bc.get('stream_chunk_s', 10.0)), ctx_s=float(bc.get('stream_context_s', 2.0)), tap=self.evt is not None)
        else:
            self.menc = MST.MSTStreamEncoder(arc.enc, tap=int((arc.cfg.get('audio_events') or {}).get('tap', 2)) if self.evt is not None else None)
        self.pool = Pooler()
        self.tpool = Pooler()
        self.xe = []
        self.xe0 = 0
        self.dd_wait = []
        self.dd_out = []
        self.ev_sent = 0
        self.dd_sent = 0
        self.xseg = None
        self.xspans = []
        self.xfdb = 0
        self.xused = set()
        self.emitted = []
        if transcript == 'external':
            from .transcribe import UttSegmenter
            self.xseg = UttSegmenter(arc.cfg.get('transcription'))
        self.kb = None
        if kb_snap and names:
            from .kbsnap import KB
            self.kb = KB(Arc._kb(names))
        self.xq = []
        self.n_x = 0
        self.st = None
        self.last_x = None
        self.domt = torch.tensor([self.dom_i], device=arc.dev)
        self.dlt = torch.tensor([self.d], device=arc.dev)
        self.audio = np.zeros(0, np.float32)
        self.a0 = 0
        self.total = 0
        self.utts = []
        self.partial = None
        self.partials = partials and transcript in ('builtin', 'moonshine_release', 'parakeet') and (arc.backbone != 'parakeet' or arc.dev.type == 'cuda')
        self.partial_every = float(partial_every_s)
        self.partial_win = float(partial_window_s)
        self.live_on = live
        self.tr = self.seg = None
        self.fdb = 0
        self.done = []
        self.partial_out = None
        self.pwords = (None, [])
        self.closed_f1 = -1
        self.cov_sent = 0.0
        if transcript in ('builtin', 'moonshine_release', 'parakeet'):
            from .transcribe import UttSegmenter
            tm = 'moonshine_release' if transcript == 'moonshine_release' else 'builtin'
            self.tr = arc._transcriber(mode=tm, decoder=decoder if tm == 'builtin' else None)
            arc._set_bias(self.tr, names, default_bias(bias_names, transcript))
            if context is not None or getattr(arc, '_ctx', None) or auto_prev:
                c = context if context is not None else getattr(arc, '_ctx', None)
                self.tr.set_context(c, auto_prev=auto_prev or getattr(arc, '_ctx_auto', False))
            self.seg = UttSegmenter(self.tr.c)
            self.last_partial = -1000000000.0
        self.pk = None
        if transcript == 'parakeet':
            from .parakeet import ParakeetFinals
            self.pk = ParakeetFinals(device=str(arc.dev), background=True)
        elif transcript not in ('builtin', 'moonshine_release', 'external', 'none'):
            raise ValueError('transcript must be builtin | moonshine_release | parakeet | external | none')
        self.worker = None
        if self.tr is not None and background:
            self.worker = _Worker(self._job, worker_threads or 1)
            import sys
            self._swi = sys.getswitchinterval()
            sys.setswitchinterval(min(self._swi, float(arc.cfg.get('stream', {}).get('switch_interval_s', 0.0005))))
            self.eng.cov_submitted = lambda: self.cov_sent
        self.sent = {k: 0 for k in ('spans', 'events', 'segments', 'utterances', 'names', 'numbers', 'diarization', 'bio_events', 'updates')}
        self.n_frames_out = 0
        self.timing = []
        self.vroles = None
        self._vr_w = 0
        self.fb = None
        self.fb_q = []
        if arc.fallback_mode(fallback, strict, transcript, self.nsrc) == 'builtin':
            self.fb = Stream(arc, domain, delay_s, names, 'builtin', include_experimental, calibrated, diarization, rescore, bias_names, partials, None, partial_every_s=partial_every_s, partial_window_s=partial_window_s, background=background, worker_threads=worker_threads, live=False, name_text=name_text, decoder=decoder, name_source=None, early_commit=False, word_channel=word_channel)
            self.fb_wait = int(round(10 * wait)) + 20

    def add_words(self, events, spans=None):
        if spans:
            from .tagdec import annotate_times
            events = annotate_times(events, spans)
        return self.words.add(events, 'external')

    def set_partial_words(self, events, upto, spans=None):
        if spans:
            from .tagdec import annotate_times
            events = annotate_times(events, spans)
        self.words.set_partial(events, upto)

    def set_names(self, names):
        self.eng.set_names(Arc._kb(names))

    def set_context(self, ctx=None, auto_prev=False, max_tokens=96, **kw):
        if self.tr is not None:
            self.tr.set_context(ctx, auto_prev=auto_prev, max_tokens=max_tokens, **kw)
        if self.fb is not None:
            self.fb.set_names(names)

    def fused_words(self):
        wf = self.eng.wfeed
        av = dict(zip(wf.keys, wf.av)) if wf is not None else {}
        oe = dict(zip(wf.keys, wf.oe)) if wf is not None else {}
        with self.words.lock:
            fin = list(self.words.finals)
        out = []
        for w in fin:
            k = (w['start'], w['end'], w['w'])
            e = dict(text=w['w'], start=w['start'], end=w['end'], is_final=True, arrive_f=int(av.get(k, 10 ** 9)), source=w.get('src') or 'external')
            if k in oe:
                e['oe_f'] = int(oe[k])
            if w.get('tag'):
                e['tag'] = w['tag']
                e['tag_begin'] = bool(w.get('tag_begin'))
            out.append(e)
        return out

    def add_voice_segment(self, voice, start, end):
        if self.vroles is None:
            from .roles import VoiceRoles
            self.vroles = VoiceRoles(self.eng, self.domain if self.domain in self.a.domains else 'unknown', self.a.cfg.get('roles'))
        self.vroles.add_segment(voice, start, end)

    def voice_roles(self):
        if self.vroles is None:
            return {}
        self._roles_step()
        return self.vroles.roles()

    def _roles_step(self):
        if self.vroles is None:
            return
        with self.words.lock:
            fin = self.words.finals[self._vr_w:]
            self._vr_w = len(self.words.finals)
        if fin:
            self.vroles.words(fin)
        self.vroles.step()

    def _get_audio(self, a, b):
        a = max(a, self.a0)
        b = min(b, self.total)
        return self.audio[a - self.a0:b - self.a0] if b > a else np.zeros(0, np.float32)

    def _check_worker(self):
        if self.worker is not None and self.worker.err:
            raise RuntimeError('transcript worker failed:\n' + self.worker.err)

    @torch.no_grad()
    def feed(self, pcm):
        t0 = time.perf_counter()
        self._check_worker()
        pcm = np.asarray(pcm, np.float32).reshape(-1)
        self.audio = np.concatenate([self.audio, pcm])
        self.total += len(pcm)
        self._xseg_step(final=False)
        xs = self._features(pcm, final=False)
        self._advance(xs, final=False)
        if self.seg is not None:
            self._segment_step(final=False)
        keep = 30 * SR
        if self.total - self.a0 > keep + SR and (self.seg is None or not self.seg.cur or self.seg.cur[0][0] * FR > self.total - keep):
            cut = self.total - keep - self.a0
            self.audio = self.audio[cut:]
            self.a0 += cut
        self._join()
        self._roles_step()
        if self.fb is not None:
            self.fb_q += [r for r in self.fb.feed(pcm)['names'] if r.get('state') == 'linked']
            self._fallback_step(final=False)
        out = self._collect()
        if self.live_on:
            out['live'] = self.eng.live()
        if self.vroles is not None:
            out['voice_roles'] = self.vroles.roles()
        self.timing.append((len(pcm) / SR, time.perf_counter() - t0))
        return out

    @torch.no_grad()
    def flush(self):
        self._xseg_step(final=True)
        xs = self._features(None, final=True)
        self._advance(xs, final=True)
        if self.seg is not None:
            self._segment_step(final=True)
        if self.worker is not None:
            self.worker.wait()
            self.worker.close()
            import sys
            sys.setswitchinterval(self._swi)
            self._check_worker()
        self._join()
        self._xgroup(final=True)
        self._dd_step(final=True)
        if self.pk is not None:
            self.utts += self.pk.drain(self.words)
        self.words.mark_covered(self.total / SR + 10)
        self.eng.finish()
        if self.fb is not None:
            self.fb_q += [r for r in self.fb.flush()['names'] if r.get('state') == 'linked']
            self._fallback_step(final=True)
        out = self._collect()
        if self.live_on:
            out['live'] = self.eng.live()
        out['speakers'] = self.eng.speaker_summary()
        out['item'] = self.eng.item_summary()
        if self.vroles is not None:
            self._roles_step()
            out['voice_roles'] = self.vroles.roles()
        if self.mode in ('builtin', 'moonshine_release', 'parakeet'):
            out['transcript_rewritten'] = rewrite(self.utts, self.eng.out['names'])
        return out

    def _features(self, pcm, final):
        if self.src is not None:
            xe, ts = self.src.flush() if final else self.src.feed(pcm)
            xs = [x.astype(np.float16).astype(np.float32) for x in xe]
        else:
            H = self.menc.flush() if final else self.menc.feed(pcm)
            T = self.menc.tap_out
            xs = self.pool.push(H.float().cpu().numpy())
            if final:
                xs += self.pool.flush()
            ts = []
            if self.evt is not None:
                ts = self.tpool.push(T.float().cpu().numpy()) if T is not None else []
                if final:
                    ts += self.tpool.flush()
            xe = xs
        self.xe += list(xe)
        if self.evt is not None:
            self.evt.push(list(xe), list(ts))
            self.evt.step(final=final)
        return xs

    def _xseg_step(self, final):
        if self.xseg is None:
            return
        nf = self.total // FR if not final else (self.total + FR - 1) // FR
        while self.xfdb < nf:
            f = self.xfdb
            for f0, f1, _ in self.xseg.feed(f, frame_db(self._get_audio(f * FR, (f + 1) * FR))):
                self.xspans.append((f0, f1))
            self.xfdb += 1
        if final:
            self.xspans += [(f0, f1) for f0, f1, _ in self.xseg.flush()]

    def _xgroup(self, final):
        if self.xseg is None:
            return
        from .tidy import group_words
        with self.words.lock:
            fin = [w for w in self.words.finals if (w['start'], w['w']) not in self.xused]
            cov = self.words.covered
        keep = []
        for f0, f1 in self.xspans:
            s, e = (f0 / 10, (f1 + 1) / 10)
            if not final and cov < e + 0.3:
                keep.append((f0, f1))
                continue
            ws = [w for w in fin if s - 0.3 <= (w['start'] + w['end']) / 2 <= e + 0.3 and (w['start'], w['w']) not in self.xused]
            for w in ws:
                self.xused.add((w['start'], w['w']))
            for u in group_words([dict(w) for w in ws], [(s, e)]):
                u.update(f0=int(u['start'] * 10), f1=max(int(u['start'] * 10), int(u['end'] * 10) - 1))
                self.utts.append(u)
                if self.a.dd_head is not None:
                    self.dd_wait.append(u)
        self.xspans = keep
        if final:
            rest = [w for w in fin if (w['start'], w['w']) not in self.xused]
            for u in group_words([dict(w) for w in rest], []):
                u.update(f0=int(u['start'] * 10), f1=max(int(u['start'] * 10), int(u['end'] * 10) - 1))
                self.utts.append(u)
            self.utts.sort(key=lambda u: u['start'])

    def _dd_step(self, final=False):
        if self.a.dd_head is None:
            return
        n = self.xe0 + len(self.xe)
        keep = []
        for u in self.dd_wait:
            if u['f1'] + 1 <= n or final:
                lo, hi = (max(u['f0'], self.xe0), min(u['f1'] + 1, n))
                u['device_directed'] = self.a.directed(self.xe[lo - self.xe0:hi - self.xe0]) if hi > lo else None
                if u['device_directed'] is not None:
                    self.dd_out.append(dict(start=u['start'], end=u['end'], **u['device_directed']))
            else:
                keep.append(u)
        self.dd_wait = keep
        lo = min([u['f0'] for u in self.dd_wait] + [n - 300])
        cut = lo - self.xe0
        if cut > 0:
            self.xe = self.xe[cut:]
            self.xe0 += cut

    def _fallback_step(self, final):
        from .textlayer import fallback_names
        eng = self.eng
        keep = []
        with self.words.lock:
            for r in self.fb_q:
                fe = int(r['end'] * 10)
                done = final or eng.f > fe + self.fb_wait or (self.words.covered >= r['end'] + 0.5 and eng.f > fe + eng.war_wait + 5 and (not any((pd['kind'] == 'name' and pd['span']['start'] <= r['end'] + 0.5 for pd in eng.dec_pending))) and (not (eng.war_cur is not None and eng.wfeed is not None and (eng.wfeed.keys[eng.war_cur['j'][0]][0] <= r['end'] + 0.5))))
                if not done:
                    keep.append(r)
                    continue
                for x in fallback_names([y for y in eng.out['names'][-300:] if y['end'] > r['start'] - 60], [r]):
                    x['id'] = eng._new_id('name')
                    x.pop('revises', None)
                    eng.out['names'].append(x)
        self.fb_q = keep

    def _advance(self, xs, final):
        rows = []
        for x in xs:
            self.last_x = x
            k = self.n_x
            self.n_x += 1
            if k >= self.d:
                rows.append(x)
        if final and self.last_x is not None:
            rows += [self.last_x] * min(self.d, self.n_x)
        if not rows:
            return
        xb = torch.as_tensor(np.stack(rows), device=self.a.dev)
        h, self.st = self.a.net.trunk.step(xb[None], self.domt, self.dlt, self.st)
        self.eng.feed(h[0], xb)

    def _submit(self, job):
        if self.worker is not None:
            self.worker.submit(job)
        else:
            self._job(job)

    def _segment_step(self, final):
        nf = self.total // FR if not final else (self.total + FR - 1) // FR
        closed = []
        while self.fdb < nf:
            f = self.fdb
            closed += self.seg.feed(f, frame_db(self._get_audio(f * FR, (f + 1) * FR)))
            self.fdb += 1
        if final:
            closed += self.seg.flush()
        pad = int(self.tr.c['pad_s'] * SR)
        for f0, f1, why in closed:
            self.closed_f1 = max(self.closed_f1, f1)
            if self.pk is not None:
                a = max(0, f0 * FR - 2400)
                self.pk.submit(self._get_audio(a, (f1 + 1) * FR + 2400), a / SR, f0, f1)
                self.words.clear_interim((f1 + 1) / 10 + 0.5)
                self.partial_out = None
                continue
            lo = max(0, f0 * FR - pad, self.a0)
            self._submit(('final', f0, f1, why, lo, self._get_audio(lo, (f1 + 1) * FR + pad).copy()))
        if self.pk is not None:
            for u in self.pk.poll(self.words):
                self.utts.append(u)
        else:
            cov = self.seg.cur[0][0] / 10 if self.seg.cur else self.fdb / 10
            if cov > self.cov_sent or not self.cov_sent:
                self.cov_sent = cov
                self._submit(('cov', cov))
        if self.partials and (not final) and self.seg.cur:
            sp = self.seg.open_span()
            now = self.total / SR
            if sp and now - self.last_partial >= self.partial_every - 1e-09 and (not (self.worker is not None and self.worker.partial_pending)):
                self.last_partial = now
                u0 = max(0, sp[0] * FR - pad)
                ws = max(u0, self.total - int(self.partial_win * SR), self.a0)
                self._submit(('partial', sp[0], u0, ws, self._get_audio(ws, self.total).copy(), self.total))

    def _job(self, job):
        kind = job[0]
        if kind == 'cov':
            self.words.mark_covered(job[1])
        elif kind == 'final':
            _, f0, f1, why, lo, buf = job
            get = lambda a, b: _slice(buf, lo, a, b)
            u = self.tr.utterance(f0, f1, why, get)
            u['words'] = tag_words(u, self.a._words_for_stream(u, get)) if u['text'] else []
            u['source'] = self.mode
            with self.words.lock:
                self.words.add(u['words'], 'moonshine')
                self.words.clear_interim((f1 + 1) / 10 + 0.5)
            po = self.partial_out
            if po is not None and po['f0'] <= f1:
                self.partial_out = None
            self.done.append(u)
        elif kind == 'partial':
            _, f0, u0, ws, buf, n_total = job
            if f0 <= self.closed_f1 or not len(buf):
                return
            t = time.time()
            st = dict(self.tr.stats)
            from .moon import encoder_out, states_output
            from .align import align_eo
            x = buf if len(buf) >= 3200 else np.pad(buf, (0, 3200 - len(buf)))
            eo = encoder_out(self.tr.m, torch.as_tensor(x, device=self.a.dev)[None].float())
            txt = self.tr._gen(dict(encoder_outputs=states_output(self.tr.m, eo.last_hidden_state)), len(x), None)
            pu = dict(text=txt, tag_spans=self.tr.tagged_spans(txt)) if self.tr.tagged else dict(text=txt)
            if self.tr.c.get('loopfix', False):
                from .transcribe import loopfix
                txt = loopfix(txt)
                if pu.get('tag_spans') and txt != pu['text']:
                    from .tagdec import map_spans
                    pu = dict(text=txt, tag_spans=map_spans(pu['text'].split(), pu['tag_spans'], txt.split()))
            t_dec = time.time()
            W = align_eo(self.a.moon_for('moonshine_release'), self.a.tok, self.a.dev, eo, len(x), txt, ws / SR, dict(self.a.cfg.get('alignment') or {}, _fast_dtw=True), hop_s=self.a.hop_s) if txt and len(buf) >= 800 else []
            self.tr.stats.update({k: st[k] for k in ('decode_ms', 'utterances', 'audio_s')})
            self.tr.stats['partial_decodes'] = self.tr.stats.get('partial_decodes', 0) + 1
            self.tr.stats['partial_ms'] = self.tr.stats.get('partial_ms', 0.0) + (time.time() - t) * 1000
            self.tr.stats['partial_align_ms'] = self.tr.stats.get('partial_align_ms', 0.0) + (time.time() - t_dec) * 1000
            W = tag_words(pu, [dict(text=w['w'], start=w['start'], end=w['end'], is_final=False, source='partial') for w in W])
            prev = self.pwords[1] if self.pwords[0] == f0 else []
            if ws > u0 and prev:
                cut = ws / SR + 1.0
                W = [w for w in prev if (w['start'] + w['end']) / 2 < cut] + [w for w in W if (w['start'] + w['end']) / 2 >= cut]
            self.pwords = (f0, W)
            if f0 <= self.closed_f1:
                return
            self.words.set_partial(W, n_total / SR)
            self.partial_out = dict(start=round(f0 / 10, 1), end=round(n_total / SR, 2), text=' '.join((w['text'] for w in W)) if W else txt, words=[{k: w[k] for k in ('text', 'start', 'end', 'tag', 'tag_begin') if k in w} for w in W], final=False, f0=f0)

    def _join(self):
        while self.done:
            u = self.done.pop(0)
            self.utts.append(u)
            if self.a.dd_head is not None:
                self.dd_wait.append(u)
            if self.kb:
                from .kbsnap import add_spans
                with self.words.lock:
                    recent = [r for r in self.eng.out['names'][-300:] if r['end'] > u['start'] - 1]
                    self.eng.out['names'] += add_spans(u.get('words') or [], recent, self.kb, lambda: self.eng._new_id('name'))
        self._xgroup(final=False)
        self._dd_step()
        po = self.partial_out
        self.partial = None if po is None else {k: v for k, v in po.items() if k != 'f0'}

    def _collect(self):
        e = self.eng.out
        out = {}
        for k, n in self.sent.items():
            out[k] = e[k][n:]
            self.sent[k] = len(e[k])
        if self.kb:
            from .kbsnap import link
            for r in out['names']:
                link(r, self.kb)
            for x in out['updates']:
                r = x.get('record') if isinstance(x, dict) else None
                if isinstance(r, dict) and 'heard' in r and str(r.get('tag', '')).startswith('entity.mention'):
                    link(r, self.kb)
        from .tidy import tidy_name, tidy_number, keep_number, same_mention
        with self.words.lock:
            fw = list(self.words.finals[-400:])
        nn = []
        for r in out['names']:
            tidy_name(r)
            if any((same_mention(r, o) for o in self.emitted)):
                continue
            self.emitted = (self.emitted + [r])[-300:]
            nn.append(r)
        out['names'] = nn
        out['numbers'] = [tidy_number(extend_number(r, fw)) for r in out['numbers'] if keep_number(r)]
        for x in out['updates']:
            r = x.get('record') if isinstance(x, dict) else None
            if isinstance(r, dict) and 'heard' in r:
                if x.get('type') == 'name':
                    tidy_name(r)
                elif x.get('type') == 'number':
                    tidy_number(r)
        if self.evt is not None:
            out['audio_events'] = self.evt.out[self.ev_sent:]
            self.ev_sent = len(self.evt.out)
        if self.a.dd_head is not None:
            out['device_directed'] = self.dd_out[self.dd_sent:]
            self.dd_sent = len(self.dd_out)
        R = self.eng.R
        fr = {}
        for q, v in self.eng.frames.items():
            if q in R.out_ids and R.out_ids[q]:
                fr[q] = v
        nf = self.eng.f
        if nf > self.n_frames_out:
            out['frames'] = dict(start_frame=self.n_frames_out, n=nf - self.n_frames_out, probs={q: np.concatenate(v)[self.n_frames_out:] for q, v in fr.items() if v})
            self.n_frames_out = nf
        out['transcript'] = [utt_out(u, True) if 'tag_spans' in u else u for u in self.utts[getattr(self, '_utt_sent', 0):]]
        self._utt_sent = len(self.utts)
        out['transcript_partial'] = self.partial
        out['time_s'] = round(self.total / SR, 3)
        out['decided_until_s'] = round(self.eng.f / 10, 1)
        return out

    def latency_report(self):
        if not self.timing:
            return {}
        ms = np.array([t for _, t in self.timing]) * 1000
        au = sum((a for a, _ in self.timing))
        return dict(chunks=len(ms), ms_p50=round(float(np.median(ms)), 2), ms_p90=round(float(np.percentile(ms, 90)), 2), ms_p99=round(float(np.percentile(ms, 99)), 2), ms_max=round(float(ms.max()), 2), rtf=round(float(ms.sum() / 1000 / max(au, 1e-09)), 4))

def _words_for_stream(self, u, get):
    pad = int(self.cfg.get('transcription', {}).get('pad_s', 0.15) * SR)
    a = max(0, int(u['f0'] * FR) - pad)
    b = int((u['f1'] + 1) * FR) + pad
    W = self._align(get(a, b), u['text'], a / SR)
    return [dict(text=w['w'], start=w['start'], end=w['end'], is_final=True) for w in W]
Arc._words_for_stream = _words_for_stream
