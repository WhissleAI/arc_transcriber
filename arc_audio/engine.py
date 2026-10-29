import math
import numpy as np
import torch
from .model import zero_carry, NEG, C_KINDS, MENTION_KINDS, encode_names
from .ontology import QUERIES, POOLED, FINE_DOMAINS
from .words import own_end, NUMWORD
from . import numvalue as NV
from . import numlib as NL
from . import textlayer as TL
SPK_SWITCH = 5
UTT_PAUSE = 5
SEG_SWITCH = 10
FILLERS = {'uh', 'um', 'erm', 'er', 'ah', 'eh', 'hmm', 'mm', 'uhm', 'mhm', 'huh'}
BACKCHANNEL = {'mmhmm', 'mhmm', 'uhhuh', 'uhuh', 'mmm', 'hm', 'ok', 'okay', 'yeah', 'yep', 'nope'}
STOP = {'the', 'and', 'for', 'but', 'with', 'out', 'off', 'from', 'that', 'this', 'has', 'had', 'have', 'was', 'were', 'are', 'his', 'her', 'its', 'our', 'your', 'their', 'they', 'them', 'she', 'him', 'who', 'what', 'when', 'then', 'than', 'there', 'here', 'not', 'now', 'all', 'any', 'can', 'will', 'just', 'into', 'onto', 'over', 'very', 'yes', 'yeah', 'okay', 'right', 'well', 'you', 'one', 'two'}
DISPLAY_STOP = STOP | {'a', 'an', 'of', 'to', 'in', 'on', 'at', 'by', 'is', 'be', 'been', 'it', 'he', 'we', 'i', 'me', 'my', 'us', 'so', 'do', 'did', 'does', 'if', 'or', 'as', 'up', 'no', 'oh', 'go', 'get', 'got', 'let', 'lets', "let's", "that's", "it's", "he's", "she's", "i'm", "you're", "we're", "they're", "there's", "what's", "don't", "can't", "didn't", "isn't", 'about', 'after', 'again', 'back', 'down', 'good', 'great', 'big', 'little', 'more', 'most', 'much', 'some', 'such', 'these', 'those', 'too', 'way', 'where', 'which', 'while', 'why', 'how', 'would', 'could', 'should', 'being', 'because', 'through', 'again', 'game', 'guys', 'man', 'time', 'thing', 'things', 'people', 'really', 'think', 'know', 'like', 'going', 'gonna', 'want', 'look', 'see', 'said', 'says', 'say', 'make', 'made', 'take', 'come', 'came', 'put', 'nice', 'play', 'shot', 'ball', 'goal', 'team', 'point', 'points', 'year', 'years', 'today', 'tonight', 'now'}
SPAN_GAP = 1
RING = 1200

def _w(t):
    import re
    return re.sub("[^a-z0-9']", '', t.lower())

def is_filler(t):
    return _w(t) in FILLERS

def is_content(t):
    w = _w(t)
    return len(w) >= 3 and w.isalpha() and (w not in FILLERS) and (w not in STOP)

def numeric_word(t):
    try:
        return bool(NL.tokens_of([t])[0])
    except Exception:
        return False

def connector(t):
    try:
        tk = NL.tokens_of([t])[0]
    except Exception:
        return True
    return all((x in ('and', 'SEP', 'a', 'point', 'NEG') for x in tk))

def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))

class Readout:

    def __init__(self, model, cfg, dom_i, include_experimental=False, calibrated=True):
        tab = model.head.table
        self.tab = tab
        self.dom_i = dom_i
        self.la = np.asarray(cfg['prior']['la'], np.float32)[dom_i] if calibrated else np.zeros(tab.n, np.float32)
        emit = np.asarray(cfg['emit']['mask'], bool)[dom_i]
        st = cfg['tag_status']
        self.status = {}
        for i in tab.trained:
            s = st.get(tab.ids[i], 'shipped')
            if s.startswith('not_shipped') or not tab.valid[dom_i, i]:
                continue
            if not emit[i]:
                s = 'experimental:not_supported_in_domain'
            if s == 'shipped' or include_experimental:
                self.status[i] = s
        self.cols = {q: list(tab.cols[q]) for q in QUERIES}
        self.out_cols = {q: [j for j, i in enumerate(self.cols[q]) if i in self.status] for q in QUERIES}
        self.out_ids = {q: [tab.ids[self.cols[q][j]] for j in self.out_cols[q]] for q in QUERIES}
        self.grp_loc = {}
        for q in QUERIES:
            if q == 'span_end':
                continue
            for g, mem in tab.members.items():
                loc = [self.cols[q].index(i) for i in mem if i in self.cols[q]]
                if loc:
                    self.grp_loc.setdefault(q, []).append((g, loc))
        self.thr = cfg.get('thresholds', {})
        self.dom = FINE_DOMAINS[dom_i]

    def probs(self, q, z):
        c = self.cols[q]
        la = self.la[c][None]
        valid = z > NEG / 2
        p = _sig(np.where(valid, z + la, -50.0))
        for g, loc in self.grp_loc.get(q, []):
            zg = np.where(valid[:, loc], z[:, loc] + la[:, loc], -np.inf)
            m = zg.max(1, keepdims=True)
            ok = np.isfinite(m)
            e = np.where(np.isfinite(zg), np.exp(zg - np.where(ok, m, 0)), 0.0)
            s = e.sum(1, keepdims=True)
            p[:, loc] = np.where(ok, e / np.maximum(s, 1e-30), 0.0)
        return p.astype(np.float32)

    def threshold(self, tag_id, q, default=0.5):
        return float(self.thr.get(f'{tag_id}@{q}', {}).get(self.dom, default))

class Engine:

    def __init__(self, model, cfg, dom_i, delay, words, kb=None, include_experimental=False, calibrated=True, diarization='slots', word_wait_s=6.0, rescore=True, provisional=None, name_text=None, name_source='head', early=None, word_channel=True, pieces=None, word_arrival='live', word_source='builtin', text_layer=None):
        self.m, self.cfg, self.dom_i, self.d = (model, cfg, dom_i, int(delay))
        self.R = Readout(model, cfg, dom_i, include_experimental, calibrated)
        tab = self.tab = model.head.table
        dev = next(model.parameters()).device
        self.dev = dev
        self.domt = torch.tensor([dom_i], device=dev)
        with torch.no_grad():
            self.K = model.head.keys()
        self.carry = zero_carry(1, model.trunk.cfg['d_model'], dev)
        self.words = words
        self.word_wait = float(word_wait_s)
        self.rescore = rescore
        self.inc_exp = include_experimental
        self.f = 0
        self.spk_cur, self.spk_pend, self.spk_n = (-1, None, 0)
        self.u_in, self.u_sil, self.u_spk = (False, 0, -1)
        self.in_utt, self.utt0, self.utt_last, self.utt_spk, self.utt_p = (False, None, None, -1, None)
        self.out = dict(spans=[], events=[], segments=[], utterances=[], names=[], numbers=[], diarization=[], bio_events=[], updates=[])
        self.prov = provisional
        self.early = early if early and provisional == 'words' else None
        import os as _os
        self.diag = self.early is not None or bool((cfg.get('early_commit') or {}).get('diag')) or _os.environ.get('ARC_EARLY_DIAG') == '1'
        self.n_id = {'name': 0, 'number': 0}
        self.last = None
        self.last_spk = -1
        self._live_cache = {}
        nt = dict(mode='gate', snap_frac=0.6, sim=0.85)
        nt.update({k: v for k, v in (cfg.get('name_text') or {}).items() if k != 'note'})
        if name_text is not None:
            nt['mode'] = name_text
        if nt['mode'] not in ('overlap', 'snap', 'gate'):
            raise ValueError('name_text must be overlap | snap | gate')
        self.nt = nt
        from .nametext import ListIndex
        self.lidx = ListIndex(kb or [])
        self.cov_submitted = None
        if name_source not in ('head', 'decoder', 'word'):
            raise ValueError('name_source must be head | decoder | word')
        self.war = name_source == 'word'
        self.nsrc = 'decoder' if self.war else name_source
        wc = cfg.get('word_spans') or {}
        self.war_thr = dict(wc.get('thr') or {})
        self.war_wait = int(wc.get('wait_frames', 15))
        self.war_k = int(wc.get('k', 1))
        self.war_j = 0
        self.war_cur = None
        xt = dict(exact_names=True, text_numbers=True)
        xt.update({k: v for k, v in (cfg.get('external_text') or {}).items() if k != 'note'})
        if text_layer is not None:
            xt.update(text_layer if isinstance(text_layer, dict) else dict(exact_names=bool(text_layer), text_numbers=bool(text_layer)))
        self.xt_names = self.war and bool(xt.get('exact_names'))
        self.xt_nums = self.war and bool(xt.get('text_numbers'))
        self.exn = TL.ExactNames(kb or [])
        self.tx_seen = set()
        self.tx_spans = []
        self.txn_i = 0
        self.dec_pending, self.dec_prov = ([], [])
        self._dp_ver, self._dp_wait = (None, False)
        self.speakers, self.item = ({}, None)
        self.frames = {q: [] for q in QUERIES}
        self.frames['bio'] = []
        self.frames['speaker_index'] = []
        self.open_spans = {}
        self.ev_last = {}
        self.seg = {}
        self.diar_seg = None
        self.ring_h = []
        self.ring_f0 = 0
        self.ring_a = []
        self.wfeed = None
        self.word_arrival = word_arrival
        if word_channel and getattr(model, 'word', None) is not None and (pieces is not None):
            from .wordch import WordFeed
            self.wfeed = WordFeed(model.word, pieces, dev, self.d)
        pc = cfg.get('ptr') or {}
        self.ptr_src = dict(pc.get('src_by_source') or {'moonshine': 0, 'partial': 0, 'external': 0, 'parakeet': 0})
        self.wc_src = dict((cfg.get('word_channel') or {}).get('src_by_source') or {'moonshine': 1, 'partial': 1, 'external': 0, 'parakeet': 0})
        self.ct = None
        npc = (cfg.get('name_policy') or {}).get(word_source if word_source in ('builtin', 'external') else 'external') or {}
        self.npol = dict(npc.get('*') or {}, **npc.get(FINE_DOMAINS[dom_i]) or {})
        self.word_source = word_source
        self.pending = []
        self.kb = kb or []
        self.E = None
        if self.kb and model.ent is not None:
            self.set_names(self.kb)
        from .rescorer import NameRescorer
        self.resc = NameRescorer(self.kb, self._resc_params()) if self.kb else None
        nm = cfg['numerals']
        self.unit_vocab = nm['unit_vocab']
        self.nwt = nm['nwt_vocab']
        self.bio_names = ('cough', 'throat_clear', 'breath')
        bc = cfg.get('bio_events') or {}
        self.bio_on = FINE_DOMAINS[dom_i] in bc.get('domains', FINE_DOMAINS) or include_experimental
        dcfg = cfg['diarization']
        if diarization in (None, 'auto'):
            diarization = (dcfg.get('by_domain') or {}).get(FINE_DOMAINS[dom_i], dcfg.get('default', 'slots'))
        if diarization not in ('slots', 'fused', 'voice'):
            raise ValueError('diarization must be auto | slots | fused | voice')
        self.diar_mode = diarization
        self.bank = None
        if diarization in ('fused', 'voice'):
            from .trackbank import TrackBank
            fb = dcfg['fused_bank']
            beta = 0.0 if diarization == 'voice' else float(fb['beta_by_domain'].get(FINE_DOMAINS[dom_i], fb['beta_default']))
            self.bank = TrackBank(beta=beta, tau_new=fb['tau_new'], tau_upd=fb['tau_upd'], confirm=fb['confirm'], switch=fb['switch'], smooth=fb['smooth'])

    @torch.no_grad()
    def set_names(self, kb):
        self.kb = kb
        ids = torch.as_tensor(encode_names([c['name'] for c in kb]), device=self.dev)[None]
        kinds = torch.as_tensor([C_KINDS.index(c.get('kind', 'other')) if c.get('kind', 'other') in C_KINDS else 9 for c in kb], device=self.dev)[None]
        with torch.no_grad():
            self.E = self.m.ent.names(ids, kinds)
        self.ckind = kinds[0].cpu().numpy()
        if getattr(self.m.ent, 'sim_w', None) is not None:
            from .namesim import CandTable
            self.ct = CandTable([c['name'] for c in kb])
        if hasattr(self, 'nt'):
            from .nametext import ListIndex
            self.lidx = ListIndex(kb)
        from .rescorer import NameRescorer
        self.resc = NameRescorer(kb, self._resc_params())
        self.exn = TL.ExactNames(kb)

    @torch.no_grad()
    def feed(self, h, x):
        T = h.shape[0]
        if T == 0:
            return
        m, R = (self.m, self.R)
        if self.wfeed is not None:
            self.ring_a.append(h.float() if self.dev.type == 'cuda' else h.float().cpu())
        ha = h
        h = self._fuse(h)
        hb = h[None]
        zf = m.head.logits_q('frame', hb, self.domt, self.K)[0].float().cpu().numpy()
        p_frame = R.probs('frame', zf)
        slot = m.diar.slot_logits(ha).float().cpu().numpy()
        emb = m.diar.embed(ha, x).cpu().numpy() if self.bank is not None else None
        sp_col = R.cols['frame'].index(self.tab.idx['meta.speech'])
        speech = p_frame[:, sp_col]
        spk = np.zeros(T, np.int64)
        ust = np.zeros(T, bool)
        ucl = np.zeros(T, bool)
        for t in range(T):
            spk[t], ust[t], ucl[t] = self._spk_utt(t, slot[t], speech[t])
        zin, self.carry = m.head.pooled(ha[None], torch.as_tensor(spk, device=self.dev)[None], torch.as_tensor(ust, device=self.dev)[None], self.carry, h_utt=hb if self.wfeed is not None else None)
        P = {'frame': p_frame}
        for q in QUERIES:
            if q == 'frame' or not len(R.cols[q]):
                continue
            src = zin[q] if q in POOLED else hb
            P[q] = R.probs(q, m.head.logits_q(q, src, self.domt, self.K)[0].float().cpu().numpy())
        bio = _sig(m.bio.ev(ha).float().cpu().numpy())[:, :3] if m.bio is not None else np.zeros((T, 0), np.float32)
        self.last = {q: P[q][-1] for q in P}
        self.last_spk = int(spk[-1])
        self.ring_h.append(h.float() if self.dev.type == 'cuda' else h.float().cpu())
        for t in range(T):
            f = self.f + t
            self._decode_frame(f, {q: P[q][t] for q in P}, spk[t], speech[t] >= 0.5, bio[t], ust[t], ucl[t])
            if self.bank is not None:
                self._bank_step(f, emb[t], slot[t], speech[t] >= 0.5)
        for q in QUERIES:
            if q in P:
                self.frames[q].append(P[q][:, R.out_cols[q]])
        self.frames['bio'].append(bio)
        self.frames['speaker_index'].append(spk)
        self.f += T
        self._resolve_pending(final=False)
        if self.early is not None and self.early.get('audio') and (self.nsrc == 'decoder') and (self.E is not None):
            with self.words.lock:
                self._audio_first(self.early['audio'])
        self._trim_ring()

    def _resc_params(self):
        rp = dict(self.cfg['rescorer'].get(FINE_DOMAINS[self.dom_i], self.cfg['rescorer']['unknown']))
        pol = getattr(self, 'npol', {}) or {}
        if pol.get('delta') is not None:
            rp['bias'] = float(rp.get('bias', 0.0)) + float(pol['delta'])
        if pol.get('thr') is not None:
            rp['thr'] = float(pol['thr'])
        return rp

    def _policy(self, rec, span_text, allowed):
        pol = self.npol
        toks = [t for t in span_text.split() if not is_filler(t)]
        if pol.get('require_heard') and rec.get('state') == 'linked' and (not toks):
            rec.update(entity=None, state='abstain')
            rec.pop('kb_index', None)
        tau = pol.get('fuzzy_tau')
        if tau and rec.get('state') == 'abstain' and toks and len(allowed):
            from .namesim import CandTable, sim_feats, norm as _nn
            tk = _nn(' '.join(toks)).split()
            subs = sorted({' '.join(tk[i:i + n]) for n in (1, 2, 3) for i in range(len(tk) - n + 1)})
            if subs:
                F = sim_feats(subs, CandTable([self.kb[int(r_)]['name'] for r_ in allowed]))[..., 4].astype(np.float32)
                j = int(F.max(0).argmax())
                sc = float(F.max())
                if sc >= float(tau):
                    row = int(allowed[j])
                    rec.update(entity=self.kb[row]['name'], state='linked', kb_index=row, via='text', p=round(sc, 4))

    def _ptr(self, h, i, ok, heard, W=None):
        Es = Sf = None
        src = 0
        if heard:
            if self.m.ent.span_q is not None:
                Es = self.m.ent.span_embed(heard)[None, None]
            if getattr(self.m.ent, 'sim_w', None) is not None and self.ct is not None:
                from .namesim import sim_feats
                Sf = torch.as_tensor(sim_feats([heard], self.ct), device=self.dev)[None]
            W = self.words if W is None else W
            fs = W.finals[-1].get('src', 'external') if W.finals else 'external'
            src = int(self.ptr_src.get(fs, self.ptr_src.get('external', 0)))
        cm = torch.as_tensor(ok[None, None], device=self.dev)
        return self.m.ent.ptr_logits(h, self.K[i][None, None], self.E, cm, Es=Es, Sf=Sf, src=src)[0, 0]

    def audio_h_at(self, frames):
        return self._ha_at(frames)

    def _ha_at(self, frames):
        if not self.ring_a:
            return self._h_at(frames)
        H = torch.cat(self.ring_a, 0)
        idx = [min(max(fr - self.ring_f0, 0), len(H) - 1) for fr in frames]
        return H[idx].to(self.dev)

    def _fuse(self, h):
        wf = self.wfeed
        if wf is None:
            return h
        with self.words.lock:
            fin = list(self.words.finals)
        src_of = lambda w: int(self.wc_src.get(w.get('src', 'external'), self.wc_src.get('external', 0)))
        if wf.sync(fin, self.f, src_of=src_of) and self.word_arrival in ('rule', 'given'):
            from .wordch import rule_arrival
            av = rule_arrival(np.asarray(wf.f0, np.int64), np.asarray(wf.oe, np.int64), self.d)
            if self.word_arrival == 'given':
                ga = {k: w.get('arrive_f') for k, w in zip(((w['start'], w['end'], w['w']) for w in fin), fin)}
                go = {(w['start'], w['end'], w['w']): w.get('oe_f') for w in fin}
                wf.oe = [int(go[k]) if go.get(k) is not None else o for k, o in zip(wf.keys, wf.oe)]
                av = np.asarray([max(int(ga[k]), o) if ga.get(k) is not None else int(a) for k, a, o in zip(wf.keys, av, wf.oe)], np.int64)
                av = np.maximum.accumulate(av) if len(av) else av
            wf.av = [int(a) for a in av]
        wf.encode_upto(self.f + h.shape[0], self._ha_at, f_start=self.f if self.word_arrival == 'live' else None)
        return wf.fuse(h, self.f)

    def _trim_ring(self):
        if len(self.ring_h) > 64:
            self.ring_h = [torch.cat(self.ring_h, 0)]
            if self.ring_a:
                self.ring_a = [torch.cat(self.ring_a, 0)]
        tot = sum((r.shape[0] for r in self.ring_h))
        while len(self.ring_h) > 1 and tot - self.ring_h[0].shape[0] >= RING:
            tot -= self.ring_h[0].shape[0]
            self.ring_f0 += self.ring_h[0].shape[0]
            self.ring_h.pop(0)
            if self.ring_a:
                self.ring_a.pop(0)
        if len(self.ring_h) == 1 and tot > RING + 600:
            cut = tot - RING
            self.ring_h[0] = self.ring_h[0][cut:]
            self.ring_f0 += cut
            if self.ring_a:
                self.ring_a[0] = self.ring_a[0][cut:]

    def _h_at(self, frames):
        H = torch.cat(self.ring_h, 0)
        idx = [min(max(fr - self.ring_f0, 0), len(H) - 1) for fr in frames]
        return H[idx].to(self.dev)

    def _spk_utt(self, t, slot_z, p_speech):
        s = int(np.argmax(slot_z)) - 1
        if s != self.spk_cur:
            if s == self.spk_pend:
                self.spk_n += 1
            else:
                self.spk_pend, self.spk_n = (s, 1)
            if self.spk_n >= SPK_SWITCH or (self.spk_cur == -1 and s >= 0 and (self.spk_n >= 2)):
                self.spk_cur, self.spk_pend, self.spk_n = (s, None, 0)
        else:
            self.spk_pend, self.spk_n = (None, 0)
        speech = p_speech >= 0.5
        spk = self.spk_cur if speech else -1
        start = close = False
        if speech:
            if not self.u_in or (spk >= 0 and self.u_spk >= 0 and (spk != self.u_spk)):
                close = self.u_in
                start = True
                self.u_in = True
                self.u_spk = spk
            elif spk >= 0 and self.u_spk < 0:
                self.u_spk = spk
            self.u_sil = 0
        elif self.u_in:
            self.u_sil += 1
            if self.u_sil >= UTT_PAUSE:
                close = True
                self.u_in = False
                self.u_spk = -1
        return (spk, start, close)

    def _decode_frame(self, f, p, spk, is_speech, bio, start, close):
        R, tab = (self.R, self.tab)
        t0 = f / 10.0
        if close:
            self._close_utt()
        if start:
            self.in_utt, self.utt0, self.utt_spk = (True, f, spk)
        if is_speech and self.in_utt:
            if spk >= 0 and self.utt_spk < 0:
                self.utt_spk = spk
            self.utt_last = f
            if 'utt' in p:
                self.utt_p = p['utt']
        if spk >= 0 and 'speaker' in p:
            self.speakers[int(spk)] = dict(p=p['speaker'], last=f)
        if 'item' in p:
            self.item = p['item']
        if self.diar_mode == 'slots':
            lab = int(spk)
            if self.diar_seg is None or self.diar_seg[0] != lab:
                if self.diar_seg is not None and self.diar_seg[0] >= 0:
                    self.out['diarization'].append(dict(speaker=f'S{self.diar_seg[0] + 1}', start=self.diar_seg[1] / 10, end=f / 10))
                self.diar_seg = [lab, f]
        if 'span_in' in p:
            for j in R.out_cols['span_in']:
                i = R.cols['span_in'][j]
                tid = tab.ids[i]
                pj = float(p['span_in'][j])
                thr = R.threshold(tid, 'span_in')
                o = self.open_spans.get(j)
                if pj >= thr:
                    if o is None:
                        self.open_spans[j] = [f, f, pj, 1, 0, p['span_in'].astype(np.float64)]
                    else:
                        o[1] = f
                        o[2] += pj
                        o[3] += 1
                        o[4] = 0
                        o[5] = o[5] + p['span_in']
                elif o is not None:
                    o[4] += 1
                    if o[4] > SPAN_GAP:
                        self._close_span(j, o)
                        del self.open_spans[j]
        for q in ('ev_at', 'ev_rec', 'ev_imm'):
            if q not in p:
                continue
            for j in R.out_cols[q]:
                i = R.cols[q][j]
                tid = tab.ids[i]
                pj = float(p[q][j])
                thr = R.threshold(tid, q)
                last = self.ev_last.get((q, j), -10 ** 9)
                if pj >= thr and f - last > (100 if q == 'ev_rec' else 10):
                    self.ev_last[q, j] = f
                    ev = dict(tag=tid, t=round(t0, 2), p=round(pj, 4), window={'ev_at': 'at', 'ev_rec': 'recent', 'ev_imm': 'imminent'}[q])
                    if R.status.get(i, 'shipped') != 'shipped':
                        ev['status'] = R.status[i]
                    self.out['events'].append(ev)
        for g, loc in R.grp_loc.get('frame', []):
            ok = [j for j in loc if j in set(R.out_cols['frame'])]
            if not ok:
                continue
            pv = p['frame'][ok]
            if pv.sum() <= 0:
                continue
            k = ok[int(np.argmax(pv))]
            lab = tab.ids[R.cols['frame'][k]]
            pk = float(p['frame'][k])
            s = self.seg.get(g)
            if s is None:
                self.seg[g] = [lab, f, pk, 1, None, 0]
            elif lab == s[0]:
                s[2] += pk
                s[3] += 1
                s[4], s[5] = (None, 0)
            else:
                if s[4] == lab:
                    s[5] += 1
                else:
                    s[4], s[5] = (lab, 1)
                if s[5] >= SEG_SWITCH:
                    self.out['segments'].append(dict(group=g, label=s[0], start=s[1] / 10, end=(f - s[5] + 1) / 10, p=round(s[2] / s[3], 4)))
                    self.seg[g] = [lab, f - s[5] + 1, pk * s[5], s[5], None, 0]
                else:
                    s[2] += float(p['frame'][[x for x in ok if tab.ids[R.cols['frame'][x]] == s[0]][0]])
                    s[3] += 1
        for k, nm in enumerate(self.bio_names[:len(bio)] if self.bio_on else ()):
            last = self.ev_last.get(('bio', k), -10 ** 9)
            if bio[k] >= 0.5 and f - last > 10:
                self.ev_last['bio', k] = f
                self.out['bio_events'].append(dict(tag=f'bio.{nm}', t=round(t0, 2), p=round(float(bio[k]), 4), status='uncalibrated'))

    def _close_utt(self):
        if self.utt0 is None or self.utt_p is None:
            self.in_utt = False
            return
        R, tab = (self.R, self.tab)
        p = self.utt_p
        tags, groups = ({}, {})
        ind = set(R.out_cols['utt'])
        grouped = set()
        for g, loc in R.grp_loc.get('utt', []):
            ok = [j for j in loc if j in ind]
            grouped.update(loc)
            if ok:
                k = ok[int(np.argmax(p[ok]))]
                groups[g] = dict(label=tab.ids[R.cols['utt'][k]], p=round(float(p[k]), 4))
        for j in R.out_cols['utt']:
            if j not in grouped:
                tags[tab.ids[R.cols['utt'][j]]] = round(float(p[j]), 4)
        self.out['utterances'].append(dict(start=self.utt0 / 10, end=(self.utt_last + 1) / 10, speaker=f'S{self.utt_spk + 1}' if self.utt_spk >= 0 else None, tags=tags, groups=groups))
        self.in_utt, self.utt0, self.utt_p, self.utt_spk = (False, None, None, -1)

    def _close_span(self, j, o):
        R, tab = (self.R, self.tab)
        i = R.cols['span_in'][j]
        tid = tab.ids[i]
        a, b = (o[0], o[1])
        sp = dict(tag=tid, start=a / 10, end=(b + 1) / 10, p=round(o[2] / o[3], 4))
        if R.status.get(i, 'shipped') != 'shipped':
            sp['status'] = R.status[i]
        self.out['spans'].append(sp)
        if self.nsrc == 'decoder':
            return
        if tid.startswith('entity.mention.') and self.E is not None:
            self.pending.append(dict(kind='name', tag=tid, i=i, a=a, b=b, span=sp, due=(b + 1 + self.d) / 10 + 0.3))
        elif tid == 'entity.numeral':
            self.pending.append(dict(kind='number', a=a, b=b, span=sp, due=(b + 1 + self.d) / 10 + 0.3))

    def _new_id(self, kind):
        self.n_id[kind] += 1
        return f"{('nm' if kind == 'name' else 'nu')}{self.n_id[kind]}"

    @staticmethod
    def _early_val(kind, r):
        if r is None:
            return None
        if kind == 'name':
            return (str(r.get('entity')).lower() if r.get('state') == 'linked' else None, r.get('state'))
        return (r.get('value'), r.get('value2'))

    @staticmethod
    def _early_gate(kind, g, r):
        if g is None or r is None:
            return False
        if kind == 'name':
            if r.get('state') != 'linked' or r.get('p') is None or float(r['p']) < float(g.get('p', 1.01)):
                return False
            pt = r.get('ptr') or {}
            if g.get('margin') is not None and (pt.get('margin') is None or float(pt['margin']) < float(g['margin'])):
                return False
            if g.get('sim') is not None and (pt.get('sim') is None or float(pt['sim']) < float(g['sim'])):
                return False
            return len(r.get('words') or []) >= int(g.get('n_words', 1))
        if r.get('value') is None:
            return False
        if g.get('agree') and (not (r.get('value_from_text') is not None and r.get('value_model') is not None and (str(r['value_from_text']) == str(r['value_model'])))):
            return False
        if g.get('kind') and (not r.get('kind')):
            return False
        if g.get('state') is not None and r.get('state') != g['state']:
            return False
        return len(r.get('tokens') or []) >= int(g.get('n_tokens', 1))

    def _phase(self, kind, rec, prev):
        if self.early is None or rec is None:
            return rec
        g = self.early.get(kind)
        ok = self._early_gate(kind, g, rec)
        same = prev is not None and self._early_val(kind, prev) == self._early_val(kind, rec)
        if ok and g and (int(g.get('stable', 1)) >= 2) and (not (same and prev.get('_gate'))):
            ok = False
        if same and prev.get('phase') == 'early':
            ok = True
        rec['_gate'] = self._early_gate(kind, g, rec)
        rec['phase'] = 'early' if ok else 'tentative'
        return rec

    @torch.no_grad()
    def _audio_first(self, g):
        cols = self.R.cols['span_in']
        b = self.f - 1
        st = self.__dict__.setdefault('_af', {})
        live = set()
        for j, o in list(self.open_spans.items()):
            tid = self.tab.ids[cols[j]]
            if not tid.startswith('entity.mention.'):
                continue
            key = (j, o[0])
            live.add(key)
            a = st.setdefault(key, dict(run=0, last=None, done=False))
            if a['done']:
                continue
            ok = np.isin(self.ckind, MENTION_KINDS.get(tid, list(range(len(C_KINDS)))) + ([9] if tid == 'entity.mention.team' else []))
            if not ok.any():
                a['done'] = True
                continue
            t0, t1 = (o[0] / 10.0, (b + 1) / 10.0)
            if any((min(e['end'], t1) - max(e['start'], t0) > -0.05 for e in self.dec_prov)) or t0 < self.words.final_upto:
                a['done'] = True
                continue
            i = cols[j]
            heard = self.words.view().heard_tail(b + self.d) if g.get('words') else ''
            z = self._ptr(self._h_at([b])[None], i, ok, heard)
            z[1] = NEG
            z = z.float().cpu().numpy()
            pz = np.exp(z - z.max())
            pz /= pz.sum()
            top = int(np.argmax(z))
            if top < 2 or pz[top] < float(g.get('p', 1.01)):
                a['run'], a['last'] = (0, None)
                continue
            a['run'] = a['run'] + 1 if a['last'] == top else 1
            a['last'] = top
            if a['run'] < int(g.get('hold', 3)):
                continue
            a['done'] = True
            pid = self._new_id('name') + 'p'
            rec = dict(start=round(t0, 3), end=round(t1, 3), tag=tid, heard='', model=dict(entity=self.kb[top - 2]['name'], p=round(float(pz[top]), 4)), src='audio', decoder_tag=None, entity=self.kb[top - 2]['name'], p=round(float(pz[top]), 4), state='linked', kb_index=top - 2, words=[], span_p=round(float(o[2] / max(1, o[3])), 4), head_p=None, show=True, ptr=dict(margin=round(float(z[top] - z[0]), 3), p_new=round(float(pz[0]), 4), sim=None, n_words=0), id=pid, provisional=True, phase='early', _gate=True)
            self._upd(dict(op='provisional', type='name', id=pid, revises=None, record=self._pub(rec)))
            self.dec_prov.append(dict(kind='name', start=rec['start'], end=rec['end'], pid=pid, rec=rec))
        for k in [k for k in st if k not in live]:
            del st[k]

    @staticmethod
    def _pub(r):
        return r if r is None or '_gate' not in r else {k: v for k, v in r.items() if k != '_gate'}

    def _upd(self, u, prev=None):
        if self.early is not None and prev is not None and (prev.get('phase') == 'early'):
            r = u.get('record')
            if r is None or self._early_val(u['type'], r) != self._early_val(u['type'], prev):
                u['correction'] = True
        self.out['updates'].append(u)

    @staticmethod
    def _same(kind, r, q):
        if r is None or q is None:
            return r is None and q is None
        keys = ('entity', 'state') if kind == 'name' else ('value', 'value2', 'kind', 'unit', 'value_from_text')
        if r.get('src') == 'decoder':
            keys = keys + ('heard',)
        return all((r.get(k) == q.get(k) for k in keys))

    def _commit(self, kind, rec, prov):
        if prov is None:
            if rec is not None:
                rec['id'] = self._new_id(kind)
            return rec
        pid, prec = prov
        if rec is not None and self._same(kind, rec, prec):
            rec['id'] = pid
            self.out['updates'].append(dict(op='confirm', type=kind, id=pid, revises=None, record=dict(rec, provisional=False)))
        else:
            nid = self._new_id(kind) + 'r'
            if rec is not None:
                rec['id'] = nid
                rec['revises'] = pid
            self._upd(dict(op='revise', type=kind, id=nid, revises=pid, record=None if rec is None else dict(rec, provisional=False)), prec)
        return rec

    def _provisional(self, kind, rec, members):
        old = next((q['prov'] for q in members if q.get('prov')), None)
        pid = self._new_id(kind) + ('p' if old is None else 'pr')
        if rec is not None:
            rec = self._phase(kind, dict(rec, id=pid, provisional=True), None if old is None else old[1])
        self._upd(dict(op='provisional' if old is None else 'revise', type=kind, id=pid, revises=None if old is None else old[0], record=self._pub(rec)), None if old is None else old[1])
        for q in members:
            q['prov'] = (pid, rec)

    def _resolve_pending(self, final):
        with self.words.lock:
            self._resolve_pending_locked(final)

    def _resolve_pending_locked(self, final):
        if self.war:
            self._war_step(final)
        if self.nsrc == 'decoder':
            self._dec_step(final)
        self._resolve_head(final)

    def _resolve_head(self, final):
        now = (self.f + self.d) / 10.0
        cs = self.cov_submitted() if self.cov_submitted is not None else None
        ready = lambda pd: final or self.words.covered >= pd['due'] or (now >= pd['due'] + self.word_wait and (not (cs is not None and cs >= pd['due'])))
        V = None
        if self.prov and (not final) and self.pending:
            V = self.words.view()
        pready = lambda pd: V is not None and (self.prov == 'audio' or V.covered >= pd['b'] / 10 + 0.2)
        open_m = [o[0] for j, o in self.open_spans.items() if self.tab.ids[self.R.cols['span_in'][j]].startswith('entity.mention.')]
        names = [pd for pd in self.pending if pd['kind'] == 'name']
        groups, used = ([], set())
        for k, pd in enumerate(names):
            if k in used:
                continue
            g = [pd]
            used.add(k)
            lo, hi = (pd['a'], pd['b'])
            grew = True
            while grew:
                grew = False
                for k2, q in enumerate(names):
                    if k2 not in used and q['a'] <= hi + 1 and (q['b'] >= lo - 1):
                        g.append(q)
                        used.add(k2)
                        lo, hi = (min(lo, q['a']), max(hi, q['b']))
                        grew = True
            groups.append((lo, hi, g))
        done = set()

        def read_group(g, W):
            recs = [r for r in (self._read_name(q, W) for q in g) if r is not None]
            if not recs:
                return None
            best = max(recs, key=lambda r: (r['state'] == 'linked', r['p'], r['end'] - r['start']))
            if len(recs) > 1:
                best['classes'] = sorted({r['tag'] for r in recs})
            return best
        for lo, hi, g in groups:
            if not final and any((o <= hi + 1 for o in open_m)):
                continue
            if not all((ready(q) for q in g)):
                if V is not None and all((pready(q) for q in g)) and any((not q.get('prov') or q.get('prov_n') != len(g) for q in g)) and (g[0].get('prov_try') != (V.version, len(g))):
                    rec = read_group(g, V)
                    g[0]['prov_try'] = (V.version, len(g))
                    if rec is not None:
                        self._provisional('name', rec, g)
                        for q in g:
                            q['prov_n'] = len(g)
                continue
            best = read_group(g, None)
            prov = next((q['prov'] for q in g if q.get('prov')), None)
            if prov is not None:
                prov = max((q['prov'] for q in g if q.get('prov')), key=lambda x: self._pid_order(x[0]))
            best = self._commit('name', best, prov)
            if best is not None:
                self.out['names'].append(best)
                for q in g:
                    q['span']['name'] = best['entity']
            done.update((id(q) for q in g))
        keep = []
        for pd in self.pending:
            if pd['kind'] == 'name':
                if id(pd) not in done:
                    keep.append(pd)
            elif ready(pd):
                self.out['numbers'].append(self._commit('number', self._read_number(pd), pd.get('prov')))
            else:
                if V is not None and (not pd.get('prov')) and pready(pd) and (pd.get('prov_try') != V.version):
                    pd['prov_try'] = V.version
                    rec = self._read_number(pd, V)
                    if rec['state'] != 'no_words' or self.prov == 'audio':
                        self._provisional('number', rec, [pd])
                keep.append(pd)
        self.pending = keep

    def _pid_order(self, pid):
        import re
        return int(re.sub('\\D', '', pid) or 0)

    def _read_name(self, pd, W=None):
        W = self.words if W is None else W
        a, b, i = (pd['a'], pd['b'], pd['i'])
        tid = pd['tag']
        ok = np.isin(self.ckind, MENTION_KINDS.get(tid, list(range(len(C_KINDS)))) + ([9] if tid == 'entity.mention.team' else []))
        allowed = np.where(ok)[0]
        if not len(allowed):
            return None
        ws = W.overlapping(a, b)
        if self.nt['mode'] != 'overlap' and ws:
            ws = self._text_span(pd, ws, W)
            if not ws:
                return None
        content = [w for w in ws if is_content(w['w'])]
        if W.finals and (not content) and (not (self.nt['mode'] == 'gate' and ws)):
            return None
        h = self._h_at([b])[None]
        heard = W.heard_tail(b + self.d)
        z = self._ptr(h, i, ok, heard, W)
        z[1] = NEG
        z = z.float().cpu().numpy()
        pz = np.exp(z - z.max())
        pz /= pz.sum()
        top = int(np.argmax(z))
        span_text = ' '.join((w['w'] for w in ws if not is_filler(w['w'])))
        st_, en_ = (pd['span']['start'], pd['span']['end']) if self.nt['mode'] == 'overlap' or not ws else (ws[0]['start'], ws[-1]['end'])
        rec = dict(start=st_, end=en_, tag=tid, heard=span_text, model=dict(entity=None if top == 0 else self.kb[top - 2]['name'], p=round(float(pz[top]), 4)))
        if self.rescore and self.resc is not None:
            past = W.text_between(max(0.0, a / 10 - 30), a / 10)
            r = self.resc.score((b + 1) / 10, z, allowed, span_text, past)
            rec.update(entity=self.kb[r['pred']]['name'] if r['link'] else None, p=r['p'], state='linked' if r['link'] else 'not_in_list' if r['pred'] is None else 'abstain')
            if r['link']:
                rec['kb_index'] = int(r['pred'])
            self._policy(rec, span_text, allowed)
        else:
            r = dict(pred=None if top < 2 else top - 2, sim=None)
            rec.update(entity=rec['model']['entity'], p=rec['model']['p'], state='linked' if top >= 2 else 'not_in_list')
        cz = z[2:]
        cj = r['pred'] if r.get('pred') is not None else int(np.argmax(cz)) if len(cz) else None
        if self.diag:
            rec['ptr'] = dict(margin=None if cj is None else round(float(z[2 + cj] - z[0]), 3), p_new=round(float(pz[0]), 4), sim=r.get('sim'), n_words=len(ws))
        rec['words'] = [dict(w=w['w'], start=w['start'], end=w['end']) for w in ws]
        rec['span_p'] = pd['span'].get('p')
        rec['show'] = self._show_name(tid, rec['span_p'], ws)
        return rec

    def _text_span(self, pd, ws, W):
        from .nametext import snap, gate
        import bisect
        i, tid = (pd['i'], pd['tag'])
        lo = min((w['f0'] for w in ws))
        hi = max((w['f1'] for w in ws))
        j = self.R.cols['span_in'].index(i)
        rows = self._span_rows('span_in', lo, hi)[:, j]
        k = snap(ws, rows, lo, self.R.threshold(tid, 'span_in'), float(self.nt['snap_frac']))
        if self.nt['mode'] == 'snap':
            return [ws[x] for x in k]
        F = W.finals
        n = bisect.bisect_left([w['start'] for w in F], ws[0]['start']) if len(F) < 4000 else bisect.bisect_left(F, ws[0]['start'], key=lambda w: w['start'])
        past = [w for w in F[max(0, n - 1500):n] if w['end'] <= ws[0]['start'] + 1e-06]
        kept = gate(ws, k, past, self.lidx, float(self.nt['sim']), DISPLAY_STOP | FILLERS, numeric_word)
        return [ws[x] for x in kept]

    def _show_name(self, tid, span_p, ws):
        nd = self.cfg.get('name_display') or {}
        th = (nd.get(self.R.dom) or nd.get('unknown') or {}).get(tid.split('.')[-1])
        if th is not None and span_p is not None and (float(span_p) < float(th)):
            return False
        if ws and (not any((_w(w['w']) and _w(w['w']) not in DISPLAY_STOP and (not numeric_word(w['w'])) and (not is_filler(w['w'])) for w in ws))):
            return False
        return True

    def _read_number(self, pd, W=None):
        W = self.words if W is None else W
        a, b = (pd['a'], pd['b'])
        allw = W.finals
        dec = pd.get('src') == 'decoder'
        span_ws = self._span_words(pd, W) if dec else None
        ws = [w for w in (span_ws if dec else W.overlapping(a, b)) if numeric_word(w['w'])]
        while ws and connector(ws[0]['w']):
            ws = ws[1:]
        while ws and connector(ws[-1]['w']):
            ws = ws[:-1]
        if not ws:
            r = dict(start=pd['span']['start'], end=pd['span']['end'], value=None, value2=None, kind=None, unit=None, heard='', tokens=[], value_from_text=None, value2_from_text=None, state='no_words')
            if dec:
                r.update(src='decoder', decoder_tag=pd['dtag'], kind=pd['nkind'], heard=' '.join((w['w'] for w in span_ws or [])), value_model=None, value2_model=None)
            return r
        idx = [allw.index(w) for w in ws]
        toks = []
        for j in idx:
            e = own_end(allw, j)
            s0 = allw[j]['f0']
            fr = list(range(max(s0, e - 1), e + 1))
            vl = self.m.ent.value_logits(self._h_at(fr))
            nwt = vl['nwt'].mean(0)
            toks += [self.nwt[k] for k in nwt.argmax(-1).tolist() if k != 0]
        e_last = own_end(allw, idx[-1])
        vl = self.m.ent.value_logits(self._h_at([e_last]))
        ui = int(vl['unit'][0].argmax())
        unit = self.unit_vocab[ui] if ui < len(self.unit_vocab) else None
        kind = pd['nkind'] if dec else self._number_kind(a, b)
        try:
            val = NV.finalize(kind, toks, unit, [w['w'] for w in ws]) if kind else NL.compose(toks, None)
        except Exception:
            val = None
        tw = [w for w in span_ws if w['w'].strip()] if dec else allw[idx[0]:idx[-1] + 1]
        if not tw:
            tw = ws
        k0, k1 = (allw.index(tw[0]), allw.index(tw[-1]))
        vt, unit_t = TL.text_value(kind, [w['w'] for w in tw], [allw[k1 + 1]['w']] if k1 + 1 < len(allw) else [], allw[k0 - 1]['w'] if k0 else '')
        if vt is None and kind and (unit_t is None) and (unit is not None):
            vt, _ = TL.text_value(kind, [w['w'] for w in tw], [], '', unit=unit)
            unit_t = unit if vt is not None else None
        rec = dict(start=pd['span']['start'], end=pd['span']['end'], kind=kind, unit=unit, tokens=toks, value=None if val is None else val.get('v'), value2=None if val is None else val.get('v2'), heard=' '.join((w['w'] for w in ws)), state='read', value_from_text=None if vt is None else vt.get('v'), value2_from_text=None if vt is None else vt.get('v2'))
        rec['value_model'], rec['value2_model'] = (rec['value'], rec['value2'])
        if vt is not None and vt.get('v') is not None and (any((ch.isdigit() for w in ws for ch in w['w'])) or (vt.get('v2') is not None and rec['value2'] is None) or rec['value'] is None):
            rec.update(value=vt.get('v'), value2=vt.get('v2'), state='read_text', unit=unit_t or unit)
        if dec:
            rec.update(src='decoder', decoder_tag=pd['dtag'], kind_head=self._number_kind(a, b), heard=' '.join((w['w'] for w in span_ws)))
            if vt is not None and vt.get('v') is not None:
                rec.update(value=vt.get('v'), value2=vt.get('v2'), state='read_text', unit=unit_t or unit)
        return rec

    @torch.no_grad()
    def _war_step(self, final):
        wf = self.wfeed
        if wf is None or wf.U is None or (not len(wf.U)):
            if final and self.war_cur is not None:
                self._war_emit(self.war_cur)
                self.war_cur = None
            return
        from .tagdec import NAME_TAG, NUMBER_KIND
        tab = self.tab
        if not hasattr(self, '_war_cols'):
            cin = list(tab.cols['span_in'])
            ok = lambda i: bool(tab.valid[self.dom_i, i])
            self._war_cin = torch.as_tensor(cin, device=self.dev)
            inv = {v: k for k, v in NAME_TAG.items()}
            self._war_m = [(j, inv[tab.ids[i]]) for j, i in enumerate(cin) if tab.ids[i] in inv and tab.ids[i] != 'entity.mention.legal_ref' and ok(i)]
            self._war_n = [j for j, i in enumerate(cin) if tab.ids[i] == 'entity.numeral' and ok(i)]
            ik = {v: k for k, v in NUMBER_KIND.items() if v}
            self._war_nk = [(j, ik[tab.ids[i].split('.')[-1]]) for j, i in enumerate(cin) if tab.ids[i].startswith('entity.number.') and tab.ids[i].split('.')[-1] in ik and ok(i)]
            self._war_cols = True
        n_u = len(wf.U)
        j0 = self.war_j
        if j0 >= n_u:
            if final and self.war_cur is not None:
                self._war_emit(self.war_cur)
                self.war_cur = None
            return
        lo = max(0, j0 - 2)
        U = wf.U[lo:n_u][None]
        valid = torch.ones(U.shape[:2], dtype=torch.bool, device=self.dev)
        pB, pI, _ = wf.w.war_probs(U, valid, self.K[self._war_cin], k=self.war_k)
        pB = pB[0].float().cpu().numpy()
        pI = pI[0].float().cpu().numpy()
        pin = pB + pI
        tn = float(self.war_thr.get('name', 0.5))
        tq = float(self.war_thr.get('number', 0.5))
        for j in range(j0, n_u):
            if not (final or j + self.war_k < n_u or self.f > wf.oe[j] + self.war_wait):
                break
            r = j - lo
            lab = None
            if self._war_m:
                jm, tg = max(self._war_m, key=lambda x: pin[r, x[0]])
                if pin[r, jm] >= tn:
                    lab = (tg, pB[r, jm] >= pI[r, jm], float(pin[r, jm]))
            if self._war_n and pin[r, self._war_n[0]] >= tq and (lab is None or pin[r, self._war_n[0]] > lab[2]):
                jn = self._war_n[0]
                lab = ('NUMBER', pB[r, jn] >= pI[r, jn], float(pin[r, jn]))
            cur = self.war_cur
            gap = cur is not None and wf.f0[j] - cur['oe'] > 4
            if cur is not None and (lab is None or lab[0] != cur['tag'] or lab[1] or gap):
                self._war_emit(cur)
                self.war_cur = cur = None
            if lab is not None:
                if cur is None:
                    self.war_cur = cur = dict(tag=lab[0], j=[], oe=wf.oe[j], kp=[])
                cur['j'].append(j)
                cur['oe'] = wf.oe[j]
                if lab[0] == 'NUMBER' and self._war_nk:
                    cur['kp'].append([pin[r, x[0]] for x in self._war_nk])
            self.war_j = j + 1
        if final and self.war_cur is not None:
            self._war_emit(self.war_cur)
            self.war_cur = None

    def _war_emit(self, cur):
        wf = self.wfeed
        J = cur['j']
        tag = cur['tag']
        if tag == 'NUMBER' and cur['kp']:
            m = np.mean(np.asarray(cur['kp']), 0)
            k = int(np.argmax(m))
            if m[k] >= float(self.war_thr.get('kind', 0.3)):
                tag = self._war_nk[k][1]
        keys = [(wf.keys[j][0], wf.keys[j][2]) for j in J]
        sp = dict(tag=tag, start=wf.keys[J[0]][0], end=wf.keys[J[-1]][1], keys=keys, src='word')
        self.words.spans_final.append([sp])
        self.words.tagged = True

    def _dec_pd(self, s):
        from .tagdec import NAME_TAG, NUMBER_KIND, layer
        from .words import frames_of
        L = layer(s['tag'])
        if L not in ('name', 'number') or s['tag'] == 'ENTITY_LEGAL':
            return None
        fa, fb = frames_of(s['start'], s['end'])
        pd = dict(kind=L, src='decoder', dtag=s['tag'], a=fa, b=fb, keys=set(s['keys']), span=dict(start=round(float(s['start']), 3), end=round(float(s['end']), 3), p=None))
        if L == 'name':
            tid = NAME_TAG[s['tag']]
            if tid not in self.tab.idx:
                return None
            pd.update(tag=tid, i=self.tab.idx[tid])
            pd['span']['tag'] = tid
        else:
            pd.update(tag='entity.numeral', nkind=NUMBER_KIND[s['tag']])
            pd['span']['tag'] = 'entity.numeral'
        return pd

    def _span_words(self, pd, W):
        import bisect
        F = W.finals
        s0 = pd['span']['start']
        k = bisect.bisect_left(F, s0 - 1e-06, key=lambda w: w['start'])
        out = []
        while k < len(F) and F[k]['start'] <= pd['span']['end'] + 1e-06:
            if (F[k]['start'], F[k]['w']) in pd['keys']:
                out.append(F[k])
            k += 1
        return out

    def _head_p(self, i, a, b):
        cols = self.R.cols['span_in']
        if i not in cols or b < a:
            return None
        rows = self._span_rows('span_in', a, b)
        return round(float(rows[:, cols.index(i)].mean()), 4)

    def _read_name_dec(self, pd, W=None):
        W = self.words if W is None else W
        ws = self._span_words(pd, W)
        if not ws:
            return None
        a, b, i, tid = (pd['a'], pd['b'], pd['i'], pd['tag'])
        span_text = ' '.join((w['w'] for w in ws if not is_filler(w['w'])))
        hp = self._head_p(i, a, b)
        rec = dict(start=ws[0]['start'], end=ws[-1]['end'], tag=tid, heard=span_text, model=None, src='decoder', decoder_tag=pd['dtag'])
        ok = np.isin(self.ckind, MENTION_KINDS.get(tid, list(range(len(C_KINDS)))) + ([9] if tid == 'entity.mention.team' else [])) if self.E is not None else np.zeros(0, bool)
        if self.E is not None and span_text:
            for r_ in self.exn.rows_in(span_text):
                ok[r_] = True
        allowed = np.where(ok)[0]
        if self.E is not None and len(allowed) and span_text:
            h = self._h_at([b])[None]
            heard = W.heard_tail(b + self.d)
            z = self._ptr(h, i, ok, heard, W)
            z[1] = NEG
            z = z.float().cpu().numpy()
            pz = np.exp(z - z.max())
            pz /= pz.sum()
            top = int(np.argmax(z))
            rec['model'] = dict(entity=None if top == 0 else self.kb[top - 2]['name'], p=round(float(pz[top]), 4))
            if self.rescore and self.resc is not None:
                past = W.text_between(max(0.0, a / 10 - 30), a / 10)
                r = self.resc.score((b + 1) / 10, z, allowed, span_text, past)
                rec.update(entity=self.kb[r['pred']]['name'] if r['link'] else None, p=r['p'], state='linked' if r['link'] else 'not_in_list' if r['pred'] is None else 'abstain')
                if r['link']:
                    rec['kb_index'] = int(r['pred'])
                self._policy(rec, span_text, allowed)
            else:
                r = dict(pred=None if top < 2 else top - 2, sim=None)
                rec.update(entity=rec['model']['entity'], p=rec['model']['p'], state='linked' if top >= 2 else 'not_in_list')
            cz = z[2:]
            cj = r['pred'] if r.get('pred') is not None else int(np.argmax(cz)) if len(cz) else None
            if self.diag:
                rec['ptr'] = dict(margin=None if cj is None else round(float(z[2 + cj] - z[0]), 3), p_new=round(float(pz[0]), 4), sim=r.get('sim'), n_words=len(ws))
        else:
            rec.update(entity=None, p=None, state='not_in_list')
        rec['words'] = [dict(w=w['w'], start=w['start'], end=w['end']) for w in ws]
        rec['span_p'] = rec['head_p'] = hp
        rec['show'] = any((_w(w['w']) and _w(w['w']) not in DISPLAY_STOP and (_w(w['w']) not in BACKCHANNEL) and (not numeric_word(w['w'])) and (not is_filler(w['w'])) for w in ws))
        return rec

    def _dec_match(self, pd, pool):
        best = None
        for e in pool:
            if e['kind'] != pd['kind']:
                continue
            ov = min(e['end'], pd['span']['end']) - max(e['start'], pd['span']['start'])
            if ov > -0.05 and (best is None or ov > best[0]):
                best = (ov, e)
        return None if best is None else best[1]

    def _dec_read(self, pd, W=None):
        return self._read_name_dec(pd, W) if pd['kind'] == 'name' else self._read_number(pd, W)

    def _dec_step(self, final):
        W = self.words
        if self.xt_names:
            self._tx_names()
        while W.spans_final:
            for s in W.spans_final.pop(0):
                pd = self._dec_pd(s)
                if pd is None:
                    continue
                e = self._dec_match(pd, self.dec_prov)
                if e is not None:
                    pd['prov'] = (e['pid'], e['rec'])
                    self.dec_prov.remove(e)
                self.dec_pending.append(pd)
        for e in [e for e in self.dec_prov if final or e['start'] < W.final_upto]:
            self.dec_prov.remove(e)
            self._upd(dict(op='revise', type=e['kind'], id=self._new_id(e['kind']) + 'r', revises=e['pid'], record=None), e['rec'])
        keep = []
        for pd in self.dec_pending:
            if not (final or self.f > pd['b']):
                keep.append(pd)
                continue
            rec = self._dec_read(pd)
            if rec is not None and pd['kind'] == 'name' and self._tx_overlap(rec):
                rec = None
            rec = self._commit(pd['kind'], rec, pd.get('prov'))
            if rec is not None:
                self.out['names' if pd['kind'] == 'name' else 'numbers'].append(rec)
        self.dec_pending = keep
        if self.xt_nums:
            self._tx_numbers(final)
        if self.prov and (not final) and (W.version != self._dp_ver or self._dp_wait):
            self._dec_partial()

    def _tx_overlap(self, rec):
        return any((min(rec['end'], b) > max(rec['start'], a) for a, b in self.tx_spans[-200:]))

    def _tx_names(self):
        F = self.words.finals
        if not self.exn or not F:
            return
        new = [k for k, w in enumerate(F) if (w['start'], w['w']) not in self.tx_seen] if len(self.tx_seen) < len(F) else []
        if not new:
            return
        lo = max(0, new[0] - 3)
        seg = F[lo:]
        used = [(w['start'], w['w']) in self.tx_seen and any((min(w['end'], b) > max(w['start'], a) for a, b in self.tx_spans[-50:])) for w in seg]
        newset = set(new)
        for i, j, r in self.exn.spans([w['w'] for w in seg], used):
            if not any((lo + k in newset for k in range(i, j))):
                continue
            ws = seg[i:j]
            rec = dict(start=ws[0]['start'], end=ws[-1]['end'], tag=self.exn.tag_of(r), heard=' '.join((w['w'] for w in ws)), model=None, src='text', decoder_tag=None, entity=self.kb[r]['name'], p=1.0, state='linked', kb_index=int(r), via='exact', words=[dict(w=w['w'], start=w['start'], end=w['end']) for w in ws], span_p=None, head_p=None, show=True)
            self.tx_spans.append((rec['start'], rec['end']))
            if self.resc is not None:
                self.resc.recent.append((rec['end'], int(r)))
            self.out['names'].append(self._commit('name', rec, None))
        self.tx_seen.update(((w['start'], w['w']) for w in F[new[0]:]))

    def _tx_numbers(self, final):
        F = self.words.finals
        if self.txn_i >= len(F):
            return
        wf = self.wfeed
        lab = None
        if not final and wf is not None:
            lab = {k: j for j, k in enumerate(wf.keys[:self.war_j])}
        lo = max(0, self.txn_i - 1)
        off = self.txn_i - lo
        texts = [w['w'] for w in F[lo:]]
        from .wordch import norm_word
        for i, j in TL.number_runs(texts[off:]):
            i += off
            j += off
            ws = F[lo + i:lo + j]
            if not final:
                if lo + j + 1 >= len(F) and self.f <= ws[-1]['f1'] + self.war_wait + 5:
                    break
                if lab is not None and any(((w['start'], w['end'], w['w']) not in lab for w in ws if norm_word(w['w']))):
                    break
                if any((pd['kind'] == 'number' and pd['span']['start'] <= ws[-1]['end'] + 0.5 for pd in self.dec_pending)):
                    break
                cur = self.war_cur
                if cur is not None and wf is not None and (wf.keys[cur['j'][0]][0] <= ws[-1]['end'] + 0.5):
                    break
            self.txn_i = lo + j
            a, b = (ws[0]['start'], ws[-1]['end'])
            if any((min(r['end'], b) > max(r['start'], a) for r in self.out['numbers'][-300:])):
                continue
            dom = FINE_DOMAINS[self.dom_i]
            kind = TL.kind_rule(texts, i, j, dom)
            v, unit = TL.text_value(kind, texts[i:j], texts[j:j + 1], texts[i - 1] if i else '')
            if v is None:
                continue
            tok = NV.plain(NV.word_tokens(texts[i:j])[0])
            rec = dict(start=a, end=b, kind=kind, unit=unit, tokens=tok, value=v['v'], value2=v['v2'], heard=' '.join(texts[i:j]), state='read_text', src='text', value_from_text=v['v'], value2_from_text=v['v2'], value_model=None, value2_model=None)
            self.out['numbers'].append(self._commit('number', rec, None))
        if final:
            self.txn_i = len(F)

    def _dec_partial(self):
        W = self.words
        sp = W.partial_spans()
        lo = min((w['start'] for w in W.interim), default=None)
        V = W.view() if sp else None
        used, waiting = (set(), [])
        for s in sp:
            pd = self._dec_pd(s)
            if pd is None or pd['span']['start'] < W.final_upto:
                continue
            if pd['b'] >= self.f:
                waiting.append(pd)
                continue
            pool = [e for e in self.dec_prov if id(e) not in used]
            e = self._dec_match(pd, pool)
            rec = self._dec_read(pd, V)
            if rec is not None and pd['kind'] == 'number' and (rec['state'] == 'no_words'):
                rec = None
            kind = pd['kind']
            if e is not None:
                used.add(id(e))
                if rec is not None and self._same(kind, rec, e['rec']) and (rec.get('heard') == e['rec'].get('heard')):
                    e.update(start=pd['span']['start'], end=pd['span']['end'])
                    continue
                pid = self._new_id(kind) + 'pr'
                nr = None if rec is None else self._phase(kind, dict(rec, id=pid, provisional=True), e['rec'])
                self._upd(dict(op='revise', type=kind, id=pid, revises=e['pid'], record=self._pub(nr)), e['rec'])
                if rec is None:
                    self.dec_prov.remove(e)
                else:
                    e.update(pid=pid, rec=nr, start=pd['span']['start'], end=pd['span']['end'])
            elif rec is not None:
                pid = self._new_id(kind) + 'p'
                r = self._phase(kind, dict(rec, id=pid, provisional=True), None)
                self._upd(dict(op='provisional', type=kind, id=pid, revises=None, record=self._pub(r)))
                ne = dict(kind=kind, start=pd['span']['start'], end=pd['span']['end'], pid=pid, rec=r)
                self.dec_prov.append(ne)
                used.add(id(ne))
        if lo is not None:
            for e in list(self.dec_prov):
                if id(e) in used or e['start'] < lo - 0.05:
                    continue
                if any((min(e['end'], q['span']['end']) - max(e['start'], q['span']['start']) > -0.05 for q in waiting)):
                    continue
                self.dec_prov.remove(e)
                self._upd(dict(op='revise', type=e['kind'], id=self._new_id(e['kind']) + 'pr', revises=e['pid'], record=None), e['rec'])
        self._dp_ver, self._dp_wait = (W.version, bool(waiting))

    def _number_kind(self, a, b):
        R, tab = (self.R, self.tab)
        loc = [R.cols['span_in'].index(i) for i in tab.members.get('entity.number', []) if i in R.cols['span_in']]
        if not loc or not self.frames['span_in']:
            return None
        rows = self._span_rows('span_in', a, b)
        if rows is None:
            return None
        pv = rows[:, loc].mean(0)
        return tab.ids[R.cols['span_in'][loc[int(np.argmax(pv))]]].split('.')[-1]

    def _span_rows(self, q, a, b):
        if not hasattr(self, '_full_rows'):
            self._full_rows = {}
        H = self._h_at(list(range(a, b + 1)))[None]
        z = self.m.head.logits_q(q, H, self.domt, self.K)[0].float().cpu().numpy()
        return self.R.probs(q, z)

    def _topk(self, q, pv, loc, k):
        ok = [j for j in loc if j in self._outset(q)]
        if not ok:
            return []
        o = sorted(ok, key=lambda j: -pv[j])[:k]
        return [dict(label=self.tab.ids[self.R.cols[q][j]], p=round(float(pv[j]), 3)) for j in o]

    def _outset(self, q):
        if not hasattr(self, '_os'):
            self._os = {qq: set(v) for qq, v in self.R.out_cols.items()}
        return self._os[q]

    def _groups_of(self, q, pv, k=2, kmap=None):
        return {g: self._topk(q, pv, loc, (kmap or {}).get(g, k)) for g, loc in self.R.grp_loc.get(q, []) if self._topk(q, pv, loc, 1)}

    def _indep(self, q, pv):
        grouped = set((j for _, loc in self.R.grp_loc.get(q, []) for j in loc))
        return {self.tab.ids[self.R.cols[q][j]]: round(float(pv[j]), 3) for j in self.R.out_cols[q] if j not in grouped}

    @torch.no_grad()
    def live(self, names=True):
        if self.last is None:
            return None
        R, tab, p = (self.R, self.tab, self.last)
        out = dict(t=round(self.f / 10, 1), frame=self.f - 1)
        V = None
        sp_out, cols = ([], R.cols['span_in'])
        ment = [j for j in R.out_cols['span_in'] if tab.ids[cols[j]].startswith('entity.mention.')]
        num_loc = [cols.index(i) for i in tab.members.get('entity.number', []) if i in cols]
        for j, o in sorted(self.open_spans.items(), key=lambda kv: kv[1][0]):
            i = cols[j]
            tid = tab.ids[i]
            r = dict(tag=tid, start=round(o[0] / 10, 1), p_now=round(float(p['span_in'][j]), 3), p_mean=round(o[2] / o[3], 3), kind_guess=None, name_guess=None, value_from_text=None)
            if R.status.get(i, 'shipped') != 'shipped':
                r['status'] = R.status[i]
            if tid == 'entity.numeral' and num_loc:
                pv = o[5][num_loc] / o[3]
                k = int(np.argmax(pv))
                r['kind_guess'] = dict(label=tab.ids[cols[num_loc[k]]].split('.')[-1], p=round(float(pv[k]), 3))
                V = V or self.words.view()
                ws = [w for w in V.overlapping(o[0], o[1]) if numeric_word(w['w'])]
                while ws and connector(ws[0]['w']):
                    ws = ws[1:]
                while ws and connector(ws[-1]['w']):
                    ws = ws[:-1]
                if ws:
                    try:
                        tt, _ = NL.tokens_of([w['w'] for w in ws])
                        kd = r['kind_guess']['label']
                        vt = NV.finalize(kd, tt, None, [w['w'] for w in ws])
                        r['value_from_text'] = dict(value=vt.get('v'), value2=vt.get('v2'), heard=' '.join((w['w'] for w in ws))) if vt else None
                    except Exception:
                        pass
            elif tid.startswith('entity.mention.') and ment:
                k = max(ment, key=lambda jj: p['span_in'][jj])
                r['kind_guess'] = dict(label=tab.ids[cols[k]].split('.')[-1], p=round(float(p['span_in'][k]), 3))
                if names and self.E is not None:
                    V = V or self.words.view()
                    key = (j, o[0], o[1], V.version)
                    c = self._live_cache.get(j)
                    if c is None or c[0] != key:
                        pd = dict(a=o[0], b=o[1], i=i, tag=tid, span=dict(start=o[0] / 10, end=(o[1] + 1) / 10))
                        rec = self._read_name(pd, V)
                        g = None if rec is None else dict(entity=rec['entity'], state=rec['state'], p=round(float(rec['p']), 3), heard=rec['heard'], start=rec['start'], end=rec['end'], show=rec['show'])
                        self._live_cache[j] = c = (key, g)
                    r['name_guess'] = c[1]
            sp_out.append(r)
        for j in [j for j in self._live_cache if j not in self.open_spans]:
            del self._live_cache[j]
        out['open_spans'] = sp_out
        u = None
        if self.in_utt and self.utt_p is not None:
            pu = self.utt_p
            ind = self._indep('utt', pu)
            gr = self._groups_of('utt', pu, 2, {'discourse.intent': 3})
            u = dict(start=round(self.utt0 / 10, 1), speaker=f'S{self.utt_spk + 1}' if self.utt_spk >= 0 else None, intent=gr.pop('discourse.intent', []), act=gr.pop('discourse.act', []), device_directed=ind.get('discourse.assistant.device_directed'), command_complete=None, groups=gr, tags=ind)
            if 'ev_at' in p and 'discourse.assistant.command_complete' in tab.idx:
                ci = tab.idx['discourse.assistant.command_complete']
                if ci in R.cols['ev_at'] and R.cols['ev_at'].index(ci) in self._outset('ev_at'):
                    u['command_complete'] = round(float(p['ev_at'][R.cols['ev_at'].index(ci)]), 3)
        out['utt'] = u

        def roles(pv):
            gr = self._groups_of('speaker', pv, 2)
            return gr.get('speaker.role', next(iter(gr.values()), []))
        cur = self.last_spk
        out['speaker'] = dict(index=cur, id=f'S{cur + 1}' if cur >= 0 else None, role=roles(self.speakers[cur]['p']) if cur in self.speakers else [])
        out['speakers'] = {f'S{k + 1}': dict(role=roles(v['p']), last_seen=round(v['last'] / 10, 1)) for k, v in sorted(self.speakers.items())}
        segs = {}
        for g, sg in self.seg.items():
            c = tab.idx[sg[0]]
            pj = p['frame'][R.cols['frame'].index(c)]
            segs[g] = dict(label=sg[0], p=round(float(pj), 3), since=round(sg[1] / 10, 1), pending=sg[4])
        out['segments'] = segs
        out['item'] = self._groups_of('item', self.item, 2) if self.item is not None else {}
        return out

    def _bank_step(self, f, e, slot_z, is_speech):
        lab, revised = self.bank.step(f, e, slot_z, is_speech)
        self.bank_labels = getattr(self, 'bank_labels', {})
        self.bank_labels[f] = lab
        for fr in revised:
            self.bank_labels[fr] = lab

    def _bank_segments(self):
        L = getattr(self, 'bank_labels', {})
        segs, cur = ([], None)
        for f in sorted(L):
            lab = L[f]
            if cur is None or cur[0] != lab or f != cur[2] + 1:
                if cur is not None and cur[0] > 0:
                    segs.append(dict(speaker=f'T{cur[0]}', start=cur[1] / 10, end=(cur[2] + 1) / 10))
                cur = [lab, f, f]
            else:
                cur[2] = f
        if cur is not None and cur[0] > 0:
            segs.append(dict(speaker=f'T{cur[0]}', start=cur[1] / 10, end=(cur[2] + 1) / 10))
        return segs

    @torch.no_grad()
    def finish(self):
        for j, o in list(self.open_spans.items()):
            self._close_span(j, o)
        self.open_spans = {}
        if self.in_utt:
            self._close_utt()
        for g, s in self.seg.items():
            self.out['segments'].append(dict(group=g, label=s[0], start=s[1] / 10, end=self.f / 10, p=round(s[2] / s[3], 4)))
        self.seg = {}
        if self.diar_mode == 'slots' and self.diar_seg is not None and (self.diar_seg[0] >= 0):
            self.out['diarization'].append(dict(speaker=f'S{self.diar_seg[0] + 1}', start=self.diar_seg[1] / 10, end=self.f / 10))
        self.diar_seg = None
        if self.bank is not None:
            self.out['diarization'] = self._bank_segments()
        self._resolve_pending(final=True)

    def speaker_summary(self):
        R, tab = (self.R, self.tab)
        out = {}
        for s, v in sorted(self.speakers.items()):
            p = v['p']
            groups, tags, grouped = ({}, {}, set())
            for g, loc in R.grp_loc.get('speaker', []):
                ok = [j for j in loc if j in set(R.out_cols['speaker'])]
                grouped.update(loc)
                if ok:
                    k = ok[int(np.argmax(p[ok]))]
                    groups[g] = dict(label=tab.ids[R.cols['speaker'][k]], p=round(float(p[k]), 4))
            for j in R.out_cols['speaker']:
                if j not in grouped:
                    tags[tab.ids[R.cols['speaker'][j]]] = round(float(p[j]), 4)
            out[f'S{s + 1}'] = dict(groups=groups, tags=tags, last_seen=round(v['last'] / 10, 1))
        return out

    def item_summary(self):
        if self.item is None:
            return {}
        R, tab = (self.R, self.tab)
        out = {}
        for g, loc in R.grp_loc.get('item', []):
            ok = [j for j in loc if j in set(R.out_cols['item'])]
            if ok:
                k = ok[int(np.argmax(self.item[ok]))]
                out[g] = dict(label=tab.ids[R.cols['item'][k]], p=round(float(self.item[k]), 4))
        return out
