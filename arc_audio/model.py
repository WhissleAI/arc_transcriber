import math
import torch, torch.nn as nn, torch.nn.functional as F
from .seq import SeqLayer
from .seq.layer import rotate_pairs
from .seq.chunked import seq_chunked_state, _f
from .ontology import QUERIES, POOLED, FINE_DOMAINS
NEG = -10000.0
S_MAX = 64

class Trunk(nn.Module):

    def __init__(self, d_in=640, d_model=512, n_layers=5, d_state=64, head_dim=64, dropout=0.2, in_drop=0.2, chunk=128, n_dom=len(FINE_DOMAINS), delays=(0, 1, 2, 3, 5, 8), adapters=64, long_mem=True):
        super().__init__()
        self.cfg = dict(d_in=d_in, d_model=d_model, n_layers=n_layers, d_state=d_state, head_dim=head_dim, dropout=dropout, in_drop=in_drop, chunk=chunk, n_dom=n_dom, delays=list(delays), adapters=adapters, long_mem=long_mem)
        self.chunk = chunk
        self.in_norm = nn.LayerNorm(d_in)
        self.inp = nn.Linear(d_in, d_model)
        self.dom_emb = nn.Embedding(n_dom, d_model)
        self.film = nn.ModuleList([nn.Linear(d_model, 2 * d_model) for _ in range(n_layers)])
        self.delays = list(delays)
        self.delay_emb = nn.Embedding(len(delays), d_model)
        self.delay_film = nn.Embedding(len(delays), d_model)
        lut = torch.full((max(delays) + 1,), -1, dtype=torch.long)
        for i, d in enumerate(delays):
            lut[d] = i
        self.register_buffer('delay_lut', lut, persistent=False)
        self.adapt_down = nn.Parameter(torch.zeros(n_dom, d_model, adapters))
        self.adapt_up = nn.Parameter(torch.zeros(n_dom, adapters, d_model))
        self.layers = nn.ModuleList([SeqLayer(d_model, d_state=d_state, rank=1, head_dim=head_dim) for _ in range(n_layers)])
        self.mlps = nn.ModuleList([nn.Sequential(nn.RMSNorm(d_model), nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, d_model)) for _ in range(n_layers)])
        self.out_norm = nn.RMSNorm(d_model)

    def _cond(self, x, dom, delay):
        di = self.delay_lut[delay.clamp(0, len(self.delay_lut) - 1)]
        if (di < 0).any():
            raise ValueError(f'delay {delay.tolist()} frames not in the supported set {self.delays}')
        h = self.inp(self.in_norm(x))
        e = self.dom_emb(dom) + self.delay_film(di)
        return (h + (self.dom_emb(dom) + self.delay_emb(di))[:, None], e)

    def _adapt(self, h, dom):
        Dn, Up = (self.adapt_down[dom].to(h.dtype), self.adapt_up[dom].to(h.dtype))
        return h + torch.einsum('btr,brd->btd', F.gelu(torch.einsum('btd,bdr->btr', h, Dn)), Up)

    def forward(self, x, dom, delay, state=None):
        h, e = self._cond(x, dom, delay)
        new = []
        for i, (lay, mlp) in enumerate(zip(self.layers, self.mlps)):
            gb = self.film[i](e)[:, None]
            h = h * (1 + gb[..., :h.shape[-1]]) + gb[..., h.shape[-1]:]
            st = None if state is None else tuple((_f(t) for t in state[i][:3]))
            s, ns = seq_chunked_state(lay, _f(h), st, self.chunk)
            h = h + s
            h = h + mlp(h)
            new.append(ns)
        return (self.out_norm(self._adapt(h, dom)), new)

    def step(self, x, dom, delay, state=None):
        h_all, e = self._cond(x, dom, delay)
        B, T, D = h_all.shape
        st = [None] * len(self.layers) if state is None else list(state)
        gbs = [self.film[i](e) for i in range(len(self.layers))]
        outs = []
        for t in range(T):
            h = h_all[:, t]
            for i, (lay, mlp) in enumerate(zip(self.layers, self.mlps)):
                gb = gbs[i]
                h = h * (1 + gb[..., :D]) + gb[..., D:]
                y, st[i] = layer_step(lay, _f(h), st[i])
                h = h + y
                h = h + mlp(h)
            outs.append(h)
        h = torch.stack(outs, 1)
        return (self.out_norm(self._adapt(h, dom)), st)

def layer_step(lay, u, state):
    Bsz = u.shape[0]
    nu = lay.norm(u.unsqueeze(1))
    alpha, beta, gamma, Bm, Cm, x, theta = lay._coeffs(nu)
    H, N, P = (lay.H, lay.N, lay.P)
    bx = Bm[:, 0] @ x[:, 0].transpose(-1, -2)
    if state is None:
        h = gamma[:, 0, :, None, None] * bx
    else:
        h0, Bl, xl = (_f(t) for t in state[:3])
        prev_bx = _f(state[3]) if len(state) > 3 else Bl @ xl.transpose(-1, -2)
        pre = alpha[:, 0, :, None, None] * h0 + beta[:, 0, :, None, None] * prev_bx
        h = rotate_pairs(pre.transpose(-1, -2), theta[:, 0].unsqueeze(-2).expand(-1, -1, P, -1)).transpose(-1, -2)
        h = h + gamma[:, 0, :, None, None] * bx
    y = (Cm[:, 0].transpose(-1, -2) @ h).reshape(Bsz, -1)
    return (lay.gate_and_proj(y, nu[:, 0]), (h, Bm[:, 0], x[:, 0], bx))

def pool_speaker(z, spk, carry):
    B, T, D = z.shape
    sp = torch.where((spk >= 0) & (spk < S_MAX), spk, torch.full_like(spk, -1))
    key = (torch.arange(B, device=z.device)[:, None] * (S_MAX + 1) + (sp + 1)) * T + torch.arange(T, device=z.device)[None]
    order = torch.argsort(key.reshape(-1), stable=True)
    zs = z.reshape(B * T, D)[order].double()
    gs = key.reshape(-1)[order] // T
    cs = torch.cumsum(zs, 0)
    start = torch.ones_like(gs, dtype=torch.bool)
    start[1:] = gs[1:] != gs[:-1]
    pos = torch.arange(B * T, device=z.device)
    first = torch.cummax(torch.where(start, pos, torch.zeros_like(pos)), 0).values
    base = torch.where(first[:, None] > 0, cs[(first - 1).clamp(min=0)], torch.zeros_like(cs))
    num = torch.empty_like(cs)
    num[order] = cs - base
    cnt = torch.empty(B * T, dtype=torch.float64, device=z.device)
    cnt[order] = (pos - first + 1).double()
    num = num.reshape(B, T, D)
    cnt = cnt.reshape(B, T)
    cs_, cn_ = carry
    sidx = sp.clamp(min=0)
    num = num + torch.gather(cs_, 1, sidx[..., None].expand(B, T, D))
    cnt = cnt + torch.gather(cn_, 1, sidx)
    out = (num / cnt.clamp(min=1)[..., None]).float() * (sp >= 0)[..., None].float()
    oh = F.one_hot(sidx, S_MAX).double() * (sp >= 0)[..., None].double()
    new = (cs_ + torch.einsum('bts,btd->bsd', oh, z.double()), cn_ + oh.sum(1))
    return (out, new)

def pool_segment(z, starts, carry):
    B, T, D = z.shape
    st = torch.zeros(B, T, dtype=torch.bool, device=z.device) if starts is None else starts
    cs = torch.cumsum(z.double(), 1)
    cs0 = F.pad(cs, (0, 0, 1, 0))[:, :-1]
    t = torch.arange(T, device=z.device)[None].expand(B, T)
    first = torch.cummax(torch.where(st, t, torch.full_like(t, -1)), 1).values
    base = torch.gather(cs0, 1, first.clamp(min=0)[..., None].expand(B, T, D))
    n = (t - first.clamp(min=0) + 1).double()
    open_ = first < 0
    num = torch.where(open_[..., None], cs + carry[0][:, None], cs - base)
    n = torch.where(open_, (t + 1).double() + carry[1][:, None], n)
    out = (num / n.clamp(min=1)[..., None]).float()
    last = first[:, -1]
    new_s = torch.where((last < 0)[:, None], carry[0] + cs[:, -1], cs[:, -1] - torch.gather(cs0, 1, last.clamp(min=0)[:, None, None].expand(B, 1, D))[:, 0])
    new_n = torch.where(last < 0, carry[1] + T, (T - last).double())
    return (out, (new_s, new_n))

def zero_carry(B, D, device):
    z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
    return dict(utt=(z(B, D), z(B)), speaker=(z(B, S_MAX, D), z(B, S_MAX)), item=(z(B, D), z(B)))

class TagHead(nn.Module):

    def __init__(self, table, n_text=384, d_model=512, d_k=512):
        super().__init__()
        self.table = table
        self.register_buffer('E', torch.zeros(table.n, n_text))
        self.register_buffer('valid', torch.as_tensor(table.valid), persistent=False)
        self.P = nn.Linear(n_text, d_k, bias=False)
        self.r = nn.Parameter(torch.zeros(table.n, d_k))
        self.b = nn.Parameter(torch.zeros(table.n))
        self.bq = nn.ParameterDict({q: nn.Parameter(torch.zeros(table.n)) for q in QUERIES})
        self.Q = nn.ModuleDict({q: nn.Linear(d_model, d_k) for q in QUERIES})
        self.pool_in = nn.ModuleDict({q: nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)) for q in POOLED})
        for q in QUERIES:
            self.register_buffer(f'cols_{q}', torch.tensor(table.cols[q], dtype=torch.long), persistent=False)

    def cols(self, q):
        return getattr(self, f'cols_{q}')

    def keys(self):
        return self.P(self.E) + self.r

    def pooled(self, h, spk, ustart, carry, h_utt=None):
        zin, new = ({}, {})
        for q in POOLED:
            hq = h_utt if q == 'utt' and h_utt is not None else h
            z = hq + self.pool_in[q](hq)
            if q == 'speaker':
                zin[q], new[q] = pool_speaker(z.float(), spk, carry[q])
            elif q == 'utt':
                zin[q], new[q] = pool_segment(z.float(), ustart, carry[q])
            else:
                zin[q], new[q] = pool_segment(z.float(), None, carry[q])
        return (zin, new)

    def logits_q(self, q, zq, dom, K=None):
        K = self.keys() if K is None else K
        c = self.cols(q)
        z = self.Q[q](zq.float()) @ K[c].T + (self.b + self.bq[q])[c]
        return z.masked_fill(~self.valid[dom][:, c][:, None, :], NEG)
NAME_MAX_BYTES = 40
C_KINDS = ['person', 'team', 'organization', 'place', 'work_product', 'event_name', 'legal_ref', 'game_entity', 'game_map', 'other']
MENTION_KINDS = {'entity.mention.person': [0, 9], 'entity.mention.team': [1, 2], 'entity.mention.organization': [2, 1, 9], 'entity.mention.place': [3, 9], 'entity.mention.work_product': [4, 9], 'entity.mention.event_name': [5, 9], 'entity.mention.legal_ref': [6, 9], 'entity.mention.game_entity': [7, 9], 'entity.mention.game_map': [8, 9]}
K_SLOTS = 4
CLOCK = 3
SPAN_G_MULT = 20.0

def encode_names(names, max_len=NAME_MAX_BYTES, keep='head'):
    import numpy as np
    ids = np.zeros((len(names), max_len), np.int64)
    for i, n in enumerate(names):
        b = ' '.join(str(n).lower().split()).encode('utf-8')
        b = b[:max_len] if keep == 'head' else b[-max_len:]
        ids[i, :len(b)] = np.frombuffer(b, np.uint8).astype(np.int64) + 1
    return ids

class NameEncoder(nn.Module):

    def __init__(self, d=128, char_dim=48):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(257, char_dim, padding_idx=0)
        self.conv_in = nn.Conv1d(char_dim, d, 3, padding=1)
        self.convs = nn.ModuleList([nn.Conv1d(d, d, 3, padding=dl, dilation=dl) for dl in (1, 2, 4)])
        self.kind = nn.Embedding(len(C_KINDS), d)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, ids, kinds):
        sh = ids.shape[:-1]
        ids = ids.reshape(-1, ids.shape[-1])
        m = (ids > 0).float()[:, None]
        x = self.emb(ids).float().transpose(1, 2)
        h = F.gelu(self.conv_in(x)) * m
        for cv in self.convs:
            h = h + F.gelu(cv(h)) * m
        mean = h.sum(-1) / m.sum(-1).clamp(min=1.0)
        mx = h.masked_fill(m == 0, -10000.0).max(-1).values * (m.sum(-1) > 0).float()
        e = self.out(torch.cat([mean, mx], -1)) + self.kind(kinds.reshape(-1))
        return F.normalize(e, dim=-1).reshape(*sh, self.d)

class EntityHeads(nn.Module):

    def __init__(self, d_model=512, d_key=512, d=128, n_units=64, span=True, span_new=True, sim=False):
        super().__init__()
        self.names = NameEncoder(d)
        self.query = nn.Sequential(nn.Linear(d_model, d), nn.GELU(), nn.Linear(d, d))
        self.key_in = nn.Linear(d_key, d_model, bias=False)
        self.special = nn.Parameter(torch.zeros(2, d))
        self.special_b = nn.Parameter(torch.zeros(2))
        self.log_scale = nn.Parameter(torch.tensor([math.log(10.0)]))
        self.nwt = nn.Linear(d_model, K_SLOTS * 43)
        self.unit = nn.Linear(d_model, n_units + 1)
        self.clock = nn.Linear(d_model, CLOCK)
        self.approx = nn.Linear(d_model, 2)
        self.n_units = n_units
        self.span_q = nn.Linear(d, d) if span else None
        if span:
            self.span_g = nn.Parameter(torch.zeros(1))
        if span and span_new:
            self.span_new = nn.Parameter(torch.zeros(1))
        self.sim_w = None
        if sim:
            self.sim_w = nn.Parameter(torch.zeros(3, 6))
            self.sim_new = nn.Parameter(torch.zeros(3, 3))
            if span:
                self.span_g_src = nn.Parameter(torch.zeros(3))

    def ptr_logits(self, h, cls_key, E, cmask, Es=None, Sf=None, src=0):
        qr = self.query(h.float() + self.key_in(cls_key.float()))
        use_span = Es is not None and self.span_q is not None
        if use_span:
            qr = qr + self.span_q(Es.float())
        q = F.normalize(qr, dim=-1)
        sc = self.log_scale.clamp(max=math.log(50.0)).exp()
        s = torch.einsum('btd,brd->btr', q, E.float()) * sc
        if use_span:
            g_ = self.span_g if getattr(self, 'span_g_src', None) is None else self.span_g + self.span_g_src[int(src)]
            s = s + SPAN_G_MULT * g_ * sc * torch.einsum('btd,brd->btr', Es.float(), E.float())
        use_sim = Sf is not None and self.sim_w is not None
        if use_sim:
            Sf = Sf.float()
            s = s + SPAN_G_MULT * sc * torch.einsum('btrf,f->btr', Sf, self.sim_w[int(src)])
        s = s.masked_fill(~cmask, NEG)
        sp = torch.einsum('btd,kd->btk', q, F.normalize(self.special, dim=-1)) * sc + self.special_b
        if use_sim:
            hs = (Sf.abs().sum((-1, -2)) > 0).float()
            mb = Sf[..., 4].masked_fill(~cmask, 0.0).max(-1).values
            msh = Sf[..., 5].masked_fill(~cmask, 0.0).max(-1).values
            sn = self.sim_new[int(src)]
            tn = sn[0] + sn[1] * (1.0 - mb) + sn[2] * (1.0 - msh)
            sp = torch.cat([sp[..., :1] + (SPAN_G_MULT * sc * hs * tn)[..., None], sp[..., 1:]], -1)
        if use_span and getattr(self, 'span_new', None) is not None:
            ms = torch.einsum('btd,brd->btr', Es.float(), E.float()).masked_fill(~cmask, -1.0).max(-1).values.clamp(min=-1.0)
            has = (Es.float().abs().sum(-1) > 0).float()
            sp = torch.cat([sp[..., :1] + SPAN_G_MULT * self.span_new * sc * has[..., None] * (1.0 - ms[..., None]), sp[..., 1:]], -1)
        return torch.cat([sp, s], -1)

    def span_embed(self, text):
        ids = torch.as_tensor(encode_names([text], keep='tail'), device=self.special.device)
        return self.names(ids, torch.zeros(1, dtype=torch.long, device=ids.device))[0]

    def value_logits(self, h):
        return dict(nwt=self.nwt(h).float().reshape(*h.shape[:-1], K_SLOTS, -1), unit=self.unit(h).float(), clock=self.clock(h).float(), approx=self.approx(h).float())

class BioHead(nn.Module):
    EV = ('cough', 'throat_clear', 'breath', 'sneeze')

    def __init__(self, d_model=512):
        super().__init__()
        self.ev = nn.Linear(d_model, len(self.EV))

class DiarHeads(nn.Module):

    def __init__(self, d_model=512, d_in=640, k=8, d=128, hidden=256, n_extra=0):
        super().__init__()
        self.k = k + n_extra
        self.slot = nn.Linear(d_model, 1 + k)
        self.h_proj = nn.Linear(d_model, hidden)
        self.x_norm = nn.LayerNorm(d_in)
        self.x_proj = nn.Linear(d_in, hidden)
        self.out = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.slot_ext = nn.Linear(d_model, n_extra) if n_extra else None

    def slot_logits(self, h):
        z = self.slot(h)
        return torch.cat([z, self.slot_ext(h)], -1) if self.slot_ext is not None else z

    def embed(self, h, x):
        return F.normalize(self.out(self.h_proj(h.float()) + self.x_proj(self.x_norm(x.float()))), dim=-1)

class ArcNet(nn.Module):

    def __init__(self, table, cfg):
        super().__init__()
        self.trunk = Trunk(**cfg['trunk'])
        d = self.trunk.cfg['d_model']
        self.head = TagHead(table, n_text=int(cfg.get('n_text', 384)), d_model=d)
        self.ent = EntityHeads(d_model=d, d_key=d, n_units=int(cfg.get('n_units', 64)), span=bool(cfg.get('ptr_span_text')), span_new=bool(cfg.get('ptr_span_new')), sim=bool(cfg.get('ptr_sim_feat')))
        self.diar = DiarHeads(d_model=d, d_in=self.trunk.cfg['d_in'], n_extra=max(0, int(cfg.get('diar_slots', 8)) - 8))
        self.bio = BioHead(d) if cfg.get('bio_head') else None
        self.word = None
        if cfg.get('word_ch'):
            from .wordch import WordChannel
            self.word = WordChannel(d_model=d, d_k=d, n_in=len(table.cols['span_in']), k_slots=K_SLOTS, n_nwt=int(cfg.get('n_nwt', 43)))
        self.cfg = cfg

    def trunk_run(self, x, dom, delay, state=None, recurrent=False):
        return (self.trunk.step if recurrent else self.trunk)(x, dom, delay, state)
