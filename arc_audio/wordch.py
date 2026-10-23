import math, re
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
P_MAX = 8
K_CTX = 8
BACK = 6
ENC_WIN = 32
N_ENC = 2
RF = (ENC_WIN - 1) * N_ENC
OFF_CLIP = 32
SRC_ROW = dict(npz=0, ms=1, corrupt=2, none=3)
_PUNCT = re.compile('^[^\\w]+|[^\\w]+$')

def norm_word(w):
    return _PUNCT.sub('', ' '.join(str(w).lower().split()))

def own_ends(f0, f1):
    f0 = np.asarray(f0, np.int64)
    f1 = np.asarray(f1, np.int64)
    oe = f1.copy()
    nxt = np.r_[f0[1:], np.iinfo(np.int64).max]
    m = (f0 < nxt) & (nxt <= f1)
    oe[m] = nxt[m] - 1
    return np.maximum(oe, f0)

def utt_ends(f0, oe, gap=4):
    W = len(f0)
    if not W:
        return np.zeros(0, np.int64)
    brk = np.r_[f0[1:] - oe[:-1] > gap, True]
    ue = np.where(brk, oe, np.iinfo(np.int64).max)
    return np.minimum.accumulate(ue[::-1])[::-1]

def rule_arrival(f0, oe, d, mode='final', Lf=15, L=8):
    if not len(f0):
        return np.zeros(0, np.int64)
    a = oe + L if mode == 'partial' else utt_ends(f0, oe) + Lf
    a = np.maximum(a, oe + d)
    return np.maximum.accumulate(a) - d

class Pieces:

    def __init__(self, tok_path):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(tok_path)
        self.c = {}

    def __call__(self, w):
        if w not in self.c:
            ids = self.t.encode(w, add_special_tokens=False).ids if w else []
            if len(ids) > P_MAX:
                ids = ids[:P_MAX - 1] + ids[-1:]
            if len(self.c) > 200000:
                self.c.clear()
            self.c[w] = ids
        return self.c[w]

def word_rows(texts, f0, oe, src_row, pieces, prev_oe=None):
    W = len(texts)
    tok = np.zeros((W, P_MAX), np.int64)
    npc = np.zeros(W, np.int64)
    flags = np.zeros((W, 5), np.int64)
    for i, w in enumerate(texts):
        p = pieces(w)
        npc[i] = len(p)
        tok[i, :len(p)] = p
        hd = any((ch.isdigit() for ch in w))
        flags[i, 0] = int(hd)
        flags[i, 1] = int(hd and w.isdigit())
    flags[:, 2] = np.clip(npc, 0, P_MAX)
    if W:
        flags[:, 3] = np.clip(oe - f0 + 1, 1, 20)
        flags[:, 4] = np.clip(np.r_[20 if prev_oe is None else f0[0] - prev_oe - 1, f0[1:] - oe[:-1] - 1], 0, 20)
    return (tok, npc, flags)

def frame_window(f0, avail, t0, t1, K=K_CTX, back=BACK):
    t = np.arange(t0, t1)
    n = np.searchsorted(avail, t, side='right')
    p = np.searchsorted(f0, t, side='right')
    q = np.minimum(p, n)
    idx = q[:, None] - back + np.arange(K)[None]
    ok = (idx >= 0) & (idx < n[:, None])
    return np.where(ok, idx, -1)

def _rope(x, pos):
    dh = x.shape[-1]
    inv = 1.0 / 10000 ** (torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh)
    ang = pos.float()[..., None] * inv
    while ang.dim() < x.dim():
        ang = ang.unsqueeze(1)
    c, s_ = (torch.cos(ang), torch.sin(ang))
    x1, x2 = (x[..., 0::2], x[..., 1::2])
    return torch.stack([x1 * c - x2 * s_, x1 * s_ + x2 * c], -1).flatten(-2)

class EncLayer(nn.Module):

    def __init__(self, d=256, h=4, ff=512, win=ENC_WIN, dropout=0.1):
        super().__init__()
        self.h, self.win = (h, win)
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, valid, pos0=0):
        Bn, W, D = x.shape
        h, w = (self.h, self.win)
        dh = D // h
        q, k, v = self.qkv(self.n1(x)).float().view(Bn, W, 3, h, dh).unbind(2)
        pos = (torch.arange(W, device=x.device) + pos0)[None].expand(Bn, W)
        q = _rope(q.transpose(1, 2), pos)
        k = _rope(k.transpose(1, 2), pos)
        v = v.transpose(1, 2)
        nb = (W + w - 1) // w
        Wp = nb * w
        pad = Wp - W
        if pad:
            q, k, v = (F.pad(t_, (0, 0, 0, pad)) for t_ in (q, k, v))
        vm = F.pad(valid, (0, pad)) if pad else valid
        kp = F.pad(k, (0, 0, w, 0)).view(Bn, h, nb + 1, w, dh)
        vp = F.pad(v, (0, 0, w, 0)).view(Bn, h, nb + 1, w, dh)
        kb = torch.cat([kp[:, :, :-1], kp[:, :, 1:]], 3)
        vb = torch.cat([vp[:, :, :-1], vp[:, :, 1:]], 3)
        qb = q.view(Bn, h, nb, w, dh)
        qi = torch.arange(nb, device=x.device)[:, None] * w + torch.arange(w, device=x.device)[None]
        ki = torch.arange(nb, device=x.device)[:, None] * w - w + torch.arange(2 * w, device=x.device)[None]
        mp = F.pad(vm, (w, 0)).view(Bn, nb + 1, w)
        km = torch.cat([mp[:, :-1], mp[:, 1:]], 2)
        m = (ki[:, None, :] <= qi[:, :, None]) & (ki[:, None, :] > qi[:, :, None] - w) & (ki[:, None, :] >= 0)
        m = m[None] & km[:, :, None, :]
        m = m | (ki[:, None, :] == qi[:, :, None])[None]
        a = F.scaled_dot_product_attention(qb.permute(0, 2, 1, 3, 4).reshape(Bn * nb, h, w, dh), kb.permute(0, 2, 1, 3, 4).reshape(Bn * nb, h, 2 * w, dh), vb.permute(0, 2, 1, 3, 4).reshape(Bn * nb, h, 2 * w, dh), attn_mask=m.reshape(Bn * nb, 1, w, 2 * w))
        a = a.view(Bn, nb, h, w, dh).permute(0, 2, 1, 3, 4).reshape(Bn, h, Wp, dh)[:, :, :W].transpose(1, 2).reshape(Bn, W, D)
        x = x + self.drop(self.o(a.to(x.dtype)))
        return x + self.drop(self.ff(self.n2(x)))

class WordChannel(nn.Module):

    def __init__(self, d_model=512, d_k=512, n_in=41, k_slots=4, n_nwt=43, d_w=256, d_c=128, heads_c=4, n_tok=32768, d_e=320):
        super().__init__()
        self.register_buffer('tok_E', torch.zeros(n_tok, d_e, dtype=torch.float16))
        self.ppos = nn.Parameter(torch.zeros(P_MAX, d_e))
        self.W1 = nn.Linear(d_e, d_w)
        self.W2 = nn.Linear(d_e, d_w, bias=False)
        self.ln = nn.LayerNorm(d_w)
        self.fl = nn.ModuleList([nn.Embedding(n, d_w) for n in (2, 2, P_MAX + 1, 21, 21)])
        self.src = nn.Embedding(len(SRC_ROW) + 1, d_w)
        self.Wa = nn.Linear(2 * d_model, d_w)
        self.hav = nn.Embedding(2, d_w)
        self.enc = nn.ModuleList([EncLayer(d_w) for _ in range(N_ENC)])
        self.enc_norm = nn.LayerNorm(d_w)
        self.hc = heads_c
        self.fq = nn.Linear(d_model, d_c)
        self.fk = nn.Linear(d_w, d_c)
        self.fv = nn.Linear(d_w, d_c)
        self.R0 = nn.Embedding(2 * OFF_CLIP + 1, d_c)
        self.R1 = nn.Embedding(2 * OFF_CLIP + 1, d_c)
        self.fo = nn.Linear(d_c, d_model)
        self.war_r = nn.Linear(d_w, d_w)
        self.war_k = nn.Embedding(3, d_w)
        self.war_ln = nn.LayerNorm(d_w)
        self.QB = nn.Linear(d_w, d_k)
        self.QI = nn.Linear(d_w, d_k)
        self.bB = nn.Parameter(torch.zeros(n_in))
        self.bI = nn.Parameter(torch.zeros(n_in))
        self.nwt = nn.Linear(d_w, k_slots * n_nwt)
        self.k_slots, self.n_nwt = (k_slots, n_nwt)

    def embed_in(self, tok, npc, flags, src, hm, he, have):
        Et = self.tok_E[tok].float()
        pm = (torch.arange(P_MAX, device=tok.device)[None, None] < npc[..., None]).float()[..., None]
        mean = ((Et + self.ppos) * pm).sum(2) / npc.clamp(min=1)[..., None].float()
        last = torch.gather(Et, 2, (npc - 1).clamp(min=0)[..., None, None].expand(-1, -1, 1, Et.shape[-1]))[:, :, 0]
        e = self.ln(self.W1(mean) + self.W2(last))
        for i, m in enumerate(self.fl):
            e = e + m(flags[..., i].clamp(0, m.num_embeddings - 1))
        e = e + self.src(src.clamp(0, self.src.num_embeddings - 1))
        return e + self.Wa(torch.cat([hm, he], -1)) + self.hav(have.long())

    def run_enc(self, x, valid, pos0=0):
        for lay in self.enc:
            x = lay(x, valid, pos0)
        return self.enc_norm(x) * valid[..., None].float()

    def frame_ctx(self, h, u, f0, oe, wctx, t0=0):
        Bn, T, Kc = wctx.shape
        ok = wctx >= 0
        idx = wctx.clamp(min=0)
        dc = self.fk.out_features
        hh = self.hc
        dh = dc // hh
        flat = idx.reshape(Bn, T * Kc)
        gk = torch.gather(self.fk(u), 1, flat[..., None].expand(-1, -1, dc)).view(Bn, T, Kc, dc)
        gv = torch.gather(self.fv(u), 1, flat[..., None].expand(-1, -1, dc)).view(Bn, T, Kc, dc)
        t = torch.arange(t0, t0 + T, device=h.device)[None, :, None]
        o0 = (t - torch.gather(f0, 1, flat).view(Bn, T, Kc)).clamp(-OFF_CLIP, OFF_CLIP) + OFF_CLIP
        o1 = (t - torch.gather(oe, 1, flat).view(Bn, T, Kc)).clamp(-OFF_CLIP, OFF_CLIP) + OFF_CLIP
        gk = gk + self.R0(o0) + self.R1(o1)
        q = self.fq(h).view(Bn, T, 1, hh, dh)
        sc = (q * gk.view(Bn, T, Kc, hh, dh)).sum(-1).float() / math.sqrt(dh)
        sc = sc.masked_fill(~ok[..., None], float('-inf'))
        any_ = ok.any(-1)
        att = torch.softmax(sc, 2).nan_to_num(0.0)
        ctx = (att[..., None] * gv.view(Bn, T, Kc, hh, dh).float()).sum(2).reshape(Bn, T, dc)
        return self.fo(ctx.to(h.dtype)) * any_[..., None].to(h.dtype)

    def war_logits(self, u, valid, K_in, k):
        Bn, W, D = u.shape
        if k:
            us = F.pad(u[:, k:], (0, 0, 0, k))
            ok = F.pad(valid[:, k:], (0, k)) & valid
        else:
            us = torch.zeros_like(u)
            ok = valid
        v = self.war_ln(u + self.war_r(us) + self.war_k.weight[k])
        return (self.QB(v).float() @ K_in.T + self.bB, self.QI(v).float() @ K_in.T + self.bI, ok, v)

    @torch.no_grad()
    def war_probs(self, u, valid, K_in, k=1):
        zB, zI, ok, v = self.war_logits(u, valid, K_in, k)
        if k:
            zB0, zI0, _, _ = self.war_logits(u, valid, K_in, k - 1) if k > 1 else self.war_logits(u, valid, K_in, 0)
            zB = torch.where(ok[..., None], zB, zB0)
            zI = torch.where(ok[..., None], zI, zI0)
        p = torch.softmax(torch.stack([torch.zeros_like(zB), zB, zI], -1), -1)
        return (p[..., 1], p[..., 2], v)

    def nwt_logits(self, v):
        return self.nwt(v).float().reshape(*v.shape[:-1], self.k_slots, self.n_nwt)

class WordFeed:

    def __init__(self, wch, pieces, dev, d):
        self.w, self.P, self.dev, self.d = (wch, pieces, dev, int(d))
        self.keys = []
        self.txt, self.f0, self.oe, self.f1, self.src, self.av = ([], [], [], [], [], [])
        self.X = None
        self.U = None
        self.n_x = 0
        self.ver = None

    def __len__(self):
        return len(self.keys)

    def sync(self, finals, f_now, arrival=None, src_of=None):
        ws = [w for w in finals if norm_word(w['w'])]
        keys = [(w['start'], w['end'], w['w']) for w in ws]
        n0 = 0
        m = min(len(keys), len(self.keys))
        while n0 < m and keys[n0] == self.keys[n0]:
            n0 += 1
        if n0 == len(keys) == len(self.keys):
            return False
        for L in (self.keys, self.txt, self.f0, self.oe, self.f1, self.src, self.av):
            del L[n0:]
        self.n_x = min(self.n_x, n0)
        if self.X is not None:
            self.X = self.X[:self.n_x]
        if self.U is not None:
            self.U = self.U[:min(len(self.U), n0)]
        for j in range(n0, len(ws)):
            w = ws[j]
            self.keys.append(keys[j])
            self.txt.append(norm_word(w['w']))
            self.f0.append(int(w['f0']))
            self.f1.append(int(w['f1']))
            self.src.append(src_of(w) if src_of else SRC_ROW['ms'])
            self.av.append(None)
        f0 = np.asarray(self.f0, np.int64)
        oe = own_ends(f0, np.asarray(self.f1, np.int64))
        old = self.oe[:min(n0, self.n_x)]
        self.oe = [int(x) for x in oe]
        self.oe[:len(old)] = old
        if arrival is not None:
            self.av = [int(x) for x in arrival]
        else:
            for j in range(n0, len(self.keys)):
                a = max(int(f_now), self.oe[j])
                if j:
                    a = max(a, self.av[j - 1])
                self.av[j] = a
        return True

    @torch.no_grad()
    def encode_upto(self, f_done, h_at, f_start=None, settle=4):
        n_new = 0
        while self.n_x + n_new < len(self.keys) and self.oe[self.n_x + n_new] < f_done:
            k = self.n_x + n_new
            if k + 1 >= len(self.keys) and f_done <= self.f1[k] + settle:
                break
            n_new += 1
        if not n_new:
            return
        if f_start is not None:
            for k in range(self.n_x, len(self.av)):
                lo_ = f_start if k < self.n_x + n_new else self.av[k - 1]
                self.av[k] = max(self.av[k], lo_, self.av[k - 1] if k else lo_)
        J = list(range(self.n_x, self.n_x + n_new))
        f0 = np.asarray([self.f0[j] for j in J])
        oe = np.asarray([self.oe[j] for j in J])
        tok, npc, flags = word_rows([self.txt[j] for j in J], f0, oe, None, self.P, prev_oe=self.oe[J[0] - 1] if J[0] > 0 else None)
        hm, he = ([], [])
        for a, b in zip(f0, oe):
            H = h_at(list(range(int(a), int(b) + 1))).float()
            hm.append(H.mean(0))
            he.append(H[-1])
        T = lambda a, dt=torch.long: torch.as_tensor(np.asarray(a), device=self.dev, dtype=dt)[None]
        x = self.w.embed_in(T(tok), T(npc), T(flags), T([self.src[j] for j in J]), torch.stack(hm)[None], torch.stack(he)[None], torch.ones(1, n_new, dtype=torch.bool, device=self.dev))[0]
        self.X = x if self.X is None or not len(self.X) else torch.cat([self.X, x], 0)
        self.n_x += n_new
        n_u = 0 if self.U is None else len(self.U)
        lo = max(0, n_u - RF)
        xs = self.X[lo:self.n_x][None]
        u = self.w.run_enc(xs, torch.ones(xs.shape[:2], dtype=torch.bool, device=self.dev), pos0=lo)[0][n_u - lo:]
        self.U = u if self.U is None or not len(self.U) else torch.cat([self.U, u], 0)

    @torch.no_grad()
    def fuse(self, h, f_start):
        T = h.shape[0]
        n_u = 0 if self.U is None else len(self.U)
        if not n_u:
            return h
        f0 = np.asarray(self.f0[:n_u], np.int64)
        av = np.asarray(self.av[:n_u], np.int64)
        idx = frame_window(f0, av, f_start, f_start + T)
        if (idx < 0).all():
            return h
        lo = int(idx[idx >= 0].min())
        hi = int(idx.max()) + 1
        loc = np.where(idx >= 0, idx - lo, -1)
        dv = self.dev
        u = self.U[lo:hi][None]
        F0 = torch.as_tensor(f0[lo:hi], device=dv)[None]
        OE = torch.as_tensor(np.asarray(self.oe[lo:hi], np.int64), device=dv)[None]
        c = self.w.frame_ctx(h[None], u, F0, OE, torch.as_tensor(loc, device=dv)[None], t0=f_start)[0]
        return h + c.to(h.dtype)
