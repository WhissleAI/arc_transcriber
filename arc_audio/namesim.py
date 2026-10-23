import re
import numpy as np
N_SIM = 6
DIM = 4096
_TOKRE = re.compile('[^a-z0-9 ]+')

def norm(s):
    return ' '.join(_TOKRE.sub(' ', str(s).lower()).split())

def phon(t):
    if not t:
        return ''
    if t.isdigit():
        return t
    s = t
    for a, b in (('tch', 'ch'), ('sch', 'sk'), ('ph', 'f'), ('ck', 'k'), ('dg', 'j'), ('gh', ''), ('qu', 'kw'), ('wr', 'r'), ('x', 'ks'), ('q', 'k'), ('z', 's'), ('v', 'f'), ('sh', 'x'), ('ch', 'x'), ('th', '0')):
        s = s.replace(a, b)
    if s.startswith(('kn', 'gn', 'pn')):
        s = s[1:]
    s = re.sub('c(?=[eiy])', 's', s).replace('c', 'k')
    head, rest = (s[:1], s[1:])
    rest = re.sub('[aeiouy]', '', rest)
    rest = re.sub('h', '', rest)
    rest = re.sub('w(?![aeiou])', '', rest)
    head = 'a' if head in 'aeiouy' else head
    out = head + rest
    return re.sub('(.)\\1+', '\\1', out)

def _bigram_rows(strs):
    M = np.zeros((len(strs), DIM), np.float32)
    for i, s in enumerate(strs):
        p = f' {s} '
        if len(p) >= 3:
            M[i, list({(ord(p[j]) * 131 + ord(p[j + 1])) % DIM for j in range(len(p) - 1)})] = 1.0
    return (M, M.sum(1))

def _dice(A, nA, B, nB):
    return 2.0 * (A @ B.T) / np.maximum(nA[:, None] + nB[None, :], 1.0)

class CandTable:

    def __init__(self, names):
        self.names = [norm(n) for n in names]
        R = len(self.names)
        self.full, self.nfull = _bigram_rows(self.names)
        self.phon, self.nphon = _bigram_rows([' '.join((phon(t) for t in n.split())) for n in self.names])
        toks = sorted({t for n in self.names for t in n.split() if len(t) >= 3})
        self.tok_i = {t: i for i, t in enumerate(toks)}
        self.tok, self.ntok = _bigram_rows(toks)
        self.tok_ph = np.array([phon(t) for t in toks], object)
        ix = [[self.tok_i[t] for t in n.split() if t in self.tok_i] for n in self.names]
        self.tix = np.full((R, max([1] + [len(x) for x in ix])), -1, np.int64)
        for r, x in enumerate(ix):
            self.tix[r, :len(x)] = x
        self.R = R

def sim_feats(spans, ct):
    S, R = (len(spans), ct.R)
    out = np.zeros((S, R, N_SIM), np.float16)
    if S == 0 or R == 0:
        return out
    sp = [norm(s) for s in spans]
    F, nF = _bigram_rows(sp)
    s_full = _dice(F, nF, ct.full, ct.nfull)
    P, nP = _bigram_rows([' '.join((phon(t) for t in s.split())) for s in sp])
    s_phon = _dice(P, nP, ct.phon, ct.nphon)
    s_tok = np.zeros((S, R), np.float32)
    s_ptok = np.zeros((S, R), np.float32)
    if len(ct.tok_i):
        stoks = sorted({t for s in sp for t in s.split() if len(t) >= 3})
        if stoks:
            T_, nT_ = _bigram_rows(stoks)
            D = _dice(T_, nT_, ct.tok, ct.ntok)
            PE = (np.array([phon(t) for t in stoks], object)[:, None] == ct.tok_ph[None, :]).astype(np.float32)
            ok = ct.tix >= 0
            g = ct.tix.clip(min=0)
            Dn = np.where(ok[None], D[:, g], 0.0).max(-1)
            Pn = np.where(ok[None], PE[:, g], 0.0).max(-1)
            si = {t: i for i, t in enumerate(stoks)}
            for k, s in enumerate(sp):
                ix = [si[t] for t in s.split() if t in si]
                if ix:
                    s_tok[k] = Dn[ix].max(0)
                    s_ptok[k] = Pn[ix].max(0)
    best = np.maximum.reduce([s_full, 0.9 * s_tok, 0.85 * s_ptok, 0.9 * s_phon])
    sh = np.clip((best - 0.6) / 0.4, 0, 1) ** 2
    out[...] = np.stack([s_full, s_tok, s_ptok, s_phon, best, sh], -1).astype(np.float16)
    return out

def span_texts(ids):
    out = []
    for row in np.asarray(ids):
        b = bytes((int(x) - 1 for x in row if x > 0))
        out.append(b.decode('utf-8', errors='ignore'))
    return out
