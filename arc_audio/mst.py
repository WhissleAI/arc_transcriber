import numpy as np
import torch
import torch.nn.functional as F
FRAME = 80
HOP = 320

def windows(enc):
    return [tuple(w) for w in enc.config.sliding_windows]

def lookahead_frames(enc):
    return sum((r - 1 for _, r in windows(enc) if r > 0))

def n_frames(n_samples):
    n5 = n_samples // FRAME
    return 0 if n5 == 0 else ((n5 + 1) // 2 + 1) // 2

class MSTStreamEncoder:

    def __init__(self, enc, dev=None, dtype=None):
        self.enc = enc
        p = next(enc.parameters())
        self.dev, self.dtype = (dev or p.device, dtype or p.dtype)
        self.em = enc.embedder
        self.win = windows(enc)
        self.L = len(enc.layers)
        self.pcm = np.zeros(0, np.float32)
        self.e = None
        self.e0 = 0
        self.n_e = 0
        self.c1, self.c10, self.n_c1 = (None, 0, 0)
        self.H = [None] * (self.L + 1)
        self.H0 = [0] * (self.L + 1)
        self.n_H = [0] * (self.L + 1)
        self.n_out = 0

    @torch.no_grad()
    def _embed_frames(self, x):
        x = torch.as_tensor(x, device=self.dev, dtype=self.dtype).reshape(1, -1, FRAME)
        h = self.em.cmvn(x)
        h = self.em.comp(h)
        return F.silu(self.em.linear(h))[0]

    @staticmethod
    def _conv(conv, buf, b0, n_have, n_done):
        J = (n_have + 1) // 2
        if J <= n_done or buf is None:
            return (None, n_done)
        lo, hi = (2 * n_done - 4, 2 * (J - 1) + 1)
        x = buf[max(lo, 0) - b0:hi - b0]
        if lo < 0:
            x = torch.cat([torch.zeros(-lo, x.shape[1], device=x.device, dtype=x.dtype), x], 0)
        y = F.conv1d(x.T[None], conv.weight, conv.bias, stride=2)[0].T
        return (y, J)

    @torch.no_grad()
    def _layer(self, l, q0, q1, k_hi):
        layer = self.enc.layers[l]
        left, right = self.win[l]
        k_lo = max(0, q0 - left + 1)
        base = self.H0[l]
        X = self.H[l][k_lo - base:k_hi - base][None]
        xn = layer.input_layernorm(X)
        at = layer.self_attn
        hd = at.head_dim
        Q = at.q_proj(xn[:, q0 - k_lo:q1 - k_lo]).view(1, q1 - q0, -1, hd).transpose(1, 2)
        K = at.k_proj(xn).view(1, X.shape[1], -1, hd).transpose(1, 2)
        V = at.v_proj(xn).view(1, X.shape[1], -1, hd).transpose(1, 2)
        qi = torch.arange(q0, q1, device=self.dev)[:, None]
        ki = torch.arange(k_lo, k_hi, device=self.dev)[None, :]
        d = qi - ki
        ok = (d >= 0) & (d < left) | (d < 0) & (-d < right)
        mask = torch.zeros(ok.shape, device=self.dev, dtype=Q.dtype).masked_fill(~ok, torch.finfo(Q.dtype).min)
        if at.num_key_value_groups > 1:
            K = K.repeat_interleave(at.num_key_value_groups, 1)
            V = V.repeat_interleave(at.num_key_value_groups, 1)
        w = torch.softmax(Q @ K.transpose(2, 3) * at.scaling + mask, dim=-1, dtype=torch.float32).to(Q.dtype)
        o = (w @ V).transpose(1, 2).reshape(1, q1 - q0, -1)
        h = X[:, q0 - k_lo:q1 - k_lo] + at.o_proj(o)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        return h[0]

    @staticmethod
    def _cat(buf, new):
        return new if buf is None else torch.cat([buf, new], 0)

    def _trim(self):
        keep = max((w[0] for w in self.win)) + 4
        for l in range(self.L + 1):
            if self.H[l] is None:
                continue
            need_from = self.n_H[l + 1] - keep if l < self.L else self.n_H[l]
            cut = max(0, need_from - self.H0[l])
            if cut > 0:
                self.H[l] = self.H[l][cut:]
                self.H0[l] += cut
        if self.e is not None:
            cut = max(0, 2 * self.n_c1 - 8 - self.e0)
            if cut > 0:
                self.e = self.e[cut:]
                self.e0 += cut
        if self.c1 is not None:
            cut = max(0, 2 * self.n_H[0] - 8 - self.c10)
            if cut > 0:
                self.c1 = self.c1[cut:]
                self.c10 += cut

    @torch.no_grad()
    def _advance(self, final):
        y, J = self._conv(self.em.conv1, self.e, self.e0, self.n_e, self.n_c1)
        if y is not None:
            self.c1 = self._cat(self.c1, F.silu(y))
            self.n_c1 = J
        y, J = self._conv(self.em.conv2, self.c1, self.c10, self.n_c1, self.n_H[0])
        if y is not None:
            self.H[0] = self._cat(self.H[0], y)
            self.n_H[0] = J
        for l in range(self.L):
            right = self.win[l][1]
            hi = self.n_H[l] if final else self.n_H[l] - max(0, right - 1)
            if hi > self.n_H[l + 1]:
                h = self._layer(l, self.n_H[l + 1], hi, self.n_H[l] if final else min(self.n_H[l], hi + max(0, right - 1)))
                self.H[l + 1] = self._cat(self.H[l + 1], h)
                self.n_H[l + 1] = hi
        out = None
        if self.n_H[self.L] > self.n_out:
            new = self.H[self.L][self.n_out - self.H0[self.L]:]
            out = self.enc.final_norm(new[None])[0]
            self.n_out = self.n_H[self.L]
        self._trim()
        return out

    def feed(self, pcm):
        self.pcm = np.concatenate([self.pcm, np.asarray(pcm, np.float32).reshape(-1)])
        n = len(self.pcm) // FRAME
        if n:
            self.e = self._cat(self.e, self._embed_frames(self.pcm[:n * FRAME]))
            self.n_e += n
            self.pcm = self.pcm[n * FRAME:]
        out = self._advance(final=False)
        return out if out is not None else torch.zeros(0, self.enc.config.hidden_size, device=self.dev, dtype=self.dtype)

    def flush(self):
        self.pcm = np.zeros(0, np.float32)
        out = self._advance(final=True)
        return out if out is not None else torch.zeros(0, self.enc.config.hidden_size, device=self.dev, dtype=self.dtype)

@torch.no_grad()
def encode_offline(enc, wav, chunk_s=30.0):
    s = MSTStreamEncoder(enc)
    outs = []
    n = int(chunk_s * 16000)
    for i in range(0, len(wav), n):
        outs.append(s.feed(wav[i:i + n]))
    outs.append(s.flush())
    return torch.cat(outs, 0).float().cpu().numpy()

@torch.no_grad()
def reference_offline(enc, wav):
    p = next(enc.parameters())
    n = len(wav) // FRAME * FRAME
    x = np.asarray(wav[:n], np.float32)
    X = torch.as_tensor(x, device=p.device, dtype=p.dtype)[None]
    return enc(X, attention_mask=torch.ones_like(X, dtype=torch.long)).last_hidden_state[0].float().cpu().numpy()
