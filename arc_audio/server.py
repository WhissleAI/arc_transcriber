import glob, json, os, tarfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
SR, HOP, NFFT, WIN = (16000, 160, 512, 400)
FRAME_STRIDE = 1280
MS_STRIDE = 320
STRIDE, RF = (320, 3920)
_REN = [('.pre_encode.conv.', '.subsampling.layers.'), ('.pre_encode.out.', '.subsampling.linear.'), ('.self_attn.linear_q.', '.self_attn.q_proj.'), ('.self_attn.linear_k.', '.self_attn.k_proj.'), ('.self_attn.linear_v.', '.self_attn.v_proj.'), ('.self_attn.linear_out.', '.self_attn.o_proj.'), ('.self_attn.linear_pos.', '.self_attn.relative_k_proj.'), ('.self_attn.pos_bias_u', '.self_attn.bias_u'), ('.self_attn.pos_bias_v', '.self_attn.bias_v'), ('.conv.batch_norm.', '.conv.norm.')]

def fetch_backbone(spec):
    p = os.environ.get('ARC_PARAKEET_NEMO')
    if p and os.path.exists(p):
        return p
    from huggingface_hub import hf_hub_download
    return hf_hub_download(spec.get('repo', 'nvidia/parakeet-tdt-0.6b-v2'), spec.get('file', 'parakeet-tdt-0.6b-v2.nemo'), revision=spec.get('revision'))

def load_nemo_state(path):
    with tarfile.open(path) as tf:
        name = next((m.name for m in tf.getmembers() if m.name.endswith('model_weights.ckpt')))
        sd = torch.load(tf.extractfile(name), map_location='cpu', weights_only=True)
    return {k: v for k, v in sd.items() if k.startswith(('encoder.', 'preprocessor.'))}

def map_keys(sd):
    out = {}
    for k, v in sd.items():
        if not k.startswith('encoder.'):
            continue
        k2 = '.' + k[len('encoder.'):]
        for a, b in _REN:
            k2 = k2.replace(a, b)
        out[k2[1:]] = v
    return out

def encoder_config():
    from transformers import ParakeetEncoderConfig
    return ParakeetEncoderConfig(hidden_size=1024, num_hidden_layers=24, num_attention_heads=8, intermediate_size=4096, hidden_act='silu', attention_bias=False, convolution_bias=False, conv_kernel_size=9, subsampling_factor=8, subsampling_conv_channels=256, num_mel_bins=128, subsampling_conv_kernel_size=3, subsampling_conv_stride=2, dropout=0.0, dropout_positions=0.0, layerdrop=0.0, activation_dropout=0.0, attention_dropout=0.0, max_position_embeddings=5000, scale_input=False)

class LogMel(nn.Module):

    def __init__(self, window, fb):
        super().__init__()
        self.register_buffer('window', window.float(), persistent=False)
        self.register_buffer('fb', fb.float().reshape(fb.shape[-2], fb.shape[-1]), persistent=False)

    @torch.no_grad()
    def forward(self, x, lengths):
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            B, L = x.shape
            flen = lengths // HOP
            tm = torch.arange(L, device=x.device)[None] < lengths[:, None]
            x = torch.cat([x[:, :1], x[:, 1:] - 0.97 * x[:, :-1]], 1).masked_fill(~tm, 0.0)
            s = torch.stft(x, n_fft=NFFT, hop_length=HOP, win_length=WIN, center=True, window=self.window, return_complex=True, pad_mode='constant')
            s = torch.view_as_real(s)
            p = torch.sqrt(s.pow(2).sum(-1)).pow(2.0)
            m = torch.log(torch.matmul(self.fb, p) + 2 ** (-24))
            T = m.shape[-1]
            v = torch.arange(T, device=x.device)[None] < flen[:, None]
            n = v.sum(1).clamp(min=1)[:, None].float()
            mu = torch.where(v[:, None], m, 0.0).sum(2) / n
            sd = torch.sqrt(torch.where(v[:, None], m - mu[..., None], 0.0).pow(2).sum(2) / (n - 1).clamp(min=1))
            sd = sd.masked_fill(sd.isnan(), 0.0) + 1e-05
            m = ((m - mu[..., None]) / sd[..., None]).masked_fill(~v[:, None], 0.0)
        return (m.transpose(1, 2).contiguous(), flen)

class Backbone(nn.Module):

    def __init__(self, path, tap=12, device='cpu', half=None):
        super().__init__()
        from transformers import ParakeetEncoder
        sd = load_nemo_state(path)
        for k in list(sd):
            if sd[k].is_floating_point() and (not k.startswith('preprocessor.')):
                sd[k] = sd[k].half()
        cfg = encoder_config()
        cfg._attn_implementation = 'sdpa'
        try:
            from transformers.initialization import no_init_weights
            with no_init_weights():
                self.enc = ParakeetEncoder(cfg)
        except ImportError:
            self.enc = ParakeetEncoder(cfg)
        self.enc.half()
        miss, unexp = self.enc.load_state_dict(map_keys(sd), strict=False, assign=True)
        miss = [k for k in miss if not k.endswith('inv_freq')]
        if miss or unexp:
            raise RuntimeError(f'backbone weights mismatch: {miss[:5]} {unexp[:5]}')
        self.mel = LogMel(sd['preprocessor.featurizer.window'], sd['preprocessor.featurizer.fb'])
        del sd
        self.tap = tap
        half = str(device).startswith('cuda') if half is None else half
        self.dtype = torch.float16 if half else torch.float32
        self.enc.to(device=device, dtype=self.dtype).eval()
        self.mel.to(device)
        for p in self.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, x, lengths, want_tap=True):
        feats, flen = self.mel(x, lengths)
        e = self.enc
        dev = x.device
        am = torch.arange(feats.shape[1], device=dev)[None] < flen[:, None]
        with torch.autocast(dev.type, enabled=False):
            h = e.subsampling(feats.to(self.dtype), am.long())
            pos = e.encode_positions(h)
            om = e._get_output_attention_mask(am.long(), target_length=h.shape[1])
            att = (om[:, None, :] & om[:, :, None])[:, None]
            keep = om[..., None]
            h = h.masked_fill(~keep, 0.0)
            htap = h if want_tap and self.tap == 0 else None
            for i, layer in enumerate(e.layers):
                h = layer(h, attention_mask=att, position_embeddings=pos).masked_fill(~keep, 0.0)
                if want_tap and i + 1 == self.tap:
                    htap = h
        return (h, om, htap)

class MemProj(nn.Module):

    def __init__(self, d_in=1024, d=320, hid=1024):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, hid)
        self.fc2 = nn.Linear(hid, d)
        self.out = nn.LayerNorm(d)

    def forward(self, P):
        return self.out(self.fc2(F.gelu(self.fc1(self.ln(P.float())))))

class FrameProj(nn.Module):

    def __init__(self, d_in=1024, d_f=512, d_out=640, mlp=True, hid=1024):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d_f)
        self.mlp = nn.Sequential(nn.LayerNorm(2 * d_f), nn.Linear(2 * d_f, hid), nn.GELU(), nn.Linear(hid, d_out)) if mlp else None

    def frame(self, P):
        return self.proj(self.ln(P.float()))

    def head(self, X):
        return self.mlp(X) if self.mlp is not None else X

class Heads(nn.Module):

    def __init__(self, pos_off=2):
        super().__init__()
        self.pos_off = int(pos_off)
        self.mem = MemProj()
        self.arc = FrameProj(d_f=512, d_out=640, mlp=True)
        self.tap = FrameProj(d_f=320, d_out=640, mlp=False)

def frame_of_enc(g):
    return torch.ceil((g * STRIDE + RF) / SR * 10 - 1e-09).long() - 1

def ms_frames(n_samples):
    return int(n_samples) // MS_STRIDE + 1

def pool_range(Z, n_ms, k0, k1):
    D = Z.shape[1]
    out = torch.zeros(max(0, k1 - k0), 2 * D)
    if k1 <= k0:
        return out
    T = Z.shape[0]
    g0 = max(0, 5 * k0 - 12)
    g1 = min(n_ms, 5 * k1 + 5)
    g = torch.arange(g0, g1)
    f = frame_of_enc(g)
    sel = (f >= k0) & (f < k1)
    g, f = (g[sel], f[sel] - k0)
    if len(g) == 0:
        return out
    E = Z.index_select(0, torch.clamp(g // 4, max=T - 1)).float()
    nb = k1 - k0
    s = torch.zeros(nb, D).index_add(0, f, E)
    c = torch.zeros(nb).index_add(0, f, torch.ones(len(f)))
    mx = torch.full((nb, D), -1e+30).scatter_reduce(0, f[:, None].expand(-1, D), E, 'amax', include_self=True)
    x = torch.cat([s / c.clamp(min=1)[:, None], mx], 1)
    have = c > 0
    x[~have] = 0.0
    return x

def mem_with_pos(heads, pos_emb, P):
    M = heads.mem(P)
    T = M.shape[1]
    W = pos_emb.weight
    i = torch.arange(T, device=M.device)
    corr = W.index_select(0, torch.clamp(4 * i + heads.pos_off, max=W.shape[0] - 1)) - W.index_select(0, torch.clamp(i, max=W.shape[0] - 1))
    return M + corr.to(M.dtype)[None]

class EncOut(dict):
    __getattr__ = dict.get

class Shim(nn.Module):

    def __init__(self, bb, heads, pos_emb):
        super().__init__()
        self._bb = [bb]
        self.heads = heads
        self._pos = [pos_emb]
        self.config = None

    def forward(self, input_values=None, attention_mask=None, **kw):
        x = input_values
        lengths = attention_mask.sum(1) if attention_mask is not None else torch.full((x.shape[0],), x.shape[1], device=x.device)
        bb = self._bb[0]
        P, pm, _ = bb(x.float().to(next(self.heads.parameters()).device), lengths, want_tap=False)
        M = mem_with_pos(self.heads, self._pos[0], P)
        return EncOut(last_hidden_state=M, attention_mask=pm)

class ServerFeatures:

    def __init__(self, bb, heads, chunk_s=30.0, ctx_s=4.0, tap=True):
        self.bb, self.heads = (bb, heads)
        self.CH = int(round(chunk_s * SR / FRAME_STRIDE)) * FRAME_STRIDE
        self.CTX = int(round(ctx_s * SR / FRAME_STRIDE)) * FRAME_STRIDE
        self.want_tap = tap
        self.audio = np.zeros(0, np.float32)
        self.a0 = 0
        self.total = 0
        self.c = 0
        self.Z, self.Zt = ([], [])
        self.nP = 0
        self.k = 0
        self.dev = next(heads.parameters()).device

    def _chunk(self, c, N):
        s0 = max(0, c - self.CTX)
        s1 = min(N, c + self.CH + self.CTX)
        x = torch.as_tensor(self.audio[s0 - self.a0:s1 - self.a0], dtype=torch.float32, device=self.dev)[None]
        P, om, Pt = self.bb(x, torch.tensor([s1 - s0], device=self.dev), want_tap=self.want_tap)
        a = (c - s0) // FRAME_STRIDE
        b = a + (min(N, c + self.CH) - c + FRAME_STRIDE - 1) // FRAME_STRIDE
        b = min(b, int(om[0].sum()))
        with torch.no_grad():
            self.Z.append(self.heads.arc.frame(P[0, a:b]).float().cpu())
            if Pt is not None:
                self.Zt.append(self.heads.tap.frame(Pt[0, a:b]).float().cpu())
        self.nP += b - a

    def _emit(self, final):
        if not self.Z:
            return ([], [])
        Z = torch.cat(self.Z)
        self.Z = [Z]
        Zt = torch.cat(self.Zt) if self.Zt else None
        if Zt is not None:
            self.Zt = [Zt]
        if final:
            n_ms = ms_frames(self.total)
            k1 = int(frame_of_enc(torch.tensor(n_ms - 1))) + 1
        else:
            n_ms = 4 * self.nP
            k1 = max(0, (4 * self.nP + 8 + 4) // 5)
            while k1 > 0 and 5 * (k1 - 1) - 8 >= 4 * self.nP:
                k1 -= 1
        if k1 <= self.k:
            return ([], [])
        with torch.no_grad():
            X = self.heads.arc.head(pool_range(Z, n_ms, self.k, k1))
            Xt = self.heads.tap.head(pool_range(Zt, n_ms, self.k, k1)) if Zt is not None else torch.zeros(k1 - self.k, 640)
        j0 = int(frame_of_enc(torch.tensor(0)))
        for j in range(self.k, min(k1, j0)):
            X[j - self.k] = 0.0
            Xt[j - self.k] = 0.0
        self.k = k1
        X, Xt = (X.numpy(), Xt.numpy())
        return (list(X), list(Xt))

    def feed(self, pcm):
        pcm = np.asarray(pcm, np.float32).reshape(-1)
        self.audio = np.concatenate([self.audio, pcm])
        self.total += len(pcm)
        while self.total >= self.c + self.CH + self.CTX:
            self._chunk(self.c, self.total)
            self.c += self.CH
            keep = max(0, self.c - self.CTX)
            if keep > self.a0:
                self.audio = self.audio[keep - self.a0:]
                self.a0 = keep
        return self._emit(False)

    def flush(self):
        N = self.total
        while self.c < N:
            self._chunk(self.c, N)
            self.c += self.CH
        return self._emit(True)

def load_heads(path, device):
    from safetensors.torch import load_file
    sd = load_file(path)
    heads = Heads()
    heads.load_state_dict({k[len('adapter.'):]: v for k, v in sd.items() if k.startswith('adapter.')})
    return (heads.to(device).eval(), sd)
