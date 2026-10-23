import numpy as np
import torch
SR = 16000
ENC_HOP = 320
import contextlib, threading
_LOCK = threading.Lock()

@contextlib.contextmanager
def _eager(moon):
    cfgs = []
    for mod in moon.model.decoder.modules():
        c = getattr(mod, 'config', None)
        if c is not None and hasattr(c, '_attn_implementation') and all((c is not x for x, _ in cfgs)):
            cfgs.append((c, c._attn_implementation))
    with _LOCK:
        try:
            for c, _ in cfgs:
                c._attn_implementation = 'eager'
            yield
        finally:
            for c, v in cfgs:
                c._attn_implementation = v

def dtw(cost):
    N, F = cost.shape
    D = np.full((N + 1, F + 1), np.inf, np.float64)
    D[0, 0] = 0
    tr = np.zeros((N + 1, F + 1), np.int8)
    for i in range(1, N + 1):
        ci = cost[i - 1]
        for j in range(1, F + 1):
            a, b, c = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
            if a <= b and a <= c:
                D[i, j] = ci[j - 1] + a
                tr[i, j] = 0
            elif b <= c:
                D[i, j] = ci[j - 1] + b
                tr[i, j] = 1
            else:
                D[i, j] = ci[j - 1] + c
                tr[i, j] = 2
    i, j = (N, F)
    ti, fj = ([], [])
    while i > 0 and j > 0:
        ti.append(i - 1)
        fj.append(j - 1)
        t = tr[i, j]
        if t == 0:
            i -= 1
            j -= 1
        elif t == 1:
            i -= 1
        else:
            j -= 1
    return (np.array(ti[::-1]), np.array(fj[::-1]))

def _dtw_fast(cost):
    N, F = cost.shape
    if N * F < 4000:
        return dtw(cost)
    D = np.full((N + 1, F + 1), np.inf)
    D[0, 0] = 0
    tr = np.zeros((N + 1, F + 1), np.int8)
    for i in range(1, N + 1):
        diag = D[i - 1, :-1]
        up = D[i - 1, 1:]
        best = np.minimum(diag, up)
        bt = np.where(diag <= up, 0, 1).astype(np.int8)
        row = np.empty(F + 1)
        row[0] = np.inf
        ci = cost[i - 1]
        prev = np.inf
        for j in range(1, F + 1):
            b = best[j - 1]
            if b <= prev:
                v = ci[j - 1] + b
                tr[i, j] = bt[j - 1]
            else:
                v = ci[j - 1] + prev
                tr[i, j] = 2
            row[j] = v
            prev = v
        D[i] = row
    i, j = (N, F)
    ti, fj = ([], [])
    while i > 0 and j > 0:
        ti.append(i - 1)
        fj.append(j - 1)
        t = tr[i, j]
        if t == 0:
            i -= 1
            j -= 1
        elif t == 1:
            i -= 1
        else:
            j -= 1
    return (np.array(ti[::-1]), np.array(fj[::-1]))

def _dtw_vec(cost):
    N, F = cost.shape
    D = np.full((N + 1, F + 1), np.inf)
    D[0, 0] = 0
    tr = np.zeros((N + 1, F + 1), np.int8)
    for i in range(1, N + 1):
        diag = D[i - 1, :-1]
        up = D[i - 1, 1:]
        best = np.minimum(diag, up)
        bt = np.where(diag <= up, 0, 1).astype(np.int8)
        C = np.cumsum(cost[i - 1])
        Cp = np.concatenate([[0.0], C[:-1]])
        m = best - Cp
        acc = np.minimum.accumulate(m)
        D[i, 1:] = C + acc
        tr[i, 1:] = np.where(acc < m, 2, bt)
    i, j = (N, F)
    ti, fj = ([], [])
    while i > 0 and j > 0:
        ti.append(i - 1)
        fj.append(j - 1)
        t = tr[i, j]
        if t == 0:
            i -= 1
            j -= 1
        elif t == 1:
            i -= 1
        else:
            j -= 1
    return (np.array(ti[::-1]), np.array(fj[::-1]))

def _medfilt(x, w):
    if w <= 1 or x.shape[-1] < w:
        return x
    p = w // 2
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(p, p)], mode='edge')
    st = np.lib.stride_tricks.sliding_window_view(xp, w, axis=-1)
    return np.median(st, -1)

@torch.no_grad()
def cross_attention(moon, eo, ids, dev):
    D = torch.tensor([ids], device=dev, dtype=torch.long)
    with _eager(moon):
        out = moon(encoder_outputs=eo, decoder_input_ids=D, output_attentions=True)
    return np.stack([a[0].float().cpu().numpy() for a in out.cross_attentions])

def token_frames(att, heads=None, medfilt=7, fast=False):
    L = att.shape[0]
    if heads is None:
        heads = [(l, h) for l in range(L // 2, L) for h in range(att.shape[1])]
    A = np.stack([att[l, h, :-1] for l, h in heads])
    A = (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-06)
    A = _medfilt(A, medfilt).mean(0)
    return _dtw_vec(-A) if fast else _dtw_fast(-A)

def words_from_tokens(tok, ids, path, n_frames, t0, hop_s, offset_s, t_lo, t_hi):
    ti, fj = path
    first = {}
    last = {}
    for i, j in zip(ti, fj):
        first.setdefault(i, j)
        last[i] = j
    pieces = tok.convert_ids_to_tokens(ids)
    words = []
    for k, p in enumerate(pieces):
        new = p.startswith('▁') or not words
        txt = p.replace('▁', '')
        f0, f1 = (first.get(k, 0), last.get(k, 0))
        if new:
            words.append([txt, f0, f1])
        else:
            words[-1][0] += txt
            words[-1][2] = max(words[-1][2], f1)
    out = []
    for w, f0, f1 in words:
        if not w:
            continue
        s = t0 + f0 * hop_s + offset_s
        e = t0 + (f1 + 1) * hop_s + offset_s
        s = min(max(s, t_lo), t_hi)
        e = min(max(e, s + 0.04), t_hi)
        out.append(dict(w=w, start=round(s, 3), end=round(e, 3)))
    return out

def encode(moon, dev, dtype, wav):
    from .moon import encoder_out
    x = torch.as_tensor(np.asarray(wav, np.float32), device=dev)[None].to(dtype)
    return encoder_out(moon, x)

def text_ids(moon, tok, text):
    ids = tok(text).input_ids
    st = moon.config.decoder_start_token_id
    if not ids or ids[0] != st:
        ids = [st] + list(ids)
    return [i for i in ids if i != tok.eos_token_id]

@torch.no_grad()
def align_eo(moon, tok, dev, eo, n_samples, text, t0, cfg=None, return_att=False, hop_s=None):
    cfg = cfg or {}
    if not text or not text.strip():
        return []
    F = eo.last_hidden_state.shape[1]
    ids = text_ids(moon, tok, text)
    if len(ids) < 2 or F < 2:
        return []
    from .moon import states_output
    eo2 = states_output(moon, eo.last_hidden_state[0:1])
    att = cross_attention(moon, eo2, ids, dev)
    if hop_s is None:
        from .moon import is_streaming_backbone
        hop_s = ENC_HOP / SR if is_streaming_backbone(moon) else n_samples / SR / F
    heads = [tuple(h) for h in cfg['heads']] if cfg.get('heads') else None
    path = token_frames(att, heads, cfg.get('medfilt', 7), fast=bool(cfg.get('_fast_dtw')))
    W = words_from_tokens(tok, ids[1:], path, F, t0, hop_s, float(cfg.get('offset_s', 0.0)), t0, t0 + n_samples / SR)
    return (W, att, ids, hop_s) if return_att else W

@torch.no_grad()
def align_text(moon, tok, dev, dtype, wav, text, t0, cfg=None, return_att=False):
    if not text or not text.strip() or len(wav) < 800:
        return []
    return align_eo(moon, tok, dev, encode(moon, dev, dtype, wav), len(wav), text, t0, cfg, return_att)
