import torch
from .scan import rotate_by
from .layer import SeqLayer

def _f(t):
    return t.float() if t is not None and t.dtype in (torch.float16, torch.bfloat16) else t

def _shift(x):
    return torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)

def chunked_safe(alpha, Beff, xeff, Ctil, chunk: int=64):
    B, T, H, N, K = Beff.shape
    P = xeff.shape[-2]
    la = torch.log(alpha.clamp_min(1e-30))
    g = torch.zeros(B, H, N, P, dtype=Beff.dtype, device=Beff.device)
    ys = []
    for s0 in range(0, T, chunk):
        e = min(s0 + chunk, T)
        L = e - s0
        Bc, xc, Cc = (Beff[:, s0:e], xeff[:, s0:e], Ctil[:, s0:e])
        cl = torch.cumsum(la[:, s0:e], dim=1)
        d = torch.exp(cl)
        rel = cl.unsqueeze(2) - cl.unsqueeze(1)
        tril = torch.tril(torch.ones(L, L, dtype=torch.bool, device=Beff.device)).view(1, L, L, 1)
        Ldec = torch.exp(rel.masked_fill(~tril, float('-inf')))
        G = torch.einsum('blhnr,bmhnk->bhlmrk', Cc, Bc)
        y_intra = torch.einsum('blmh,bhlmrk,bmhpk->blhrp', Ldec, G, xc)
        y_inter = torch.einsum('blh,blhnr,bhnp->blhrp', d, Cc, g)
        ys.append(y_intra + y_inter)
        coef = torch.exp(cl[:, -1:, :] - cl)
        g = d[:, -1].unsqueeze(-1).unsqueeze(-1) * g + torch.einsum('blh,blhnk,blhpk->bhnp', coef, Bc, xc)
    return torch.cat(ys, dim=1)

def _chunked_from_coeffs(alpha, beta, gamma, Bc, xc, Cm, theta, chunk):
    Phi = torch.cumsum(theta, dim=1)
    Phi_prev = _shift(Phi)
    Btil = rotate_by(Bc, -Phi)
    Ctil = rotate_by(Cm, -Phi)
    Btil_prev = rotate_by(_shift(Bc), -Phi_prev)
    Beff = torch.cat([gamma[..., None, None] * Btil, beta[..., None, None] * Btil_prev], dim=-1)
    xeff = torch.cat([xc, _shift(xc)], dim=-1)
    return chunked_safe(alpha, Beff, xeff, Ctil, chunk)

def seq_chunked(layer: SeqLayer, u: torch.Tensor, chunk: int=64):
    nu = layer.norm(u)
    alpha, beta, gamma, Bm, Cm, x, theta = layer._coeffs(nu)
    y = _chunked_from_coeffs(alpha, beta, gamma, Bm, x, Cm, theta, chunk)
    Bsz, T = u.shape[:2]
    return layer.gate_and_proj(y.reshape(Bsz, T, -1), nu)

def chunked_safe_state(alpha, Beff, xeff, Ctil, g0=None, chunk: int=64):
    B, T, H, N, K = Beff.shape
    P = xeff.shape[-2]
    la = torch.log(alpha.clamp_min(1e-30))
    g = torch.zeros(B, H, N, P, dtype=Beff.dtype, device=Beff.device) if g0 is None else g0.to(Beff.dtype)
    ys = []
    for s0 in range(0, T, chunk):
        e = min(s0 + chunk, T)
        L = e - s0
        Bc, xc, Cc = (Beff[:, s0:e], xeff[:, s0:e], Ctil[:, s0:e])
        cl = torch.cumsum(la[:, s0:e], dim=1)
        d = torch.exp(cl)
        rel = cl.unsqueeze(2) - cl.unsqueeze(1)
        tril = torch.tril(torch.ones(L, L, dtype=torch.bool, device=Beff.device)).view(1, L, L, 1)
        Ldec = torch.exp(rel.masked_fill(~tril, float('-inf')))
        G = torch.einsum('blhnr,bmhnk->bhlmrk', Cc, Bc)
        y_intra = torch.einsum('blmh,bhlmrk,bmhpk->blhrp', Ldec, G, xc)
        y_inter = torch.einsum('blh,blhnr,bhnp->blhrp', d, Cc, g)
        ys.append(y_intra + y_inter)
        coef = torch.exp(cl[:, -1:, :] - cl)
        g = d[:, -1].unsqueeze(-1).unsqueeze(-1) * g + torch.einsum('blh,blhnk,blhpk->bhnp', coef, Bc, xc)
    return (torch.cat(ys, dim=1), g)

def seq_chunked_state(layer: SeqLayer, u: torch.Tensor, state=None, chunk: int=64):
    nu = layer.norm(u)
    alpha, beta, gamma, Bm, Cm, x, theta = layer._coeffs(nu)
    Phi = torch.cumsum(theta, dim=1)
    Phi_prev = _shift(Phi)
    if state is None:
        h0 = None
        Bp = torch.zeros_like(Bm[:, :1])
        xp = torch.zeros_like(x[:, :1])
    else:
        h0, Bl, xl = state
        Bp = Bl[:, None].to(Bm.dtype)
        xp = xl[:, None].to(x.dtype)
    Bsh = torch.cat([Bp, Bm[:, :-1]], 1)
    xsh = torch.cat([xp, x[:, :-1]], 1)
    Btil = rotate_by(Bm, -Phi)
    Ctil = rotate_by(Cm, -Phi)
    Btil_prev = rotate_by(Bsh, -Phi_prev)
    Beff = torch.cat([gamma[..., None, None] * Btil, beta[..., None, None] * Btil_prev], dim=-1)
    xeff = torch.cat([x, xsh], dim=-1)
    y, g = chunked_safe_state(alpha, Beff, xeff, Ctil, h0, chunk)
    hT = rotate_by(g.unsqueeze(1), Phi[:, -1:])[:, 0]
    Bsz, T = u.shape[:2]
    return (layer.gate_and_proj(y.reshape(Bsz, T, -1), nu), (hT, Bm[:, -1], x[:, -1]))
