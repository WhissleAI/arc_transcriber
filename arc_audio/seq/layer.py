from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def rotate_pairs(h: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    a, b = (h[..., 0::2], h[..., 1::2])
    c, s = (torch.cos(ang), torch.sin(ang))
    out = torch.empty_like(h)
    out[..., 0::2] = c * a - s * b
    out[..., 1::2] = s * a + c * b
    return out

def ssm_scan(alpha, beta, gamma, theta, S, Cm, P, state=None, prev_S=None):
    B, T, H, N, _ = S.shape
    R = Cm.shape[-1]
    h = torch.zeros(B, H, N, P, device=S.device, dtype=S.dtype) if state is None else state
    ys = []
    for t in range(T):
        prev = S[:, t - 1] if t > 0 else prev_S
        pre = alpha[:, t, :, None, None] * h
        if prev is not None:
            pre = pre + beta[:, t, :, None, None] * prev
        h = rotate_pairs(pre.transpose(-1, -2), theta[:, t].unsqueeze(-2).expand(-1, -1, P, -1)).transpose(-1, -2)
        h = h + gamma[:, t, :, None, None] * S[:, t]
        ys.append((Cm[:, t].transpose(-1, -2) @ h).reshape(B, H * R * P))
    return (torch.stack(ys, 1), h, S[:, -1])

class SeqLayer(nn.Module):

    def __init__(self, d_model: int, d_state: int=64, rank: int=1, head_dim: int=32, dt_min: float=0.001, dt_max: float=0.5):
        super().__init__()
        assert d_state % 2 == 0, 'd_state must be even: the state rotates in pairs'
        assert d_model % head_dim == 0, 'd_model must divide into heads of head_dim'
        self.d_model, self.N, self.R, self.P = (d_model, d_state, rank, head_dim)
        self.H = d_model // head_dim
        self.in_proj = nn.Linear(d_model, d_model * rank)
        self.B_proj = nn.Linear(d_model, self.H * d_state * rank)
        self.C_proj = nn.Linear(d_model, self.H * d_state * rank)
        self.x_proj = nn.Linear(d_model, d_model * rank)
        self.dt_proj = nn.Linear(d_model, self.H)
        self.theta_proj = nn.Linear(d_model, self.H * (d_state // 2))
        self.lam_proj = nn.Linear(d_model, self.H)
        nn.init.zeros_(self.lam_proj.weight)
        nn.init.zeros_(self.lam_proj.bias)
        self.out_proj = nn.Linear(d_model * rank, d_model)
        self.norm = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)
        rms = nn.RMSNorm if hasattr(nn, 'RMSNorm') else nn.LayerNorm
        self.B_norm = rms(d_state)
        self.C_norm = rms(d_state)
        self.B_bias = nn.Parameter(torch.ones(self.H, d_state, rank))
        self.C_bias = nn.Parameter(torch.ones(self.H, d_state, rank))
        self.A_log = nn.Parameter(torch.log(torch.rand(self.H) * 0.9 + 0.05))
        dt = torch.exp(torch.rand(self.H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.dt_proj.bias.data = dt + torch.log(-torch.expm1(-dt))

    def _coeffs(self, u: torch.Tensor):
        Bsz, T, _ = u.shape
        H, N, R, P = (self.H, self.N, self.R, self.P)
        dt = F.softplus(self.dt_proj(u))
        A = -torch.exp(self.A_log)
        alpha = torch.exp(dt * A)
        lam = torch.sigmoid(self.lam_proj(u))
        gamma = lam * dt
        beta = (1.0 - lam) * dt * alpha
        Bm = self.B_proj(u).view(Bsz, T, H, R, N)
        Bm = self.B_norm(Bm).permute(0, 1, 2, 4, 3) + self.B_bias
        Cm = self.C_proj(u).view(Bsz, T, H, R, N)
        Cm = self.C_norm(Cm).permute(0, 1, 2, 4, 3) + self.C_bias
        x = self.x_proj(u).view(Bsz, T, H, P, R)
        theta = self.theta_proj(u).view(Bsz, T, H, N // 2) * dt.unsqueeze(-1)
        return (alpha, beta, gamma, Bm, Cm, x, theta)

    def gate_and_proj(self, y_flat: torch.Tensor, nu: torch.Tensor):
        return self.out_proj(y_flat * F.silu(self.in_proj(nu)))

    def forward(self, u: torch.Tensor, state=None, return_state=False):
        nu = self.norm(u)
        alpha, beta, gamma, Bm, Cm, x, theta = self._coeffs(nu)
        S = Bm @ x.transpose(-1, -2)
        y_flat, h, _ = ssm_scan(alpha, beta, gamma, theta, S, Cm, self.P, state=state)
        out = self.gate_and_proj(y_flat, nu)
        return (out, h) if return_state else out

    def step(self, u_t: torch.Tensor, state, prev_bx=None):
        Bsz = u_t.shape[0]
        nu = self.norm(u_t.unsqueeze(1))
        alpha, beta, gamma, Bm, Cm, x, theta = self._coeffs(nu)
        H, N, P = (self.H, self.N, self.P)
        h = torch.zeros(Bsz, H, N, P, device=u_t.device, dtype=u_t.dtype) if state is None else state
        pre = alpha[:, 0, :, None, None] * h
        if prev_bx is not None:
            pre = pre + beta[:, 0, :, None, None] * prev_bx
        h = rotate_pairs(pre.transpose(-1, -2), theta[:, 0].unsqueeze(-2).expand(-1, -1, P, -1)).transpose(-1, -2)
        bx = Bm[:, 0] @ x[:, 0].transpose(-1, -2)
        h = h + gamma[:, 0, :, None, None] * bx
        y = (Cm[:, 0].transpose(-1, -2) @ h).reshape(Bsz, -1)
        return (self.gate_and_proj(y, nu[:, 0]), h, bx)
