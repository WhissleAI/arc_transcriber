from __future__ import annotations
import math
import torch

def rotate_by(v: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    a, b = (v[..., 0::2, :], v[..., 1::2, :])
    c, s = (torch.cos(ang).unsqueeze(-1), torch.sin(ang).unsqueeze(-1))
    even, odd = (c * a - s * b, s * a + c * b)
    return torch.stack((even, odd), dim=-2).flatten(-3, -2)
