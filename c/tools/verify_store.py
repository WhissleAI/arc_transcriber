"""Hashes every tensor of an Arc safetensors file as arc_audio loads it (sum of float32 bit patterns weighted by position, mod 2^64), to compare with
tests/store_hash.c.

Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import sys, numpy as np, torch
from arc_audio.weights import load_q4
from safetensors.torch import load_file
p = sys.argv[1]
try:
    sd = load_q4(p)
except Exception:
    sd = load_file(p)
for k, v in sd.items():
    bits = v.detach().to(torch.float32).contiguous().numpy().reshape(-1).view(np.uint32).astype(np.uint64)
    h = int(np.sum(bits * np.arange(1, len(bits) + 1, dtype=np.uint64), dtype=np.uint64))
    print(k, v.numel(), format(h, '016x'))
