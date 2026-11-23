"""Token n-gram LM shallow fusion, as libarc's arc_set_lm (c/src/arc_lm.c): the ARCLM1 file reader, a writer for the
format (for tests and converters; training is not part of this package) and the decoder's logits processor.

File (little endian, natural logs; docs/LIBARC.md#n-gram-lm):
    b"ARCLM1\\0\\0", int32 order (1-8), int32 vocab, float32 unscored, int32 cap (power of 2)
    uint8 scored[vocab], zero padded to a multiple of 8 bytes
    cap x {uint64 key, float32 logp, float32 bow}; key 0 = empty slot
key = lm_key(ids); slot (key ^ key >> 32) & (cap - 1), linear probing. logp > 0: the entry is only a context.
"""
import math
import struct

import numpy as np
import torch

try:
    from transformers import LogitsProcessor
except Exception:
    LogitsProcessor = object

MAGIC = b"ARCLM1\0\0"
MARGIN = 4.0    # ARC_LM_MARGIN: only candidates this close to the audio's best are scored by the LM
CAP = 6.0       # ARC_LM_CAP: the largest LM charge (times the weight)
MAX_ORDER = 8
_ENTRY = np.dtype([("key", "<u8"), ("logp", "<f4"), ("bow", "<f4")])
_M64 = 0xFFFFFFFFFFFFFFFF


def lm_key(ids):
    """FNV-1a 64 over the ids as little-endian int32 bytes, ^ n * 0x9E3779B97F4A7C15, | 1 (arc_lm_key)"""
    h = 1469598103934665603
    for v in ids:
        v = int(v) & 0xFFFFFFFF
        for b in range(4):
            h ^= (v >> (8 * b)) & 0xFF
            h = (h * 1099511628211) & _M64
    h ^= (len(ids) * 0x9E3779B97F4A7C15) & _M64
    return h | 1


class ArcLM:
    """an ARCLM1 file (read-only, memory-mapped); raises ValueError on a malformed file, as arc_lm_load refuses it"""

    def __init__(self, path):
        self.path = str(path)
        buf = np.memmap(self.path, dtype=np.uint8, mode="r")
        if len(buf) < 24:
            raise ValueError(f"lm {path}: too short for an ARCLM1 header")
        head = bytes(buf[:24])
        if head[:8] != MAGIC:
            raise ValueError(f"lm {path}: " + ("unsupported ARCLM version" if head[:5] == b"ARCLM" else "not an ARCLM file (bad magic)"))
        order, vocab, uns, cap = struct.unpack("<iifi", head[8:24])
        if not 1 <= order <= MAX_ORDER:
            raise ValueError(f"lm {path}: order must be 1-8")
        if not 1 <= vocab <= 1 << 24:
            raise ValueError(f"lm {path}: bad vocabulary size")
        if cap < 2 or cap > 1 << 30 or cap & (cap - 1):
            raise ValueError(f"lm {path}: table size must be a power of 2")
        if not math.isfinite(uns) or uns > 0:
            raise ValueError(f"lm {path}: bad unscored log-probability")
        sc = (vocab + 7) & ~7
        if len(buf) != 24 + sc + cap * 16:
            raise ValueError(f"lm {path}: file size does not match its header")
        self.order, self.vocab, self.unscored, self.cap = order, vocab, np.float32(uns), cap
        self.scored = np.asarray(buf[24:24 + vocab]).astype(bool)
        tab = np.frombuffer(buf, dtype=_ENTRY, count=cap, offset=24 + sc)
        used = tab["key"] != 0
        if not np.all(tab["key"][used] & 1) or not np.all(np.isfinite(tab["logp"][used])) or not np.all(np.isfinite(tab["bow"][used])):
            raise ValueError(f"lm {path}: bad table entry")
        if used.all():
            raise ValueError(f"lm {path}: table has no empty slot")
        self.entries = int(used.sum())
        self._key, self._logp, self._bow = tab["key"], tab["logp"], tab["bow"]
        self._mask = cap - 1

    def _find(self, ids):
        k = lm_key(ids)
        i = (k ^ (k >> 32)) & 0xFFFFFFFF & self._mask
        while True:
            e = int(self._key[i])
            if e == k:
                return i
            if e == 0:
                return -1
            i = (i + 1) & self._mask

    def is_scored(self, tok):
        return 0 <= tok < self.vocab and bool(self.scored[tok])

    def logp(self, hist, tok):
        """ln P(tok | hist) (hist: the scored tokens so far, oldest first), float32 as libarc computes it"""
        k = min(len(hist), self.order - 1)
        bo = np.float32(0.0)
        while k >= 0:
            g = list(hist[len(hist) - k:]) + [tok] if k else [tok]
            i = self._find(g)
            if i >= 0 and self._logp[i] <= 0:
                return np.float32(bo + self._logp[i])
            if k > 0:
                c = self._find(g[:-1])
                if c >= 0:
                    bo = np.float32(bo + self._bow[c])
            k -= 1
        return np.float32(bo + np.float32(2.0) * self.unscored)


def write_lm(path, order, vocab, grams, unscored=-5.0, scored=None):
    """write an ARCLM1 file. grams: {ids tuple: (logp, bow)} (logp > 0: a context only); scored: ids the model scores
    (default: every id in a unigram). Returns the table size."""
    if scored is None:
        scored = {g[0] for g in grams if len(g) == 1}
    cap = 2
    while cap < 2 * len(grams):
        cap *= 2
    tab = np.zeros(cap, dtype=_ENTRY)
    mask = cap - 1
    for g, (lp, bo) in grams.items():
        k = lm_key(g)
        i = (k ^ (k >> 32)) & 0xFFFFFFFF & mask
        while tab["key"][i]:
            if int(tab["key"][i]) == k:
                raise ValueError(f"key collision: {g}")
            i = (i + 1) & mask
        tab[i] = (k, lp, bo)
    sc = np.zeros((vocab + 7) & ~7, dtype=np.uint8)
    sc[sorted(t for t in scored if 0 <= t < vocab)] = 1
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<iifi", order, vocab, unscored, cap))
        f.write(sc.tobytes())
        f.write(tab.tobytes())
    return cap


def fuse(lm, weight, topk, row, gen, eos):
    """arc_lm_fuse on one row of scores (a 1-D float tensor; returns a new one). gen: the generated tokens, start
    token first."""
    if lm is None or not weight > 0:
        return row
    need = lm.order - 1
    hist = []
    for t in reversed(gen):
        if len(hist) >= need:
            break
        if lm.is_scored(t):
            hist.append(t)
    hist.reverse()
    row = torch.log_softmax(row.float(), -1)
    V = row.shape[-1]
    k = min(topk if topk > 0 else 16, 64, V)
    vals, idx = torch.topk(row, k)
    vals = vals.cpu().numpy().astype(np.float32)
    idx = [int(i) for i in idx.cpu().tolist()]
    w = np.float32(weight)
    best = vals[0]
    lo = np.float32(best - np.float32(MARGIN))
    lp = [None] * k
    top = None
    for j, t in enumerate(idx):
        if vals[j] >= lo and t != eos and lm.is_scored(t):
            lp[j] = lm.logp(hist, t)
            top = lp[j] if top is None or lp[j] > top else top
    keep = np.empty(k, np.float32)
    adj_best = np.float32(0.0)
    for j, t in enumerate(idx):
        a = np.float32(0.0)
        if lp[j] is not None:
            a = max(np.float32(lp[j] - top), np.float32(-CAP))
        elif vals[j] < lo:
            a = np.float32(-CAP)
        a = np.float32(a * w)
        if j == 0:
            adj_best = a
        keep[j] = np.float32(vals[j] + a)
    end = np.float32(row[eos].item() + adj_best) if 0 <= eos < V else None
    out = row + np.float32(-w * np.float32(CAP)).item()
    out[torch.tensor(idx, device=row.device)] = torch.tensor(keep, device=row.device)
    if end is not None:
        out[eos] = float(end)
    return out


class LMFusion(LogitsProcessor):
    """shallow fusion as a logits processor (after the suppressed ids, before the repeat guard, as libarc orders it).
    start: the index of the decoder start token in input_ids (after the context prompt)."""

    def __init__(self, lm, weight, topk, start, eos):
        self.lm, self.w, self.k, self.start, self.eos = lm, float(weight), int(topk), int(start), int(eos)

    def __call__(self, input_ids, scores):
        if not self.w > 0:
            return scores
        rows = [fuse(self.lm, self.w, self.k, scores[b], input_ids[b, self.start:].tolist(), self.eos) for b in range(scores.shape[0])]
        return torch.stack(rows, 0).to(scores.dtype)
