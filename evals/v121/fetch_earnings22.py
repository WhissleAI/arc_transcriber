"""Pick a numbers subset of Earnings-22 (Rev.com, CC BY-SA 4.0; real earnings calls, many non-native speakers) from the public
Hugging Face copy `distil-whisper/earnings22` (config `chunked`, written-form references with digits), and write the chosen
segments as 16 kHz WAV files plus a manifest.

Selection (deterministic, seed 0): read every shard's text column (range reads, no audio); keep segments of 2-20 s with no
"<inaudible>" / "<crosstalk>" style markup; pick one 100-segment row group per call file for up to `--groups` different
calls; from each, take every segment whose reference has a digit, plus up to `--controls` digit-free segments. Only those
row groups' audio is downloaded.

    python -I fetch_earnings22.py OUT_DIR [--groups 24] [--controls 4]
"""
import argparse
import io
import json
import os
import random
import re

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfFileSystem

REPO = "datasets/distil-whisper/earnings22"


def ok_text(t):
    return "<" not in t and len(t.split()) >= 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--groups", type=int, default=24)
    ap.add_argument("--controls", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "wav"), exist_ok=True)
    fs = HfFileSystem()
    shards = sorted(fs.glob(f"{REPO}/chunked/test-*.parquet"))
    cand = []                                   # (shard, rg, file_id, n_digit_segments)
    for sh in shards:
        p = pq.ParquetFile(fs.open(sh, block_size=1 << 20))
        for rg in range(p.metadata.num_row_groups):
            rows = p.read_row_group(rg, columns=["file_id", "transcription", "start_ts", "end_ts"]).to_pylist()
            good = [r for r in rows if ok_text(r["transcription"]) and 2 <= r["end_ts"] - r["start_ts"] <= 20]
            nd = sum(bool(re.search(r"\d", r["transcription"])) for r in good)
            fid = max(set(r["file_id"] for r in rows), key=[r["file_id"] for r in rows].count)
            cand.append((sh, rg, fid, nd))
        print(sh.rsplit("/", 1)[-1], "read", flush=True)
    rnd = random.Random(0)
    by_file = {}
    for c in cand:
        if c[3] >= 8:
            by_file.setdefault(c[2], []).append(c)
    files = sorted(by_file)
    rnd.shuffle(files)
    picks = [rnd.choice(by_file[f]) for f in files[:a.groups]]
    manifest = []
    for sh, rg, fid, _ in sorted(picks):
        p = pq.ParquetFile(fs.open(sh, block_size=1 << 22))
        rows = p.read_row_group(rg).to_pylist()
        good = [r for r in rows if ok_text(r["transcription"]) and 2 <= r["end_ts"] - r["start_ts"] <= 20]
        dig = [r for r in good if re.search(r"\d", r["transcription"])]
        ctl = [r for r in good if not re.search(r"\d", r["transcription"])]
        rnd.shuffle(ctl)
        for r in dig + ctl[:a.controls]:
            wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
            wav = wav.mean(1)
            if sr != 16000:                         # a few calls are stored at 24 / 44.1 / 48 kHz
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(sr, 16000)
                wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)
            sid = f"{r['file_id']}_{r['segment_id']}".replace("/", "_")      # segment ids repeat across calls
            sf.write(os.path.join(a.out, "wav", sid + ".wav"), wav, 16000, subtype="PCM_16")
            manifest.append(dict(id=sid, file_id=r["file_id"], shard=sh.rsplit("/", 1)[-1], row_group=rg,
                                 has_digit=bool(re.search(r"\d", r["transcription"])), ref=r["transcription"],
                                 seconds=round(len(wav) / 16000, 2)))
        print(sh.rsplit("/", 1)[-1], rg, fid, len(manifest), flush=True)
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=0)
    print(len(manifest), "segments,", sum(m["has_digit"] for m in manifest), "with digits,",
          len(set(m["file_id"] for m in manifest)), "calls")


if __name__ == "__main__":
    main()
