"""CPU vs MPS (Apple GPU), end to end: predict() and stream() (100 ms chunks fed as fast as the stream takes them) on clips of three
lengths built from a LibriSpeech manifest (consecutive utterances of one chapter joined with 0.3 s of silence). Records wall
time and the plain / written transcripts, and how far each device's text is from CPU's (word edit distance).

    python -I device_check.py LS_MANIFEST --backbone mini|full --out dev_<backbone>.json [--devices cpu mps] [--lengths 15 60 180]
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twscore  # noqa: E402


def clips(manifest, lengths):
    from arc_audio import load_audio
    man = json.load(open(manifest))
    base = os.path.dirname(os.path.abspath(manifest))
    by = collections.defaultdict(list)
    for m in man:
        by["-".join(m["id"].split("-")[:2])].append(m)
    chapters = sorted(by.values(), key=len, reverse=True)
    out = []
    for k, L in enumerate(lengths):
        parts, refs, tot = [], [], 0.0
        for m in chapters[k % len(chapters)]:
            w = load_audio(os.path.join(base, m["wav"]))
            parts += [w, np.zeros(4800, np.float32)]
            refs.append(m["ref"])
            tot += len(w) / 16000 + 0.3
            if tot >= L:
                break
        wav = np.concatenate(parts)[: int(L * 16000)] if tot >= L else np.concatenate(parts)
        out.append(dict(target_s=L, seconds=round(len(wav) / 16000, 1), wav=wav))
    return out


def text(utts):
    return " ".join(" ".join(u.get("text", "") for u in utts or []).split())


def run_stream(arc, wav):
    s = arc.stream()
    tr, trw = [], []
    t0 = time.perf_counter()
    for i in range(0, len(wav), 1600):
        r = s.feed(wav[i:i + 1600])
        tr += r.get("transcript") or []
        trw += r.get("transcript_written") or []
    f = s.flush()
    dt = time.perf_counter() - t0
    tr += f.get("transcript") or []
    trw += f.get("transcript_written") or []
    return dict(s=dt, plain=text(tr), written=text(trw), written_all=text(f.get("transcript_written_all")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--backbone", default="mini")
    ap.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    ap.add_argument("--lengths", nargs="+", type=float, default=[15, 60, 180])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from arc_audio import Arc
    C = clips(a.manifest, a.lengths)
    res = dict(backbone=a.backbone, torch=torch.__version__, threads=torch.get_num_threads(), runs=[])
    for dev in a.devices:
        arc = Arc.from_pretrained("WhissleAI/arc-v1.1", backbone=a.backbone, revision="v1.2.0", device=dev, threads=a.threads)
        arc.predict(C[0]["wav"][:16000 * 5])                       # warm-up
        for c in C:
            t0 = time.perf_counter()
            out = arc.predict(c["wav"])
            if dev == "mps":
                torch.mps.synchronize()
            p = dict(s=time.perf_counter() - t0, plain=text(out["transcript"]), written=text(out["transcript_written"]),
                     numbers=len(out.get("numbers") or []))
            st = None if a.no_stream else run_stream(arc, c["wav"])
            res["runs"].append(dict(device=dev, seconds=c["seconds"], predict=p, stream=st))
            print(dev, c["seconds"], f"predict {p['s']:.2f}s", f"stream {st['s']:.2f}s" if st else "", flush=True)
        del arc
    cpu = {r["seconds"]: r for r in res["runs"] if r["device"] == "cpu"}
    for r in res["runs"]:
        ref = cpu.get(r["seconds"])
        if not ref:
            continue
        for mode in ("predict", "stream"):
            if r[mode] and ref[mode]:
                a_, b_ = ref[mode]["plain"].split(), r[mode]["plain"].split()
                r[mode]["word_diff_vs_cpu"] = twscore.edit_distance(a_, b_)
                r[mode]["identical_to_cpu"] = ref[mode]["plain"] == r[mode]["plain"]
                r[mode]["written_identical_to_cpu"] = ref[mode]["written"] == r[mode]["written"]
        if r["stream"]:
            r["stream"]["words_vs_predict_same_device"] = twscore.edit_distance(r["predict"]["plain"].split(),
                                                                               r["stream"]["plain"].split())
    json.dump(res, open(a.out, "w"), indent=1)
    for r in res["runs"]:
        print(r["device"], r["seconds"], {m: {k: v for k, v in r[m].items() if k not in ("plain", "written", "written_all")}
                                          for m in ("predict", "stream") if r[m]})


if __name__ == "__main__":
    main()
