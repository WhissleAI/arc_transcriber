"""TypeWhisper's 10 English suites through Arc, as their worker runs it (bench/scripts/arc-worker.py in TypeWhisper PR #16):
`Arc.from_pretrained("WhissleAI/arc-v1.1", backbone, revision="v1.2.0", device, threads=None)`, then `arc.predict(wav)` with
the package defaults, timed around predict() only. Writes one JSON line per clip with the plain transcript (their formula)
and, on 1.2.1, the written one.

    python -I run_suites.py --tw TW_CHECKOUT --backbone mini --out rows.jsonl [--device cpu] [--speed speed.json]
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twscore  # noqa: E402


def join(utts):
    return " ".join(" ".join(u.get("text", "").strip() for u in utts or []).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tw", required=True)
    ap.add_argument("--backbone", default="mini")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed", help="also run their speed protocol (30 clips, 3 warm-up clips, 3 passes, median) to this JSON")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import torch
    import arc_audio
    from arc_audio import Arc, load_audio
    t0 = time.time()
    arc = Arc.from_pretrained("WhissleAI/arc-v1.1", backbone=a.backbone, revision="v1.2.0", device=a.device, threads=a.threads)
    load_s = time.time() - t0
    meta = dict(arc_audio=arc_audio.__version__, backbone=a.backbone, device=a.device, torch=torch.__version__,
                torch_threads=torch.get_num_threads(), load_s=round(load_s, 1), loadavg=os.getloadavg())
    print(json.dumps(meta), file=sys.stderr)
    bench = os.path.join(a.tw, "bench")
    with open(a.out, "w") as f:
        f.write(json.dumps(dict(meta=meta)) + "\n")
        n = 0
        for suite in twscore.EN_SUITES:
            for t in json.load(open(os.path.join(bench, "tests", suite + ".json")))["tests"]:
                wav = load_audio(os.path.join(bench, "audio", t["audioFile"]))
                s = time.perf_counter()
                out = arc.predict(wav)
                ms = (time.perf_counter() - s) * 1000
                row = dict(suite=suite, id=t["id"], audio_s=len(wav) / 16000, ms=ms, plain=join(out.get("transcript")))
                if "transcript_written" in out:
                    row["written"] = join(out["transcript_written"])
                row["utts"] = [[{x: w.get(x) for x in ("text", "tag", "tag_begin") if w.get(x) is not None}
                                for w in u.get("words") or []] for u in out.get("transcript") or []]   # for rewrite.py
                row["numbers"] = [{k: v for k, v in x.items() if k in ("heard", "kind", "value", "date", "iso")}
                                  for x in out.get("numbers") or []]
                f.write(json.dumps(row) + "\n")
                f.flush()
                n += 1
                if a.limit and n >= a.limit:
                    return
    if a.speed:
        ss = json.load(open(os.path.join(bench, "speed", "speed-set.json")))
        clips = [(c, load_audio(os.path.join(bench, "audio", c["audioFile"]))) for c in ss["clips"]]
        for c, wav in clips[:3]:
            arc.predict(wav)                                          # warm-up
        times = {c["testId"]: [] for c, _ in clips}
        for _ in range(3):
            for c, wav in clips:
                s = time.perf_counter()
                arc.predict(wav)
                times[c["testId"]].append(time.perf_counter() - s)
        med = {k: statistics.median(v) for k, v in times.items()}
        audio = sum(len(w) / 16000 for _, w in clips)
        res = dict(meta, speed_factor=audio / sum(med.values()), audio_s=audio,
                   buckets={b: sum(len(w) / 16000 for c, w in clips if c["bucket"] == b) /
                            sum(med[c["testId"]] for c, _ in clips if c["bucket"] == b) for b in ("short", "medium", "long")},
                   loadavg_after=os.getloadavg())
        json.dump(res, open(a.speed, "w"), indent=1)
        print(json.dumps(res), file=sys.stderr)


if __name__ == "__main__":
    main()
