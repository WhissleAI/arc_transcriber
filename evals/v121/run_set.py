"""Run Arc's predict() over a manifest (JSON list of {id, wav, ref}; `wav` relative to the manifest) and write one JSON line per
utterance with the plain transcript, the written one and the number records.

    python -I run_set.py MANIFEST --backbone mini|full --out rows.jsonl [--device cpu] [--threads N] [--limit N]
"""
import argparse
import json
import os
import time


def join(utts):
    return " ".join(" ".join(u.get("text", "").strip() for u in utts or []).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--backbone", default="mini")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torch
    import arc_audio
    from arc_audio import Arc, load_audio
    arc = Arc.from_pretrained("WhissleAI/arc-v1.1", backbone=a.backbone, revision="v1.2.0", device=a.device, threads=a.threads)
    man = json.load(open(a.manifest))
    base = os.path.dirname(os.path.abspath(a.manifest))
    done = set()
    if os.path.exists(a.out):                         # resume
        done = {json.loads(x).get("id") for x in open(a.out)}
    with open(a.out, "a") as f:
        if not done:
            f.write(json.dumps(dict(meta=dict(arc_audio=arc_audio.__version__, backbone=a.backbone, device=a.device,
                                              torch_threads=torch.get_num_threads()))) + "\n")
        for k, m in enumerate(man):
            if a.limit and k >= a.limit:
                break
            if m["id"] in done:
                continue
            wav = load_audio(os.path.join(base, m["wav"]))
            s = time.perf_counter()
            out = arc.predict(wav)
            ms = (time.perf_counter() - s) * 1000
            row = dict(id=m["id"], audio_s=len(wav) / 16000, ms=ms, plain=join(out.get("transcript")),
                       written=join(out.get("transcript_written")) if "transcript_written" in out else None,
                       numbers=[{x: v for x, v in n.items() if x in ("heard", "kind", "value", "value2", "unit", "date", "iso")}
                                for n in out.get("numbers") or []],
                       # the tagged words, so the written form can be re-rendered with another arc_audio.written (rewrite.py)
                       utts=[[{x: w.get(x) for x in ("text", "tag", "tag_begin") if w.get(x) is not None} for w in u.get("words") or []]
                             for u in out.get("transcript") or []])
            f.write(json.dumps(row) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
