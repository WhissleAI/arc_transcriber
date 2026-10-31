"""Date records: render data/dates.tsv with macOS voices (written to files; nothing is played), run predict(),
and check each sentence's NUMBER_DATE record: heard, date {year, month, day}, iso, plus the written transcript.

    python -I run_dates.py --backbone mini|full --work DIR --out dates_<backbone>.json [--voices Samantha Daniel]
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def render(text, voice, path):
    if os.path.exists(path):
        return
    aiff = path[:-4] + ".aiff"
    subprocess.run(["say", "-v", voice, "-o", aiff, text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff, path], check=True)
    os.remove(aiff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="mini")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voices", nargs="+", default=["Samantha", "Daniel"])
    a = ap.parse_args()
    from arc_audio import Arc, load_audio
    arc = Arc.from_pretrained("WhissleAI/arc-v1.1", backbone=a.backbone, revision="v1.2.0", device="cpu")
    os.makedirs(a.work, exist_ok=True)
    rows = []
    for k, line in enumerate(open(os.path.join(HERE, "data", "dates.tsv"))):
        if line.startswith("#") or not line.strip():
            continue
        text, iso = (line.rstrip("\n").split("\t") + [""])[:2]
        for v in a.voices:
            p = os.path.join(a.work, f"date{k:02d}_{v}.wav")
            render(text, v, p)
            out = arc.predict(load_audio(p))
            recs = [n for n in out.get("numbers") or [] if n.get("kind") == "date"]
            other = [dict(heard=n.get("heard"), kind=n.get("kind"), value=n.get("value")) for n in out.get("numbers") or []
                     if n.get("kind") != "date"]
            got = [n.get("iso") for n in recs]
            rows.append(dict(text=text, voice=v, expected_iso=iso or None,
                             plain=" ".join(u["text"] for u in out["transcript"]),
                             written=" ".join(u["text"] for u in out["transcript_written"]),
                             date_records=[{x: n.get(x) for x in ("heard", "value", "date", "iso")} for n in recs],
                             other_numbers=other,
                             iso_right=(iso in got) if iso else (not recs)))
            r = rows[-1]
            print(("OK  " if r["iso_right"] else "BAD ") + f"{v:8s} {text} -> {r['date_records']} | {r['written']}", flush=True)
    n_ok = sum(r["iso_right"] for r in rows)
    print(f"{a.backbone}: {n_ok}/{len(rows)} sentences with the expected date record")
    json.dump(dict(backbone=a.backbone, right=n_ok, n=len(rows), rows=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
