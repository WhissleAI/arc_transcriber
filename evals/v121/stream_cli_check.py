"""Written output through every path on real recordings: predict()["transcript_written"], stream() finals (`transcript_written` of
each feed / flush) and flush()["transcript_written_all"], and `arc-transcribe --written` in txt / srt / vtt / json. The recordings
are built from an Earnings-22 manifest (fetch_earnings22.py): consecutive segments of one call joined into ~60 s.

    python -I stream_cli_check.py E22_MANIFEST --work DIR --out stream_cli.json [--n 5] [--backbone mini]
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twscore  # noqa: E402


def text(utts):
    return " ".join(" ".join(u.get("text", "") for u in utts or []).split())


def cue_text(s, fmt):
    """the caption text of an srt / vtt file, or the lines of a txt file without their time / speaker prefix"""
    out = []
    for line in s.splitlines():
        line = line.strip()
        if not line or line == "WEBVTT" or re.fullmatch(r"\d+", line) or "-->" in line:
            continue
        if fmt == "txt":
            line = re.sub(r"^\[\s*[\d.]+-\s*[\d.]+\]\s*(S\d+:\s*)?", "", line)
        else:
            line = re.sub(r"^(<v [^>]*>|S\d+:\s*)", "", line)
        out.append(line)
    return " ".join(" ".join(out).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--backbone", default="mini")
    ap.add_argument("--cli", default=os.path.join(os.path.dirname(sys.executable), "arc-transcribe"))
    a = ap.parse_args()
    from arc_audio import Arc, load_audio
    man = json.load(open(a.manifest))
    base = os.path.dirname(os.path.abspath(a.manifest))
    by = collections.defaultdict(list)
    for m in man:
        by[m["file_id"]].append(m)
    os.makedirs(a.work, exist_ok=True)
    recs = []
    for fid in sorted(by)[: a.n]:
        parts, tot = [], 0.0
        for m in sorted(by[fid], key=lambda m: m["id"]):
            w = load_audio(os.path.join(base, m["wav"]))
            parts += [w, np.zeros(8000, np.float32)]
            tot += len(w) / 16000 + 0.5
            if tot >= 60:
                break
        p = os.path.join(a.work, f"rec_{fid}.wav")
        sf.write(p, np.concatenate(parts), 16000, subtype="PCM_16")
        recs.append(p)
    arc = Arc.from_pretrained("WhissleAI/arc-v1.1", backbone=a.backbone, revision="v1.2.0", device="cpu")
    rows = []
    for p in recs:
        wav = load_audio(p)
        out = arc.predict(wav)
        r = dict(file=os.path.basename(p), seconds=round(len(wav) / 16000, 1), predict_plain=text(out["transcript"]),
                 predict_written=text(out["transcript_written"]))
        tw = arc.transcribe(wav, written=True)
        r["transcribe_text_written"] = " ".join(" ".join(u.get("text_written", "") for u in tw).split())
        s = arc.stream()
        fin, finw = [], []
        for i in range(0, len(wav), 1600):
            x = s.feed(wav[i:i + 1600])
            fin += x.get("transcript") or []
            finw += x.get("transcript_written") or []
        f = s.flush()
        fin += f.get("transcript") or []
        finw += f.get("transcript_written") or []
        r.update(stream_plain=text(fin), stream_written=text(finw), stream_written_all=text(f.get("transcript_written_all")),
                 stream_utts=len(fin), stream_written_utts=len(finw))
        for fmt in ("txt", "srt", "vtt", "json"):
            o = os.path.join(a.work, os.path.basename(p)[:-4] + "." + fmt)
            cmd = [a.cli, p, "--written", "--format", fmt, "-o", o, "--backbone", a.backbone]
            cp = subprocess.run(cmd, capture_output=True, text=True)
            if cp.returncode:
                r["cli_" + fmt] = "ERROR " + cp.stderr[-300:]
                continue
            s_ = open(o).read()
            if fmt == "json":
                j = json.loads(s_)
                r["cli_json_written"] = text(j.get("transcript_written"))
                r["cli_json_line_written"] = " ".join(" ".join(l.get("written", "") for l in j.get("lines") or []).split())
            else:
                r["cli_" + fmt] = cue_text(s_, fmt)
        ref = r["predict_written"].split()
        r["diff_words_vs_predict_written"] = {k: twscore.edit_distance(ref, (r.get(k) or "").split()) for k in
                                              ("transcribe_text_written", "stream_written", "stream_written_all", "cli_txt",
                                               "cli_srt", "cli_vtt", "cli_json_written", "cli_json_line_written")}
        r["stream_plain_vs_predict_plain"] = twscore.edit_distance(r["predict_plain"].split(), r["stream_plain"].split())
        rows.append(r)
        print(r["file"], r["seconds"], r["diff_words_vs_predict_written"], "stream plain diff", r["stream_plain_vs_predict_plain"],
              flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
