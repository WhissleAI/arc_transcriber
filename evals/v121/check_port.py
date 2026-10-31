"""Check twscore.py against TypeWhisper's published per-clip results: recompute werNormalized and Digits from their own
transcriptions and references. Usage: python check_port.py TW_CHECKOUT [results/*.json names ...]"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twscore  # noqa: E402

tw = sys.argv[1]
names = sys.argv[2:] or ["whissle-arc_mini.json", "whissle-arc_transcriber.json"]
tests = {}
for s in twscore.EN_SUITES:
    for t in json.load(open(os.path.join(tw, "bench/tests", s + ".json")))["tests"]:
        tests[(s, t["id"])] = t
for name in names:
    d = json.load(open(os.path.join(tw, "visualizer/public/data/results", name)))
    n = bad = 0
    per = {}
    dg = []
    for s, rows in d["resultsBySuite"].items():
        if s not in twscore.EN_SUITES:
            continue
        for r in rows:
            t = tests[(s, r["testId"])]
            w = twscore.wer_normalized(t["groundTruth"], r["transcription"])
            n += 1
            if abs(w - r["werNormalized"]) > 1e-9:
                bad += 1
                print("MISMATCH", name, r["testId"], w, r["werNormalized"])
            per.setdefault(s, []).append(r["werNormalized"])
            if s == "number-formatting-en":
                dg.append(twscore.digits_hit(t["groundTruth"], r["transcription"]))
    p = twscore.pools(per)
    print(f"{name}: {n} clips, {bad} mismatches; published-row pools: all {p['wer_en_all']:.3f}, w/o acc "
          f"{p['wer_en_no_acc']:.3f}, core w/o acc {p['wer_en_core_no_acc']:.3f}; Digits {sum(map(bool, dg))}/{len(dg)}")
