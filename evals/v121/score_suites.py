"""Score run_suites.py rows with the TypeWhisper port (twscore.py) and compare with TypeWhisper's published rows.

    python -I score_suites.py --tw TW_CHECKOUT --rows mini_120=rows.jsonl ... [--json out.json]
A rows name ending in _mini / _full is matched with TypeWhisper's whissle-arc_mini / _transcriber results.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twscore  # noqa: E402


def load_rows(p):
    rows = [json.loads(x) for x in open(p)]
    return rows[0]["meta"], rows[1:]


def summarize(rows, field, tests):
    per, dg, rtf = {}, [], []
    for r in rows:
        if field not in r:
            return None
        t = tests[(r["suite"], r["id"])]
        per.setdefault(r["suite"], []).append(twscore.wer_normalized(t["groundTruth"], r[field]))
        if r["suite"] == "number-formatting-en":
            dg.append(bool(twscore.digits_hit(t["groundTruth"], r[field])))
        rtf.append(r["ms"] / 1000 / r["audio_s"])
    out = dict(per_suite={s: sum(v) / len(v) for s, v in per.items()}, **twscore.pools(per),
               digits=sum(dg) / len(dg), digits_n=f"{sum(dg)}/{len(dg)}", mean_rtf=sum(rtf) / len(rtf))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tw", required=True)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    tests = {}
    for s in twscore.EN_SUITES:
        for t in json.load(open(os.path.join(a.tw, "bench/tests", s + ".json")))["tests"]:
            tests[(s, t["id"])] = t
    pub = {}
    for k, name in (("mini", "whissle-arc_mini.json"), ("full", "whissle-arc_transcriber.json")):
        d = json.load(open(os.path.join(a.tw, "visualizer/public/data/results", name)))
        pub[k] = {(s, r["testId"]): r for s, rs in d["resultsBySuite"].items() for r in rs}
    res = {}
    for spec in a.rows:
        name, path = spec.split("=", 1)
        meta, rows = load_rows(path)
        model = "full" if "full" in name else "mini"
        P = pub[model]
        theirs = [dict(suite=r["suite"], id=r["id"], ms=P[(r["suite"], r["id"])]["durationMs"], audio_s=r["audio_s"],
                       plain=P[(r["suite"], r["id"])]["transcription"]) for r in rows]
        same = sum(r["plain"] == P[(r["suite"], r["id"])]["transcription"] for r in rows)
        res[name] = dict(meta=meta, n=len(rows), plain=summarize(rows, "plain", tests),
                         written=summarize(rows, "written", tests),
                         typewhisper_published=summarize(theirs, "plain", tests),
                         identical_to_typewhisper=f"{same}/{len(rows)}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    suites = twscore.EN_SUITES
    cols = []
    for name, r in res.items():
        cols.append((name + " TW-published", r["typewhisper_published"]))
        cols.append((name + " plain", r["plain"]))
        if r["written"]:
            cols.append((name + " written", r["written"]))
    print("| suite | " + " | ".join(c for c, _ in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for s in suites:
        print(f"| {s} | " + " | ".join(f"{c['per_suite'][s] * 100:.1f}" for _, c in cols) + " |")
    for key, lab in (("wer_en_core_no_acc", "WER EN w/o acc (core)"), ("wer_en_all", "WER all 10 suites"),
                     ("wer_en_no_acc", "WER 9 suites w/o acc")):
        print(f"| **{lab}** | " + " | ".join(f"{c[key] * 100:.1f}" for _, c in cols) + " |")
    print("| Digits | " + " | ".join(f"{c['digits'] * 100:.0f} ({c['digits_n']})" for _, c in cols) + " |")
    print("| mean RTF (x real time) | " + " | ".join(f"{1 / c['mean_rtf']:.1f}" for _, c in cols) + " |")
    for name, r in res.items():
        print(name, "identical plain text to TypeWhisper's rows:", r["identical_to_typewhisper"], "| threads",
              r["meta"]["torch_threads"], r["meta"]["device"], r["meta"]["arc_audio"])


if __name__ == "__main__":
    main()
