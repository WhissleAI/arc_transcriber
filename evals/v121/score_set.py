"""Score run_set.py rows: corpus WER of the plain and the written transcript under two normalizers, and number recall /
precision.

- whisper: Whisper's EnglishTextNormalizer with its British -> American spelling map (package `whisper-normalizer`, the
  copy of openai/whisper's normalizer the Open ASR Leaderboard uses). It converts number words to digits ("two thirty" -> "230").
- strict: lower case, punctuation removed except inside numbers, thousands commas dropped ("3,000" -> "3000"), the same fillers
  removed (uh, um, ...), and NO number-word conversion: "forty five" against "45" counts as errors, so the written form's
  effect is visible.
- numbers: every token with a digit in the reference, as a value ("$45" / "45%" / "21st" / "3,000" -> "45" / "45" / "21" /
  "3000"; "2:30" kept), matched as a multiset against the hypothesis's: recall = matched / reference numbers, precision =
  matched / hypothesis numbers.  `tw_digits` = share of utterances with a number whose every reference number appears verbatim
  (TypeWhisper's Digits rule).

    python -I score_set.py MANIFEST rows.jsonl [more rows.jsonl] [--json out.json] [--sample 30 --sample-out s.json]
"""
import argparse
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twscore  # noqa: E402

FILLERS = {"uh", "um", "uhm", "umm", "er", "erm", "ah", "hmm", "hm", "mhm", "mm", "mmm"}
TOKEN = re.compile(r"[$€£¥₹]?\d[\d,.:/]*\d(?:%|st|nd|rd|th|s)?|[$€£¥₹]?\d(?:%|st|nd|rd|th|s)?|[a-z]+(?:'[a-z]+)*")


def whisper_norm():
    from whisper_normalizer.english import EnglishTextNormalizer
    return EnglishTextNormalizer()


def strict_tokens(s):
    s = s.lower().replace("’", "'").replace("–", " ").replace("-", " ")
    out = []
    for t in TOKEN.findall(s):
        if t in FILLERS:
            continue
        if re.search(r"\d", t):
            t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t).rstrip(".,:/")
        out.append(t)
    return out


def num_values(s):
    vals = []
    for t in strict_tokens(s):
        if re.search(r"\d", t):
            v = re.sub(r"^[$€£¥₹]", "", t)
            v = re.sub(r"(%|st|nd|rd|th|s)$", "", v)
            vals.append(v)
    return vals


def edits(r, h):
    return twscore.edit_distance(r, h)


def corpus(rows, refs, field, fn):
    e = n = 0
    for r in rows:
        ref = fn(refs[r["id"]])
        hyp = fn(r[field] or "")
        e += edits(ref, hyp)
        n += len(ref)
    return e / max(n, 1)


def numbers(rows, refs, field):
    m = rn = hn = 0
    hits = tot = 0
    for r in rows:
        rv, hv = collections.Counter(num_values(refs[r["id"]])), collections.Counter(num_values(r[field] or ""))
        m += sum((rv & hv).values())
        rn += sum(rv.values())
        hn += sum(hv.values())
        d = twscore.digits_hit(refs[r["id"]], r[field] or "")
        if d is not None:
            tot += 1
            hits += bool(d)
    return dict(recall=m / max(rn, 1), precision=m / max(hn, 1), ref_numbers=rn, hyp_numbers=hn, matched=m,
                tw_digits=hits / max(tot, 1), tw_digits_n=f"{hits}/{tot}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--sample-out")
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    refs = {m["id"]: m["ref"] for m in man}
    wn = whisper_norm()
    W = lambda s: wn(s).split()  # noqa: E731
    res = {}
    for p in a.rows:
        rows = [json.loads(x) for x in open(p)]
        meta, rows = rows[0].get("meta", {}), [r for r in rows if "id" in r]
        rec = dict(meta=meta, n=len(rows), audio_min=sum(r["audio_s"] for r in rows) / 60,
                   rtf_x=sum(r["audio_s"] for r in rows) / (sum(r["ms"] for r in rows) / 1000))
        subsets = {"all": rows}
        if "has_digit" in man[0]:
            hd = {m["id"]: m["has_digit"] for m in man}
            subsets = {"all": rows, "with_digits": [r for r in rows if hd[r["id"]]],
                       "no_digits": [r for r in rows if not hd[r["id"]]]}
        for sname, rs in subsets.items():
            rec[sname] = {}
            for field in ("plain", "written"):
                if rs and rs[0].get(field) is None:
                    continue
                rec[sname][field] = dict(n=len(rs), wer_whisper=corpus(rs, refs, field, W),
                                         wer_strict=corpus(rs, refs, field, strict_tokens), **numbers(rs, refs, field))
            rec[sname]["changed_utterances"] = sum(r["plain"] != r.get("written") for r in rs if r.get("written") is not None)
        res[os.path.basename(p)] = rec
        print(os.path.basename(p), f"n={rec['n']} {rec['audio_min']:.1f} min, {rec['rtf_x']:.1f}x real time")
        for sname in subsets:
            for field, v in rec[sname].items():
                if isinstance(v, dict):
                    print(f"  {sname:12s} {field:8s} WER whisper {v['wer_whisper'] * 100:5.2f}  strict {v['wer_strict'] * 100:5.2f}  "
                          f"numbers R {v['recall'] * 100:5.1f} P {v['precision'] * 100:5.1f} ({v['matched']}/{v['ref_numbers']} ref,"
                          f" {v['hyp_numbers']} hyp)  TW-digits {v['tw_digits_n']}")
            print(f"  {sname:12s} changed by written form: {rec[sname]['changed_utterances']}")
        if a.sample and a.sample_out:
            ch = [r for r in rows if r.get("written") and r["written"] != r["plain"]]
            pick = random.Random(0).sample(ch, min(a.sample, len(ch)))
            json.dump([dict(id=r["id"], ref=refs[r["id"]], plain=r["plain"], written=r["written"]) for r in pick],
                      open(a.sample_out, "w"), indent=1)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
