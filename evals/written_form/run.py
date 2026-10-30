"""Written form: does `transcript_written` score better than the spoken-form `transcript` against written references? See README.md.
  python evals/written_form/run.py [--voices Samantha,Daniel,Flo] [--libri DIR] [--device cpu] [--threads 4]

--libri DIR: a folder of LibriSpeech-style clips (<id>.wav, 16 kHz) with refs.json {id: reference}, e.g. the 73 validation clips of
hf-internal-testing/librispeech_asr_dummy, to check that the written form does not hurt plain speech.
"""
import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402

NUM = re.compile(r"[$€£]?\d[\d,.:]*\d(?:%|st|nd|rd|th)?|[$€£]?\d(?:%|st|nd|rd|th)?")


def normalizer():
    """Whisper's EnglishTextNormalizer (from transformers, no spelling map); a basic lower-case / no-punctuation fallback"""
    try:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
        n = EnglishTextNormalizer({})
        return "whisper EnglishTextNormalizer", n
    except Exception:
        return "basic", lambda s: " ".join(C.words(s))


def edits(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p = d[:]
        d[0] = i
        for j in range(1, len(h) + 1):
            d[j] = min(p[j] + 1, d[j - 1] + 1, p[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)]


def numbers_of(text):
    """written numbers in a text: '$45', '2:30', '12.5%', '3rd', '2,400' (trailing sentence punctuation removed)"""
    return [m.group().rstrip(".,") for m in NUM.finditer(text)]


def digit_hits(ref, hyp):
    want = numbers_of(ref)
    have = collections.Counter(numbers_of(hyp))
    hit = 0
    for x in want:
        if have[x] > 0:
            have[x] -= 1
            hit += 1
    return hit, len(want)


def bare(s):
    return " ".join(re.sub(r"[.,?!;]+(\s|$)", r"\1", s.lower()).split())


def score(rows, norm):
    out = {}
    for field in ("transcript", "written", "written_digits"):
        e = n = hit = want = exact = 0
        for r in rows:
            ref, hyp = norm(r["ref"]).split(), norm(r[field]).split()
            e += edits(ref, hyp)
            n += len(ref)
            h, w = digit_hits(r["ref"], r[field])
            hit += h
            want += w
            exact += bare(r["ref"]) == bare(r[field])
        out[field] = dict(wer=round(100 * e / max(1, n), 2), digits=round(100 * hit / max(1, want), 1), digits_n=want,
                          exact=round(100 * exact / max(1, len(rows)), 1), n=len(rows))
    return out


def run_clip(arc, path, domain="unknown"):
    from arc_audio.written import written_transcript
    out = arc.predict(path, domain=domain, return_frames=False, events=None)
    t = " ".join(u["text"] for u in out["transcript"])
    w = " ".join(u["text"] for u in out["transcript_written"] or [])
    wd = " ".join(u["text"] for u in written_transcript(out["transcript"], out["names"], domain, spell_small=False))
    return dict(transcript=t, written=w, written_digits=wd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default="Samantha,Daniel,Flo")
    ap.add_argument("--libri", help="folder with LibriSpeech-style <id>.wav and refs.json")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--clips", default=os.path.join(C.WORK, "written_form"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "run.json"))
    a = ap.parse_args()
    from arc_audio import Arc
    arc = Arc.from_pretrained("WhissleAI/arc-v1.1", backbone="mini", device=a.device, threads=a.threads)
    nname, norm = normalizer()
    sents = [l.rstrip("\n").split("\t") for l in open(os.path.join(HERE, "data", "sentences.txt"), encoding="utf-8")
             if l.strip() and not l.startswith("#")]
    rows = []
    for v in a.voices.split(","):
        for kind, ref in sents:
            r = run_clip(arc, C.render(ref, v, os.path.join(a.clips, f"{v}_{C.slug(ref)}.wav")))
            rows.append(dict(r, voice=v, kind=kind, ref=ref))
    res = dict(env=C.env_info(arc, a.threads), normalizer=nname, numbers=score(rows, norm),
               by_kind={k: score([r for r in rows if r["kind"] == k], norm) for k in sorted({r["kind"] for r in rows})})
    if a.libri:
        refs = json.load(open(os.path.join(a.libri, "refs.json")))
        lrows = [dict(run_clip(arc, os.path.join(a.libri, f"{k}.wav")), ref=ref, id=k) for k, ref in sorted(refs.items())]
        res["librispeech"] = score(lrows, norm)
        res["librispeech_rows"] = lrows
    res["rows"] = rows
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"normalizer: {nname}")
    for name, s in [("numbers", res["numbers"])] + ([("librispeech", res["librispeech"])] if a.libri else []):
        for f, x in s.items():
            print(f"{name:12s} {f:15s} WER {x['wer']:6.2f}%  digits {x['digits']:5.1f}% of {x['digits_n']}  exact {x['exact']:5.1f}%  (n={x['n']})")
    for k, s in res["by_kind"].items():
        print(f"  {k:9s} " + "  ".join(f"{f}: WER {x['wer']:.1f} digits {x['digits']:.0f}%" for f, x in s.items()))


if __name__ == "__main__":
    main()
