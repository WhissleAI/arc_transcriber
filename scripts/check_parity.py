"""Golden reference outputs for Arc: make them, or check a candidate (an exported / ported model) against them.

    python scripts/check_parity.py --run --backbone mini -o mine.json [--model DIR_OR_REPO] [--device cpu]   # run arc_audio, write a candidate
    python scripts/check_parity.py --backbone mini --candidate mine.json                                     # compare with tests/golden/mini.json
    python scripts/check_parity.py --make --backbone mini                                                    # (maintainers) rewrite the golden file

A port that does not use arc_audio writes the same JSON keys itself (see tests/golden/*.json): "output" (the predict() records it
produces) and any of the "stages" it can expose. Checks: transcript text, inline tags, linked names and number values must match
exactly; numeric stages must be within the tolerances below.
"""
import argparse, hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP = os.path.join(ROOT, "examples", "example_synth.wav")
NAMES = os.path.join(ROOT, "examples", "example_names.json")
GOLDEN = os.path.join(ROOT, "tests", "golden")
TOL = dict(feat_mean=2e-3, feat_std=2e-3, feat_head=5e-2, prob_mean=1e-2, prob_head=5e-2)
SETTINGS = dict(domain="nba", events="default", kb_snap=True, return_frames=True)


def summary(x, n=8):
    import numpy as np
    x = np.asarray(x, np.float32)
    return dict(shape=list(x.shape), mean=float(x.mean()) if x.size else 0.0, std=float(x.std()) if x.size else 0.0,
                head=[round(float(v), 5) for v in x.reshape(-1)[:n]], sha256_fp16=hashlib.sha256(x.astype(np.float16).tobytes()).hexdigest())


def run(backbone, model=None, device="cpu", threads=4):
    import numpy as np, torch
    torch.set_num_threads(threads)
    from arc_audio import Arc, load_audio
    arc = Arc.from_pretrained(model or "WhissleAI/arc-v1.1", backbone=backbone, device=device, threads=threads)
    wav = load_audio(CLIP)
    names = json.load(open(NAMES))
    x, xe, xt = arc._feats(wav, tap=True)
    out = arc.predict(wav, names=names, **SETTINGS)
    fr = out.pop("frames", {}) or {}
    out.pop("timing", None)
    stages = dict(features=summary(x), event_features=summary(xt) if xt is not None else None,
                  frames={q: summary(v) for q, v in (fr.get("probs") or {}).items()},
                  audio_events=summary(fr["audio_events"]["probs"]) if fr.get("audio_events") else None,
                  transcript=[dict(text=u["text"], tags=[(t["tag"], t["text"]) for t in u.get("tags", [])]) for u in out["transcript"]])
    keep = {k: out[k] for k in ("transcript", "names", "numbers", "audio_events", "speakers") if k in out}
    return dict(backbone=backbone, input=os.path.basename(CLIP), settings=dict(SETTINGS, names=os.path.basename(NAMES)),
                device=device, output=keep, stages=stages)


def _key_names(o):
    return sorted((n.get("heard"), n.get("entity"), n.get("state")) for n in o.get("names") or [])


def _key_numbers(o):
    return sorted((n.get("heard"), n.get("kind"), n.get("value"), n.get("value2")) for n in o.get("numbers") or [])


def compare(gold, cand):
    errs, notes = [], []
    g, c = gold["output"], cand.get("output") or {}
    gt = [u["text"] for u in g["transcript"]]; ct = [u["text"] for u in c.get("transcript") or []]
    if gt != ct:
        errs.append(f"transcript differs:\n  gold {gt}\n  cand {ct}")
    gs, cs = gold["stages"].get("transcript"), (cand.get("stages") or {}).get("transcript")
    if cs is not None and [tuple(map(tuple, u["tags"])) for u in gs] != [tuple(map(tuple, u["tags"])) for u in cs]:
        errs.append("inline tags differ")
    if _key_names(g) != _key_names(c):
        errs.append(f"names differ:\n  gold {_key_names(g)}\n  cand {_key_names(c)}")
    if _key_numbers(g) != _key_numbers(c):
        errs.append(f"numbers differ:\n  gold {_key_numbers(g)}\n  cand {_key_numbers(c)}")
    cst = cand.get("stages") or {}
    for k in ("features", "event_features", "audio_events"):
        a, b = gold["stages"].get(k), cst.get(k)
        if a is None or b is None:
            notes.append(f"stage {k}: not compared")
            continue
        if a["shape"] != b["shape"]:
            errs.append(f"{k}: shape {b['shape']} != {a['shape']}"); continue
        pre = "feat" if "features" in k else "prob"
        if abs(a["mean"] - b["mean"]) > TOL[f"{pre}_mean"] * max(1.0, abs(a["mean"])):
            errs.append(f"{k}: mean {b['mean']:.5f} vs {a['mean']:.5f}")
        if max(abs(u - v) for u, v in zip(a["head"], b["head"])) > TOL[f"{pre}_head"]:
            errs.append(f"{k}: first values {b['head']} vs {a['head']}")
        if a["sha256_fp16"] == b["sha256_fp16"]:
            notes.append(f"stage {k}: bit-identical (float16)")
    for q, a in (gold["stages"].get("frames") or {}).items():
        b = (cst.get("frames") or {}).get(q)
        if b is None:
            continue
        if a["shape"] != b["shape"] or abs(a["mean"] - b["mean"]) > TOL["prob_mean"] or \
                max(abs(u - v) for u, v in zip(a["head"], b["head"])) > TOL["prob_head"]:
            errs.append(f"frames[{q}] differ (mean {b['mean']:.4f} vs {a['mean']:.4f})")
    return errs, notes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="mini", choices=["mini", "full"])
    ap.add_argument("--model", default=None); ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--run", action="store_true"); ap.add_argument("--make", action="store_true")
    ap.add_argument("--candidate"); ap.add_argument("-o", "--output")
    a = ap.parse_args(argv)
    gpath = os.path.join(GOLDEN, f"{a.backbone}.json")
    if a.make or a.run:
        res = run(a.backbone, a.model, a.device, a.threads)
        dst = gpath if a.make else (a.output or "candidate.json")
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        json.dump(res, open(dst, "w"), indent=1, default=float)
        print("wrote", dst)
        if not a.candidate and a.run and os.path.exists(gpath) and not a.make:
            a.candidate = dst
    if a.candidate:
        errs, notes = compare(json.load(open(gpath)), json.load(open(a.candidate)))
        for n in notes:
            print("note:", n)
        for e in errs:
            print("FAIL:", e)
        print("PARITY OK" if not errs else f"PARITY FAILED ({len(errs)} checks)")
        return 0 if not errs else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
