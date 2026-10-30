"""Device-directed speech and wake names, in an operating-room scene and Steer's own setting. See README.md.
  python evals/device_directed/run.py [--voices Samantha,Daniel,Aman] [--threads 4]
Part A, wake names: is the device name heard at the start of a command, with and without name priming?
Part B, cross-talk: do commands (with or without the name) score higher on device-directed than staff talking to
each other in the same room?
"""
import argparse
import collections
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402
from arc_audio.api import context_dict  # noqa: E402


def lines(name):
    return [l[1] for l in C.read_tsv(os.path.join(HERE, "data", name))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default="Samantha,Daniel,Aman")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--clips", default=os.path.join(C.WORK, "device_directed"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "run.json"))
    a = ap.parse_args()
    voices = a.voices.split(",")
    arc = C.load_arc(a.threads)
    wake = [r for r in C.read_tsv(os.path.join(HERE, "data", "wake_names.tsv"))]   # name, setting
    cmds = {"or": lines("or_commands.tsv"), "steer": lines("steer_commands.tsv")}
    crosstalk = lines("or_crosstalk.tsv")
    rooms = {"or": (C.theatre, -24), "steer": (lambda n, s: C.room(np.zeros(n, np.float32), -55, before=0, after=0, seed=s), -26)}

    def clip(text, v, setting, k):
        w = C.load(C.render(text, v, os.path.join(a.clips, f"{v}_{C.slug(text)}.wav")))
        fn, lvl = rooms[setting]
        return C.in_room(w, fn, lvl, seed=k)

    # ---- A: wake names
    A = {}
    for name, setting in wake:
        for primed in (False, True):
            hits, misses = 0, []
            for vi, v in enumerate(voices):
                for ci, c in enumerate(cmds[setting]):
                    text = f"{name}, {c[0].lower() + c[1:]}"
                    x = clip(text, v, setting, 100 * vi + ci)
                    names = [name, "pre-op scan"] if setting == "or" else [name]
                    ctx = dict(ctx=context_dict(names=[dict(name=n, kind="product") for n in names], topic="assistant")) if primed else None
                    heard = C.text_of(arc.transcribe(x, names=names if primed else None, bias_names=primed, context=ctx))
                    first = C.words(heard)[:2]
                    ok = name.lower() in first
                    hits += ok
                    if not ok:
                        misses.append(heard)
            A[f"{name}|{'primed' if primed else 'unprimed'}"] = dict(heard=hits, n=len(voices) * len(cmds[setting]), misses=misses)
            print(f"{name:8} {'primed' if primed else 'unprimed':9} heard {hits}/{len(voices) * len(cmds[setting])}  misses: {misses[:4]}", file=sys.stderr)

    # ---- B: device-directed, commands vs cross-talk, in the operating room (name "Nova")
    groups = {"command with name": [f"Nova, {c[0].lower() + c[1:]}" for c in cmds["or"]],
              "command, no name": cmds["or"], "cross-talk (staff to staff)": crosstalk}
    B = collections.defaultdict(lambda: dict(tag=[], head=[], text=[]))
    for g, texts in groups.items():
        for vi, v in enumerate(voices):
            for ti, t in enumerate(texts):
                x = clip(t, v, "or", 1000 + 100 * vi + ti)
                o = arc.predict(x, domain="assistant", names=[dict(name="Nova", kind="product")], return_frames=False, events=None)
                tag, head = C.directed_of(o)
                B[g]["tag"].append(max(tag) if tag else 0.0)
                B[g]["head"].append(max(head) if head else 0.0)
                B[g]["text"].append(C.text_of(o["transcript"]))
    neg = B["cross-talk (staff to staff)"]
    Bt = {}
    for g in groups:
        Bt[g] = dict(n=len(B[g]["tag"]), tag_range=[min(B[g]["tag"]), max(B[g]["tag"])], tag_mean=float(np.mean(B[g]["tag"])),
                     head_range=[min(B[g]["head"]), max(B[g]["head"])], head_mean=float(np.mean(B[g]["head"])),
                     tag_ge_0_5=sum(x >= 0.5 for x in B[g]["tag"]),
                     auc_vs_crosstalk_tag=C.auc(B[g]["tag"], neg["tag"]) if g != "cross-talk (staff to staff)" else None,
                     auc_vs_crosstalk_head=C.auc(B[g]["head"], neg["head"]) if g != "cross-talk (staff to staff)" else None)

    print("\nA. wake name heard first (n per name = voices x commands)")
    for k, r in A.items():
        print(f"  {k:22} {r['heard']}/{r['n']}")
    print("\nB. device-directed in the operating room (utterance tag | transcript head)")
    for g, r in Bt.items():
        print(f"  {g:28} tag {r['tag_range'][0]:.2f}-{r['tag_range'][1]:.2f} (mean {r['tag_mean']:.2f}, >=0.5 {r['tag_ge_0_5']}/{r['n']})"
              f"  head {r['head_range'][0]:.3f}-{r['head_range'][1]:.3f} (mean {r['head_mean']:.3f})"
              + (f"  AUC vs cross-talk tag {r['auc_vs_crosstalk_tag']:.2f} head {r['auc_vs_crosstalk_head']:.2f}" if r["auc_vs_crosstalk_tag"] is not None else ""))
    C.save(a.out, dict(env=C.env_info(arc, a.threads), voices=voices, wake=A, directed=Bt, directed_raw=dict(B)))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
