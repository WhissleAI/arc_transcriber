"""Context memory: does Arc's text context (names prompt, decoder bias, previous text) help, and does carrying the
trunk's hidden state across utterances help device-directed and act? See README.md.
  python evals/context_memory/run.py [--part text|state|both] [--voices Samantha,Daniel,Flo,Aman] [--threads 4]
"""
import argparse
import collections
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
from items import APPS, BOOK, BOOKD, COMMANDS, DICT, OTHER, PAIRS, PEOPLE, WORDS  # noqa: E402
from arc_audio.api import context_dict  # noqa: E402

KIND = {**{n: "person" for n in PEOPLE}, **{n: "term" for n in WORDS}, **{n: "product" for n in APPS},
        **{n: "person" for n in BOOK}, "Pemberley": "place"}
ALL = PEOPLE + WORDS + APPS + BOOK
SCORE_NAMES = ALL + ["Priyanka", "Oluwaseun", "Tomasz", "Siddharth", "Elizabeth", "Darcy"]


def items():
    """(id, category, text, previous text, the names relevant to it)"""
    out = [(f"cmd{i:02d}", "cmd", t, None, APPS) for i, t in enumerate(COMMANDS)]
    out += [(f"dict{i:02d}", "dict", t, None, PEOPLE + WORDS) for i, t in enumerate(DICT)]
    out += [(f"book{i:02d}", "book", t, None, BOOK) for i, t in enumerate(BOOKD)]
    for i, (a, b) in enumerate(PAIRS):
        out += [(f"pairA{i:02d}", "dict", a, None, PEOPLE + WORDS + BOOK), (f"pairB{i:02d}", "pairB", b, a, PEOPLE + WORDS + BOOK)]
    return out


def ctx(names=None, prev=None):
    return dict(ctx=context_dict(names=[dict(name=n, kind=KIND.get(n, "term")) for n in names or []], topic="assistant",
                                 previous_text=prev))


CONDITIONS = {   # name -> (bias list?, prompt names?, previous text?)
    "none": (False, None, False),
    "bias (relevant names)": ("rel", None, False),
    "names prompt + bias (relevant)": ("rel", "rel", False),
    "all names (31) prompt + bias": ("all", "all", False),
    "previous text only": (False, None, True),
    "all names + bias + previous text": ("all", "all", True),
}


def text_part(arc, voices, clips):
    rows = collections.defaultdict(list)
    for name, (bias, prompt, prev) in CONDITIONS.items():
        for v in voices:
            for iid, cat, text, ptext, rel in items():
                wav = C.load(C.render(text, v, os.path.join(clips, f"{v}_{iid}.wav")))
                wav = np.concatenate([C.silence(0.3), wav, C.silence(0.5)])
                lst = {"rel": rel, "all": ALL}.get(bias) if bias else None
                pn = {"rel": rel, "all": ALL}.get(prompt) if prompt else None
                c = ctx(pn, ptext if prev else None) if (pn or (prev and ptext)) else None
                t0 = time.perf_counter()
                heard = C.text_of(arc.transcribe(wav, names=lst, bias_names=bool(lst), context=c))
                rows[name].append(dict(id=iid, voice=v, cat=cat, ref=text, heard=heard, ms=1000 * (time.perf_counter() - t0)))
        print(name, "done", file=sys.stderr)
    table = {}
    for name, rs in rows.items():
        t = {}
        for cat in ("cmd", "dict", "book", "pairB", "all"):
            sel = [r for r in rs if cat == "all" or r["cat"] == cat]
            got = tot = 0
            for r in sel:
                ref = [n for n in SCORE_NAMES if C.contains(r["ref"], n)]
                tot += len(ref)
                got += sum(C.contains(r["heard"], n) for n in ref)
            t[cat] = dict(wer_pct=100 * C.wer([(r["ref"], r["heard"]) for r in sel]), names_pct=100 * got / max(1, tot),
                          ms=float(np.mean([r["ms"] for r in sel])), n=len(sel))
        table[name] = t
    print(f"\n{'condition':34} {'all WER':>8} {'names':>7} {'cmd WER':>8} {'contin. WER':>12} {'ms':>5}")
    for name, t in table.items():
        print(f"{name:34} {t['all']['wer_pct']:7.1f}% {t['all']['names_pct']:6.1f}% {t['cmd']['wer_pct']:7.1f}% "
              f"{t['pairB']['wer_pct']:11.1f}% {t['all']['ms']:5.0f}")
    return dict(table=table, rows=rows)


def state_part(arc, voices, clips, gap_s=2.0, seed=20261029):
    """Fresh: predict() on each utterance with gap_s of room before it. Carried: one predict() over the whole session
    (the trunk's state runs on across utterances, as in a stream), utterances matched back by time."""
    seq = [("cmd", t) for t in COMMANDS] + [("dict", t) for t in DICT] + [("nd", t) for t in OTHER]
    res = dict(fresh=[], carried=[])
    for v in voices:
        order = seq[:]
        random.Random(seed + hash(v) % 1000).shuffle(order)
        parts, spans, t = [], [], 0.0
        for k, (cat, text) in enumerate(order):
            w = C.load(C.render(text, v, os.path.join(clips, f"{v}_s_{C.slug(text)}.wav")))
            w = w / max(1e-4, float(np.max(np.abs(w)))) * 0.35
            g = C.silence(gap_s)
            parts += [g, w]
            spans.append((cat, text, t + gap_s, t + gap_s + len(w) / C.SR))
            t += gap_s + len(w) / C.SR
            # fresh: the same utterance with the same room before it, from a zero state
            x = C.room(np.concatenate([g, w]), -60, level=0.35, before=0, after=0.5, seed=k)
            o = arc.predict(x, domain="assistant", return_frames=False, events=None)
            tag, head = C.directed_of(o)
            act, _ = C.act_of(o)
            res["fresh"].append(dict(voice=v, cat=cat, text=text, dd_tag=max(tag) if tag else None,
                                     dd_head=max(head) if head else None, act=act))
        x = C.room(np.concatenate(parts + [C.silence(1.0)]), -60, level=0.35, before=0, after=0, seed=1)
        o = arc.predict(x, domain="assistant", return_frames=False, events=None)
        for cat, text, a, b in spans:
            us = [u for u in o.get("utterances") or [] if u["end"] > a - 0.2 and u["start"] < b + 0.2]
            ts = [u for u in o.get("transcript") or [] if u["end"] > a - 0.2 and u["start"] < b + 0.2]
            tag = [u["tags"].get("discourse.assistant.device_directed") for u in us if "discourse.assistant.device_directed" in u.get("tags", {})]
            head = [(u.get("device_directed") or {}).get("p") for u in ts if u.get("device_directed")]
            acts = [u["groups"]["discourse.act"]["label"].replace("discourse.act.", "") for u in us if "discourse.act" in u.get("groups", {})]
            res["carried"].append(dict(voice=v, cat=cat, text=text, dd_tag=max(tag) if tag else None,
                                       dd_head=max(head) if head else None, act=acts[0] if acts else None))
        print("session", v, "done", file=sys.stderr)
    table = {}
    for mode, rs in res.items():
        row = {}
        for key in ("dd_tag", "dd_head"):
            pos = [r[key] for r in rs if r["cat"] == "cmd" and r[key] is not None]
            neg = [r[key] for r in rs if r["cat"] == "nd" and r[key] is not None]
            row[key] = dict(auc=C.auc(pos, neg), cmd_mean=float(np.mean(pos)) if pos else None,
                            other_mean=float(np.mean(neg)) if neg else None, n_cmd=len(pos), n_other=len(neg))
        cmds = [r for r in rs if r["cat"] == "cmd"]
        ok = sum(r["act"] == ("question" if r["text"].endswith("?") else "command_request") for r in cmds)
        row["act_right"] = [ok, len(cmds)]
        table[mode] = row
    print(f"\n{'trunk state':10} {'AUC (utt tag)':>14} {'AUC (head)':>11} {'act right':>10}")
    for mode, r in table.items():
        print(f"{mode:10} {r['dd_tag']['auc']:14.3f} {r['dd_head']['auc']:11.3f} {r['act_right'][0]:4d}/{r['act_right'][1]}")
    return dict(table=table, rows=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="both", choices=["text", "state", "both"])
    ap.add_argument("--voices", default=",".join(C.VOICES))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--clips", default=os.path.join(C.WORK, "context_memory"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "run.json"))
    a = ap.parse_args()
    arc = C.load_arc(a.threads)
    voices = a.voices.split(",")
    out = dict(env=C.env_info(arc, a.threads), voices=voices)
    if a.part in ("text", "both"):
        out["text"] = text_part(arc, voices, a.clips)
    if a.part in ("state", "both"):
        out["state"] = state_part(arc, voices, a.clips)
    C.save(a.out, out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
