"""Whispers, low-level speech, noise and channel conditions: what happens to the transcript, and which of Arc's
sound and channel events fire. See README.md.
  python evals/whisper_noise/run.py [--voices Samantha,Daniel,Aman] [--threads 4]
"""
import argparse
import collections
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402

USUAL = -26.0   # the speaker's usual voiced level (dBFS), as Steer learns it from earlier commands
# condition -> (voice override, level dBFS, transform, room)
CONDITIONS = {
    "clean": (None, USUAL, None, "quiet"),
    "low level (-42 dBFS)": (None, -42.0, None, "quiet"),
    "whisper voice": ("Whisper", USUAL, None, "quiet"),
    "whisper voice, low level": ("Whisper", -42.0, None, "quiet"),
    "noisy room (-38 dBFS noise)": (None, USUAL, None, "noisy"),
    "loud room (-28 dBFS noise)": (None, USUAL, None, "loud"),
    "band noise, 5 dB SNR": (None, USUAL, lambda x: C.band_noise(x, 5), "quiet"),
    "telephone (300-3400 Hz, mu-law)": (None, USUAL, C.phone, "quiet"),
    "clipped": (None, USUAL, C.clipped, "quiet"),
    "mumbled (low-passed, quiet)": (None, USUAL, C.mumbled, "quiet"),
    "ward at night (-58 dBFS)": (None, USUAL, None, "ward"),
    "whisper voice, ward, low level": ("Whisper", -42.0, None, "ward"),
}
EVENTS = ["meta.channel.telephone", "meta.channel.narrowband", "meta.clipping", "meta.degraded_audio", "meta.snr.low",
          "meta.snr.high", "scene.silence"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default="Samantha,Daniel,Aman")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--clips", default=os.path.join(C.WORK, "whisper_noise"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "run.json"))
    a = ap.parse_args()
    voices = a.voices.split(",")
    said = [r[1] for r in C.read_tsv(os.path.join(HERE, "data", "lines.tsv"))]
    arc = C.load_arc(a.threads)
    res = collections.defaultdict(lambda: dict(pairs=[], empty=0, ev=collections.defaultdict(list), intents=[]))
    for cname, (vo, lvl, fx, rname) in CONDITIONS.items():
        vs = [vo] if vo else voices
        for vi, v in enumerate(vs):
            for li, text in enumerate(said):
                w = C.load(C.render(text, v, os.path.join(a.clips, f"{v}_{C.slug(text)}.wav")))
                w = C.at_level(w, lvl)
                x = np.concatenate([C.silence(0.5), w, C.silence(0.8)])
                seed = 31 * li + vi
                if fx is not None:
                    x = fx(x)
                if rname == "noisy":
                    x = x + C.room(np.zeros(len(x), np.float32), -38, before=0, after=0, seed=seed)
                elif rname == "loud":
                    x = x + C.room(np.zeros(len(x), np.float32), -28, before=0, after=0, seed=seed)
                elif rname == "ward":
                    x = x + C.ward(len(x), seed)
                else:
                    x = x + C.room(np.zeros(len(x), np.float32), -60, before=0, after=0, seed=seed)
                o = arc.predict(x.astype(np.float32), domain="assistant", return_frames=True, events="all")
                heard = C.text_of(o["transcript"])
                r = res[cname]
                r["pairs"].append((text, heard))
                r["empty"] += not heard.strip()
                ae = o["frames"].get("audio_events") or {}
                labels, probs = ae.get("labels") or [], np.asarray(ae.get("probs") if ae.get("probs") is not None else [])
                if len(labels) and probs.size:
                    s0, s1 = int(0.5 * 10), int((0.5 + len(w) / C.SR) * 10)   # the speech's 100 ms frames
                    seg = probs[s0:max(s1, s0 + 1)]
                    for e in EVENTS:
                        if e in labels:
                            r["ev"][e].append(float(seg[:, labels.index(e)].mean()))
                it, ip = C.intent_of(o)
                r["intents"].append((text, it, ip))
        print(cname, "done", file=sys.stderr)
    table = {}
    print(f"\n{'condition':34} {'WER':>6} {'empty':>6}  " + " ".join(f"{e.split('.')[-1][:10]:>10}" for e in EVENTS))
    for cname, r in res.items():
        table[cname] = dict(wer_pct=100 * C.wer(r["pairs"]), empty=r["empty"], n=len(r["pairs"]),
                            events_mean={e: float(np.mean(v)) for e, v in r["ev"].items()},
                            examples=[p for p in r["pairs"] if C.edits(*p)[0] > 0][:6])
        t = table[cname]
        print(f"{cname:34} {t['wer_pct']:5.1f}% {t['empty']:3d}/{t['n']:<3d} " + " ".join(f"{t['events_mean'].get(e, float('nan')):10.2f}" for e in EVENTS))
    bed = [x for x in res["clean"]["intents"] if x[0] in ("Raise my head a little.", "I need the nurse.", "Show the pre-op scan, left side.")]
    print("\ntop intents on clinical lines (clean):", bed[:9])
    C.save(a.out, dict(env=C.env_info(arc, a.threads), voices=voices, usual_dbfs=USUAL, results=table, clinical_intents=bed))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
