"""Voice-command bench: spoken commands in rooms -> Arc transcript -> command parser -> right command?

Port of Steer's SteerVoiceBench (press-and-hold mode). See README.md.
  python evals/commands/run.py [--quick] [--voices Samantha,Daniel,Flo,Aman] [--files ...] [--threads 4] [--out results/run.json]
"""
import argparse
import collections
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
from parser import parse  # noqa: E402

# Steer's rooms: name -> (noise dBFS, speech peak level)
ROOMS = {"quiet": (-60, 0.35), "noisy": (-38, 0.35), "loud-room": (-28, 0.35), "soft-voice": (-55, 0.05)}
# As the app runs Arc: no prompt, the messaging-app names as decoder bias (the wake name "Steer" is added in the app too).
BIAS = ["WhatsApp", "Slack", "Messages", "Telegram", "Mail", "Safari"]
FILES = ["commands_spoken.tsv", "commands_loose.tsv", "commands_loose_heldout.tsv", "commands_loose_heldout2.tsv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="every third phrase, two voices")
    ap.add_argument("--voices", default=",".join(C.VOICES))
    ap.add_argument("--files", default=",".join(FILES))
    ap.add_argument("--no-bias", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--clips", default=os.path.join(C.WORK, "commands"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "run.json"))
    a = ap.parse_args()

    phrases = []
    for f in a.files.split(","):
        phrases += [(k, s, f) for k, s in C.read_tsv(os.path.join(HERE, "data", f))]
    voices = a.voices.split(",")
    if a.quick:
        phrases, voices = phrases[::3], voices[:2]
    oracle = sum(parse(s) == k for k, s, _ in phrases)
    print(f"{len(phrases)} phrases x {len(voices)} voices x {len(ROOMS)} rooms; parser right on the true words: {oracle}/{len(phrases)}")

    arc = C.load_arc(a.threads)
    tally = collections.defaultdict(lambda: dict(n=0, ok=0, scored=0, e=0, refw=0, secs=[]))
    fails = []
    t_start = time.time()
    for i, (kind, said, f) in enumerate(phrases):
        truth_ok = parse(said) == kind
        for v in voices:
            speech = C.load(C.render(said, v, os.path.join(a.clips, f"{v}_{C.slug(said)}.wav")))
            for ri, (rname, (db, lvl)) in enumerate(ROOMS.items()):
                x = C.room(speech, db, level=lvl, seed=hash((said, v, rname)) & 0xFFFF)
                x = x[: int(0.6 * C.SR) + len(speech) + int(0.3 * C.SR)]   # press-and-hold: 0.3 s tail after the speech
                t0 = time.perf_counter()
                heard = C.text_of(arc.transcribe(x, names=None if a.no_bias else BIAS, bias_names=not a.no_bias))
                dt = time.perf_counter() - t0
                got = parse(heard)
                for key in (rname, "all"):
                    t = tally[key]
                    t["n"] += 1
                    e, n = C.edits(said, heard)
                    t["e"] += e
                    t["refw"] += n
                    t["secs"].append(dt)
                    if truth_ok:
                        t["scored"] += 1
                        t["ok"] += got == kind
                if truth_ok and got != kind and len(fails) < 2000:
                    fails.append(dict(room=rname, voice=v, expected=kind, got=got, said=said, heard=heard))
        if i % 20 == 19:
            print(f"{i + 1}/{len(phrases)} phrases, {time.time() - t_start:.0f} s", file=sys.stderr)

    rows = {}
    print(f"\n{'room':11} {'right command':>22} {'WER':>7} {'p50 ms':>7} {'p95 ms':>7}")
    for key in list(ROOMS) + ["all"]:
        t = tally[key]
        rows[key] = dict(right=t["ok"], scored=t["scored"], right_pct=100 * t["ok"] / max(1, t["scored"]),
                         wer_pct=100 * t["e"] / max(1, t["refw"]), clips=t["n"],
                         p50_ms=1000 * C.quantile(t["secs"], 0.5), p95_ms=1000 * C.quantile(t["secs"], 0.95))
        r = rows[key]
        print(f"{key:11} {r['right_pct']:6.1f}% ({r['right']:4d}/{r['scored']:4d})   {r['wer_pct']:6.1f}% {r['p50_ms']:7.0f} {r['p95_ms']:7.0f}")
    C.save(a.out, dict(env=C.env_info(arc, a.threads), phrases=len(phrases), voices=voices, rooms=ROOMS, bias=not a.no_bias,
                       parser_oracle=[oracle, len(phrases)], results=rows, failures=fails))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
