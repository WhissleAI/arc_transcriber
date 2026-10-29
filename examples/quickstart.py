"""Arc v1.1 quick start: python examples/quickstart.py [--backbone edge|parakeet] [--device cpu|cuda]

Runs the bundled synthetic clip (fictional names) through Arc offline and live, and prints the transcript with inline tags,
linked names, numbers with values, sound / channel events and device-directed scores."""
import argparse, os
from arc_audio import Arc, load_audio

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--backbone", default="edge", choices=["edge", "parakeet"]); ap.add_argument("--device", default="cpu")
a = ap.parse_args()

arc = Arc.from_pretrained("WhissleAI/arc-v1.1", backbone=a.backbone, device=a.device, threads=4)
names = ["Marcus Delaney", {"name": "Theo Brandt", "aliases": ["Brandt"]}, "Owen Castellano",
         {"name": "Harbor City Hawks", "kind": "team", "aliases": ["Hawks"]}, {"name": "Riverside Comets", "kind": "team"}]
wav = load_audio(os.path.join(HERE, "example_synth.wav"))

# 1. whole file
out = arc.predict(wav, domain="nba", names=names)
for u in out["transcript"]:
    print(f"[{u['start']:6.2f}] {u['text']}   tags={[(t['tag'], t['text']) for t in u.get('tags', [])]}  directed={u.get('device_directed')}")
for n in out["names"]:
    print("name  ", repr(n["heard"]), "->", n["entity"], n["state"], n.get("via", ""))
for n in out["numbers"]:
    print("number", repr(n["heard"]), n.get("kind"), n.get("value"), n.get("value2"))
for e in out.get("audio_events", []):
    print("event ", e)

# 2. live, 100 ms chunks, with decoder context and context memory
s = arc.stream(domain="nba", names=names, auto_prev=True, background=False)
s.set_context(kb=["Marcus Delaney", "Theo Brandt", "Harbor City Hawks"], kb_type=["person", "person", "team"])
for i in range(0, len(wav), 1600):
    new = s.feed(wav[i:i + 1600])
    for u in new["transcript"]:
        print("live", u["start"], u["text"])
final = s.flush()
print("speakers:", final["speakers"])
