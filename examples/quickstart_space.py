#!/usr/bin/env python3
"""Arc Transcriber Space (backbone "nft2"): offline and streaming quick start.

    pip install "arc-transcriber[space] @ git+https://github.com/WhissleAI/arc_transcriber"
    python examples/quickstart_space.py                          # model + sample clips from WhissleAI/arc-transcriber-space
    python examples/quickstart_space.py my.wav other.wav         # your own clips (16 kHz mono; other rates need scipy)
    python examples/quickstart_space.py --model /local/arc-transcriber-space [--samples DIR]

The names list is what your app shows (on screen, in the current lesson). Always give a kind; see docs/NAMES_LISTS.md.
"""
import argparse, glob, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from arc_audio import Arc, load_audio  # noqa: E402

REPO = "WhissleAI/arc-transcriber-space"
NAMES = [
    {"name": "Surveyor 6", "kind": "spacecraft"},
    {"name": "Sinus Medii", "kind": "lunar_feature"},
    {"name": "Apollo 12", "kind": "mission"},
    {"name": "Ocean of Storms", "kind": "lunar_feature", "aliases": ["Oceanus Procellarum"]},
    {"name": "Tycho", "kind": "lunar_feature"},
    {"name": "Orion", "kind": "spacecraft"},
    {"name": "Copernicus", "kind": "lunar_feature"},
    {"name": "Artemis II", "kind": "mission", "aliases": ["Artemis 2"]},
]


def sample_files(model, samples):
    if samples:
        return sorted(glob.glob(os.path.join(samples, "*.wav")))
    if os.path.isdir(model):
        return sorted(glob.glob(os.path.join(model, "examples", "samples", "*.wav")))
    from huggingface_hub import snapshot_download
    d = snapshot_download(model, allow_patterns=["examples/samples/*"])
    return sorted(glob.glob(os.path.join(d, "examples", "samples", "*.wav")))


def show(out):
    for u in out.get("transcript") or []:
        print("  text  :", u.get("text"))
        tagged = [(w.get("text"), w.get("tag")) for w in u.get("words") or [] if w.get("tag")]
        if tagged:
            print("  tags  :", tagged)
    for n in out.get("names") or []:
        print(f"  name  : heard={n.get('heard')!r} tag={n.get('tag')} -> entity={n.get('entity')!r} state={n.get('state')} p={n.get('p')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="16 kHz mono clips (default: the sample clips of the model repo)")
    ap.add_argument("--model", default=REPO, help="Hugging Face repo or local folder (default: %(default)s)")
    ap.add_argument("--samples", help="folder of .wav clips to run when no files are given")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    arc = Arc.from_pretrained(a.model, backbone="nft2", device="cpu", threads=a.threads)
    files = a.files or sample_files(a.model, a.samples)
    if not files:
        sys.exit("no clips: pass .wav files or --samples DIR")

    # offline: one call per file
    for f in files:
        print(os.path.basename(f))
        show(arc.predict(load_audio(f), domain="assistant", transcript="nft2", names=NAMES, return_frames=False))

    # streaming: 100 ms chunks; the names list can be replaced mid-stream with s.set_context(names=...)
    print("streaming:", os.path.basename(files[0]))
    s = arc.stream(domain="assistant", transcript="nft2", names=NAMES)
    wav = load_audio(files[0])
    for i in range(0, len(wav), 1600):
        r = s.feed(wav[i:i + 1600])
        if r and (r.get("transcript") or r.get("names")):
            show(r)
    show(s.flush())


if __name__ == "__main__":
    main()
