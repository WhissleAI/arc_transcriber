"""Arc Transcriber Space (backbone "nft2") smoke test, CPU, ONNX encoder. Skipped when the model is not available locally:

    pip install -e ".[space,test]"
    ARC_SPACE_MODEL=/path/to/arc-transcriber-space pytest -q tests/test_space.py     # a local copy of the model repo (or its nft2/)
    pytest -q tests/test_space.py              # uses WhissleAI/arc-transcriber-space when it is already in the Hugging Face cache
"""
import glob, os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = "WhissleAI/arc-transcriber-space"
NAMES = [{"name": "Tycho", "kind": "lunar_feature"}, {"name": "Orion", "kind": "spacecraft"},
         {"name": "Apollo 12", "kind": "mission", "aliases": ["Apollo twelve"]}, {"name": "Christina Koch", "kind": "crew"}]


def test_names_normalization():
    from arc_audio.nft2 import Pieces, num_words
    assert num_words(12) == "twelve"
    assert Pieces.norm("Apollo 12") == "apollo twelve"
    assert Pieces.norm("S-IVB") == "s ivb"
    assert Pieces.norm("Chang’e 3") == "chang'e three"


def test_tidy_name_spoken_numbers():
    from arc_audio.tidy import tidy_name
    def heard(entity, ws):
        r = dict(entity=entity, heard=" ".join(ws), start=0.0, end=float(len(ws)), words=[dict(w=w, start=float(i), end=i + 0.5) for i, w in enumerate(ws)])
        return tidy_name(r)["heard"]
    assert heard("Apollo 12", ["Apollo", "twelve"]) == "Apollo twelve"
    assert heard("Surveyor 6", ["the", "Surveyor", "six"]) == "Surveyor six"
    assert heard("Artemis II", ["Artemis", "two"]) == "Artemis two"
    assert heard("Luna 24", ["Luna", "twenty-four"]) == "Luna twenty-four"
    assert heard("Tycho", ["Tycho", "crater"]) == "Tycho"


def test_space_kinds():
    from arc_audio.api import KB_KIND
    for k in ("lunar_feature", "site", "celestial", "spacecraft", "vehicle", "mission", "crew"):
        assert k in KB_KIND


def _model():
    p = os.environ.get("ARC_SPACE_MODEL")
    if p:
        return p
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(REPO, allow_patterns=["nft2/*", "examples/samples/*"], local_files_only=True)
    except Exception:
        return None


@pytest.fixture(scope="module")
def space():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    m = _model()
    if not m:
        pytest.skip("Arc Transcriber Space model not available: set ARC_SPACE_MODEL or download WhissleAI/arc-transcriber-space")
    from arc_audio import Arc
    return m, Arc.from_pretrained(m, backbone="nft2", device="cpu", threads=4)


def _clip(m):
    fs = sorted(glob.glob(os.path.join(m, "examples", "samples", "*.wav")))
    from arc_audio import load_audio
    return load_audio(fs[0] if fs else str(ROOT / "examples" / "example_synth.wav"))


def test_space_offline_and_stream(space):
    m, arc = space
    assert arc.backbone == "nft2"
    wav = _clip(m)
    out = arc.predict(wav, domain="assistant", transcript="nft2", names=NAMES, return_frames=False)
    words = [w for u in out["transcript"] for w in u.get("words") or []]
    assert words and all(isinstance(w["text"], str) and w["end"] >= w["start"] for w in words)
    assert all(u.get("source") in (None, "nft2") for u in out["transcript"])
    for n in out.get("names") or []:
        assert n["state"] in ("linked", "abstain", "not_in_list")
        assert n["entity"] is None or n["entity"] in {x["name"] for x in NAMES}

    s = arc.stream(domain="assistant", transcript="nft2", names=NAMES)
    s.set_context(names=NAMES[:2])
    s.set_context(names=NAMES)
    for i in range(0, len(wav), 1600):
        s.feed(wav[i:i + 1600])
    fin = s.flush()
    sw = [w["text"] for u in fin.get("transcript") or [] for w in u.get("words") or []]
    assert sw
    key = lambda t: "".join(c for c in t.lower() if c.isalnum())
    assert key(sw[-1]) == key(words[-1]["text"])          # the stream keeps the last word of the clip (tail of the final utterance)
    s.close()


def test_space_silence(space):
    _, arc = space
    out = arc.predict(np.zeros(16000 * 2, np.float32), domain="assistant", transcript="nft2", names=NAMES, return_frames=False)
    assert not (out.get("names") or [])
