"""CPU smoke test: downloads WhissleAI/arc-v1.1 (pinned revision, Arc Mini) from Hugging Face on first use; skipped when offline.

    pytest -q                                          # or: python -m pytest tests
    ARC_MODEL=/path/to/arc-v1.1 pytest -q              # a local copy of the model repository instead of the download
    ARC_TEST_PARAKEET=1 pytest -q                      # also run Arc Transcriber (full, Parakeet encoder) (downloads ~2.5 GB)
"""
import json, os, socket, subprocess, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "examples" / "example_synth.wav"
NAMES = ROOT / "examples" / "example_names.txt"
MODEL = os.environ.get("ARC_MODEL")


def online():
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
        return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def arc():
    if not MODEL and not online():
        pytest.skip("offline and no ARC_MODEL: cannot download WhissleAI/arc-v1.1")
    from arc_transcriber import load
    return load(MODEL) if MODEL else load()


def test_text_rules():
    from arc_audio import textlayer as TL
    v, _ = TL.text_value(None, ["102", "to", "98."])
    assert (v["v"], v["v2"]) == (102, 98)
    v, _ = TL.text_value("duration", ["25", "minutes"])
    assert v["v"] == 1500
    v, _ = TL.text_value("time_of_day", ["6", "p.m."])
    assert v["v"] == 18 * 3600


def test_event_window_matches_offline():
    import torch
    from arc_audio.events import EventHead, CTX
    torch.manual_seed(0)
    h = EventHead(d_in=16, k=5, hid=8).eval()
    x = torch.randn(23, 16)
    with torch.no_grad():
        ref = h(x[None])[0]
        parts, done = [], 0
        for n in (4, 9, 15, 23):
            hi = n if n == 23 else n - CTX
            if hi > done:
                base = max(0, done - CTX)
                parts.append(h.window(x[base:n], done - base, hi - base, n - base if n == 23 else None))
                done = hi
    assert torch.allclose(torch.cat(parts), ref, atol=1e-5)


def test_int4_unpack_roundtrip(tmp_path):
    import json as J
    import torch
    from safetensors.torch import save_file
    from arc_audio.weights import load_q4
    codes = torch.randint(0, 16, (6, 32), dtype=torch.uint8)
    s = torch.rand(6) * 0.01 + 1e-3
    z = torch.randint(0, 16, (6,), dtype=torch.uint8)
    pk = lambda c: (c.reshape(-1)[0::2] | (c.reshape(-1)[1::2] << 4)).contiguous()
    lay = {"w": dict(shape=[3, 64], axis=-1, rc=[3, 64, 0], dtype="float32")}
    save_file({"w.q": pk(codes), "w.s": s, "w.z": pk(z), "b": torch.ones(3)}, str(tmp_path / "m.q4.safetensors"),
              metadata={"format": "pt", "int4_layout": J.dumps(lay), "group_size": "32"})
    sd = load_q4(str(tmp_path / "m.q4.safetensors"))
    ref = ((codes.float() - z.float()[:, None]) * s[:, None]).reshape(3, 64)
    assert torch.equal(sd["w"], ref) and torch.equal(sd["b"], torch.ones(3))


def test_context_prompt():
    from arc_audio.api import context_dict
    c = context_dict(names=["Jane Doe", {"name": "Harbor City Hawks", "kind": "team"}], terms=["Arc"], topic="meeting", previous_text="hello")
    assert c["kb"] == ["Jane Doe", "Harbor City Hawks", "Arc"] and c["kb_type"] == ["term", "team", "term"]
    assert c["dom"] == "meeting" and c["prev"] == "hello"


def test_output_formats():
    from arc_transcriber.output import render
    out = dict(transcript=[dict(text="Theo", start=0.0, end=0.3), dict(text="Brandt", start=0.35, end=0.7), dict(text="scored", start=0.8, end=1.1),
                           dict(text="26", start=1.2, end=1.4), dict(text="points.", start=1.45, end=1.8)],
               names=[dict(start=0.0, end=0.7, heard="Theo Brandt", entity="Theo Brandt", state="linked")],
               numbers=[dict(start=1.2, end=1.8, heard="26 points.", kind="stat", value=26, value2=None)], diarization=[])
    txt = render(out, "txt")
    assert "[Theo Brandt](name: Theo Brandt)" in txt and "[26 points.](stat: 26)" in txt
    assert render(out, "srt").startswith("1\n00:00:00,000 --> 00:00:01,800")
    assert render(out, "vtt").startswith("WEBVTT")
    assert json.loads(render(out, "json"))["lines"][0]["tagged"]


def test_predict_builtin(arc):
    from arc_audio import load_audio
    from arc_transcriber.cli import load_names
    out = arc.predict(load_audio(str(CLIP)), domain="nba", names=load_names(str(NAMES)), transcript="builtin", return_frames=False)
    assert len(" ".join(u["text"] for u in out["transcript"]).split()) > 20
    assert any(n["state"] == "linked" for n in out["names"])
    pair = [n for n in out["numbers"] if "102" in n["heard"]]
    assert pair and (pair[0]["value"], pair[0]["value2"]) == (102, 98)


def test_events_and_context(arc):
    from arc_audio import load_audio
    wav = load_audio(str(CLIP))
    out = arc.predict(wav, domain="nba", events="all", return_frames=True)
    assert "audio_events" in out and out["frames"]["audio_events"]["probs"].shape[1] == 50
    assert all("device_directed" in u for u in out["transcript"])
    s = arc.stream(domain="nba", background=False, auto_prev=True)
    s.set_context(kb=["Marcus Delaney", "Theo Brandt"], kb_type=["person", "person"])
    got = [s.feed(wav[i:i + 1600]) for i in range(0, len(wav), 1600)] + [s.flush()]
    assert sum(len(g["transcript"]) for g in got) > 0


@pytest.mark.skipif(not os.environ.get("ARC_TEST_PARAKEET"), reason="ARC_TEST_PARAKEET=1 runs the full backbone")
def test_full_backbone():
    from arc_audio import Arc, load_audio
    from arc_transcriber.cli import load_names
    a = Arc.from_pretrained(MODEL or "WhissleAI/arc-v1.1", backbone="full", revision=None if MODEL else "v1.1.1")
    out = a.predict(load_audio(str(CLIP)), domain="nba", names=load_names(str(NAMES)))
    assert len(" ".join(u["text"] for u in out["transcript"]).split()) > 20
    assert any(n["state"] == "linked" for n in out["names"])


def test_cli_builtin_and_demo(arc, tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.pop("DEEPGRAM_API_KEY", None); env.pop("VERBIT_API_KEY", None); env.pop("VERBIT_WS_URL", None)
    extra = ["--model", MODEL] if MODEL else []
    base = [sys.executable, "-m", "arc_transcriber", str(CLIP), "--domain", "nba", "--names", str(NAMES), "-q"] + extra
    o = tmp_path / "b.json"
    subprocess.run(base + ["--format", "json", "-o", str(o)], check=True, env=env)
    d = json.loads(o.read_text())
    assert d["lines"] and any(n["state"] == "linked" for n in d["names"])
    o = tmp_path / "d.srt"
    subprocess.run(base + ["--asr", "deepgram", "--demo", "--strict", "--format", "srt", "-o", str(o)], check=True, env=env)
    assert "-->" in o.read_text() and "(name: " in o.read_text()
    r = subprocess.run(base + ["--asr", "deepgram"], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "DEEPGRAM_API_KEY" in r.stderr


def test_backbone_aliases(tmp_path):
    from arc_audio import Arc
    for name, folder in (("mini", "mini"), ("edge", "mini"), ("full", "full"), ("parakeet", "full"), ("server", "full")):
        assert Arc.BACKBONES[name] == folder
    with pytest.raises(ValueError):
        Arc.from_pretrained(str(tmp_path), backbone="tiny")
