"""CPU smoke test: downloads WhissleAI/arc-v1 (pinned revision) from Hugging Face on first use; skipped when offline.

    pytest -q                                  # or: python -m pytest tests
    ARC_MODEL=/path/to/arc-v1 pytest -q        # a local copy of the model instead of the download
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
        pytest.skip("offline and no ARC_MODEL: cannot download WhissleAI/arc-v1")
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
