"""Written form (arc_audio.written), the date record's month, and devices (mps / auto).

    pytest -q tests/test_written.py          # text tests run anywhere; the spoken clips need macOS `say` and the model
"""
import os, shutil, socket, subprocess, sys
from pathlib import Path

import numpy as np
import pytest

from arc_audio.written import to_written, render_span, written_words, parse_date, date_iso, spell_names

ROOT = Path(__file__).resolve().parents[1]
MODEL = os.environ.get("ARC_MODEL")

SPOKEN_TO_WRITTEN = [
    # time
    ("The meeting moved to two thirty p.m. today.", "The meeting moved to 2:30 p.m. today."),
    ("Call me at nine oh five tomorrow.", "Call me at 9:05 tomorrow."),
    ("Please arrive by eight p.m.", "Please arrive by 8 p.m."),
    ("Five o'clock works.", "5 o'clock works."),
    # money
    ("The ticket costs forty five dollars.", "The ticket costs $45."),
    ("It costs twelve euros fifty.", "It costs €12.50."),
    ("That's forty five dollars and fifty cents.", "That's $45.50."),
    ("The budget is two million dollars.", "The budget is $2 million."),
    ("Sales hit five hundred seventy eight thousand dollars.", "Sales hit $578,000."),
    # dates
    ("We meet on March third.", "We meet on March 3."),
    ("She was born on March third, twenty twenty four.", "She was born on March 3, 2024."),
    ("On the third of March we leave.", "On the 3rd of March we leave."),
    ("the twelfth of December, nineteen ninety nine", "the 12th of December, 1999"),
    # percents, decimals, cardinals, ordinals
    ("Sales grew twelve point five percent this year.", "Sales grew 12.5% this year."),
    ("There were one hundred and twenty three people.", "There were 123 people."),
    ("We need a hundred chairs.", "We need 100 chairs."),
    ("It weighs about two point five kilos.", "It weighs about 2.5 kilos."),
    ("He finished twenty first in the race.", "He finished 21st in the race."),
    ("He scored twenty six points.", "He scored 26 points."),
]

KEEP_WORDS = [
    "I have two kids and one dog.",         # single-digit counts stay words
    "the first time",                       # ordinals 1-9 stay words
    "Oh, I love it.",                       # 'oh' / 'love' are not numbers outside a span
    "no one came",
    "two thirty",                           # no tag, no 'at', no 'p.m.': a time or 230? keep the words
    "Twelve fifty for the pair.",           # money? a time? keep
    "I may fifth",                          # 'may' the verb is not a month
    "between two and three",
]


@pytest.mark.parametrize("spoken,written", SPOKEN_TO_WRITTEN)
def test_spoken_to_written(spoken, written):
    assert to_written(spoken) == written


@pytest.mark.parametrize("text", KEEP_WORDS)
def test_keeps_words_when_unsure(text):
    assert to_written(text) == text


def test_spell_small_off():
    assert to_written("I have two kids", spell_small=False) == "I have 2 kids"
    assert to_written("the third floor", spell_small=False) == "the 3rd floor"


def test_kinds_from_tags():
    assert render_span(["two", "thirty"], "time_of_day") == "2:30"
    assert render_span(["two", "thirty"], None) is None                       # untyped NUMBER, no time context
    assert render_span(["three", "one"], "score", "soccer") == "3–1"
    assert render_span(["one", "hundred", "and", "two", "to", "ninety", "eight."], "score", "nba") == "102–98."
    assert render_span(["one", "hundred", "and", "two", "to", "ninety", "eight"], "score", "news") == "102 to 98"
    assert render_span(["five", "five", "five", "one", "two", "three", "four"], "phone") == "555-1234"
    assert render_span(["eight", "hundred", "five", "five", "five", "one", "two", "one", "two"], "phone") == "800-555-1212"
    assert render_span(["two", "oh", "five"], "id_pii") == "205"
    assert render_span(["double", "seven", "three"], "id_pii") == "773"
    assert render_span(["four", "twelve"], "ref_number") == "412"
    assert render_span(["two", "thirty", "left"], "clock") == "2:30 left"
    assert render_span(["nineteen", "ninety"], "date") == "1990"
    assert render_span(["fourth"], "ordinal") is None
    assert render_span(["2:30"], "time_of_day") is None                       # already written: unchanged


def test_tagged_words_and_times():
    W = [dict(text="moved", start=0.5, end=0.8), dict(text="to", start=0.8, end=1.0),
         dict(text="two", start=1.0, end=1.3, tag="NUMBER_TIME", tag_begin=True), dict(text="thirty", start=1.3, end=1.7, tag="NUMBER_TIME"),
         dict(text="on", start=1.7, end=1.9),
         dict(text="March", start=1.9, end=2.3, tag="NUMBER_DATE", tag_begin=True), dict(text="third,", start=2.3, end=2.7, tag="NUMBER_DATE"),
         dict(text="costs", start=2.9, end=3.2),
         dict(text="forty", start=3.5, end=3.9, tag="NUMBER_MONEY", tag_begin=True), dict(text="five", start=3.9, end=4.1, tag="NUMBER_MONEY"),
         dict(text="dollars.", start=4.1, end=4.5, tag="NUMBER_MONEY")]
    text, words = written_words(W)
    assert text == "moved to 2:30 on March 3, costs $45."
    w = [x for x in words if x.get("written_from") == "two thirty"][0]
    assert (w["start"], w["end"], w["kind"]) == (1.0, 1.7, "time_of_day")


def test_date_parts():
    assert parse_date(["March", "third"]) == dict(year=None, month=3, day=3)
    assert parse_date(["the", "third", "of", "March"]) == dict(year=None, month=3, day=3)
    assert parse_date(["March", "third,", "twenty", "twenty", "four"]) == dict(year=2024, month=3, day=3)
    assert parse_date(["February", "thirtieth"]) is None
    assert date_iso(dict(year=None, month=3, day=3)) == "--03-03"
    assert date_iso(dict(year=2024, month=3, day=3)) == "2024-03-03"
    assert date_iso(dict(year=2024, month=None, day=None)) == "2024"


def test_date_record_keeps_month():
    from arc_audio.api import extend_number
    from arc_audio.tidy import tidy_number
    W = [dict(text="on", start=1.7, end=1.9), dict(text="March", start=1.96, end=2.3), dict(text="third,", start=2.28, end=2.72),
         dict(text="and", start=2.7, end=2.9)]
    r = dict(start=1.96, end=2.72, kind="date", heard="March third", value=3, value2=None)   # as Arc's decoder record
    r = tidy_number(extend_number(r, W))
    assert (r["heard"], r["date"], r["iso"], r["value"]) == ("March third", dict(year=None, month=3, day=3), "--03-03", 3)
    r = dict(start=1.0, end=1.3, kind="date", heard="third", value=3, value2=None)            # 'the third of March': span is 'third'
    W = [dict(text="the", start=0.8, end=1.0), dict(text="third", start=1.0, end=1.3), dict(text="of", start=1.3, end=1.4),
         dict(text="March.", start=1.4, end=1.8)]
    r = tidy_number(extend_number(r, W))
    assert (r["heard"], r["iso"], r["end"]) == ("third of March", "--03-03", 1.8)


def test_spell_names_does_not_add_words():
    W = [dict(text="Brant", start=1.0, end=1.3), dict(text="drives.", start=1.3, end=1.6)]
    names = [dict(state="linked", entity="Theo Brandt", words=[dict(w="Brant", start=1.0, end=1.3)])]
    assert [w["text"] for w in spell_names(W, names)] == ["Brandt", "drives."]


def test_resolve_device():
    import torch
    from arc_audio.api import resolve_device
    assert resolve_device("cpu") == "cpu"
    auto = resolve_device("auto")
    assert auto == ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if not torch.backends.mps.is_available():
        with pytest.warns(RuntimeWarning):
            assert resolve_device("mps") == "cpu"


def _gpu():
    import torch
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None


@pytest.mark.skipif(_gpu() is None, reason="needs an MPS or CUDA device")
def test_server_features_on_gpu():
    """Arc Transcriber's feature path (ServerFeatures) on a GPU device gives the CPU result (it used to fail on MPS with
    'Passed CPU tensor to MPS op'). A stand-in backbone replaces the Parakeet encoder, so no download is needed."""
    import torch
    from arc_audio.server import Heads, ServerFeatures

    class BB:
        def __call__(self, x, lengths, want_tap=True):
            T = int(lengths[0]) // 1280 + 1
            P = torch.randn(1, T, 1024, generator=torch.Generator().manual_seed(int(lengths[0]))).to(x.device)
            return P, torch.ones(1, T, dtype=torch.bool, device=x.device), (P if want_tap else None)
    outs = {}
    for dev in ("cpu", _gpu()):
        torch.manual_seed(0)
        sf = ServerFeatures(BB(), Heads().to(dev).eval(), chunk_s=2.0, ctx_s=0.5, tap=True)
        xs, _ = sf.feed(np.zeros(16000 * 5, np.float32))
        x2, _ = sf.flush()
        outs[dev] = np.stack(xs + x2)
    assert np.abs(outs["cpu"] - outs[_gpu()]).max() < 1e-3


def test_pool_device():
    from arc_audio.model import zero_carry, pool_device
    assert zero_carry(1, 8, "mps")["utt"][0].device.type == "cpu"     # MPS has no float64: the pooled sums stay on the CPU
    assert pool_device("cuda").type == "cuda" and pool_device("cpu").type == "cpu"


# ---------------------------------------------------------------- spoken clips (macOS `say` + the model)

def online():
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
        return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def arc():
    if sys.platform != "darwin" or not shutil.which("say"):
        pytest.skip("spoken clips need macOS `say`")
    if not MODEL and not online() and not os.environ.get("HF_HUB_OFFLINE"):
        pytest.skip("offline and no ARC_MODEL")
    from arc_transcriber import load
    return load(MODEL) if MODEL else load()


def say(tmp_path, text, voice="Samantha"):
    aiff, wav = tmp_path / "x.aiff", tmp_path / f"{abs(hash(text))}.wav"
    subprocess.run(["say", "-v", voice, "-o", str(aiff), "--", text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], check=True)
    return str(wav)


def test_spoken_clip_written(arc, tmp_path):
    """'2:30', 'March 3rd' and '45 dollars' read aloud: Arc's transcript has words; transcript_written has digits; the date
    record covers the month."""
    out = arc.predict(say(tmp_path, "The meeting moved to 2:30 on March 3rd, and the ticket costs 45 dollars."), return_frames=False)
    text = " ".join(u["text"] for u in out["transcript"])
    written = " ".join(u["text"] for u in out["transcript_written"])
    assert "forty five dollars" in text and "$45" in written
    assert "March 3," in written
    d = [n for n in out["numbers"] if n.get("kind") == "date"]
    assert d and d[0].get("iso") == "--03-03" and "March" in d[0]["heard"]


def test_transcribe_written_flag(arc, tmp_path):
    u = arc.transcribe(say(tmp_path, "Sales grew 12.5 percent."), written=True)[0]
    assert "12.5%" in u["text_written"] and "text_written" not in arc.transcribe(say(tmp_path, "Sales grew 12.5 percent."))[0]
