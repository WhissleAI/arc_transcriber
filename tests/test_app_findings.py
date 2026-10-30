"""1.2.1 runtime / API fixes: utterance closing for short commands and Stream.end_turn(), whole-span name rewrite, opt-in switch
interval, kb_snap command words and ranked candidates, the list check before not_in_list, live hold-back, command_complete
hold-off and per-domain output masks. Unit tests run anywhere; the model tests use Arc Mini (skipped when offline and no ARC_MODEL).
"""
import gc, os, socket, sys, types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "examples" / "example_synth.wav"
MODEL = os.environ.get("ARC_MODEL")
SR, FR = 16000, 1600


def online():
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
        return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def arc():
    if not MODEL and not online():
        pytest.skip("offline and no ARC_MODEL: cannot download WhissleAI/arc-v1.1")
    from arc_transcriber import load
    return load(MODEL) if MODEL else load()


# ---- 1. short utterances close on a pause ----------------------------------------------------------------------------------

def _segment(levels, **cfg):
    from arc_audio.transcribe import UttSegmenter
    seg = UttSegmenter(cfg)
    out = []
    for f, db in enumerate(levels):
        out += [(f0, f1, why, f) for f0, f1, why in seg.feed(f, db)]
    return out, seg


def test_short_utterance_closes_on_pause():
    levels = [-70.0] * 30 + [-20.0] * 5 + [-70.0] * 200          # 0.5 s of speech, then 20 s of silence
    out, _ = _segment(levels)
    assert len(out) == 1
    f0, f1, why, at = out[0]
    assert why == "pause" and (f0, f1) == (30, 34)
    assert at - f1 <= 8                                              # closed 0.8 s after the speech (was 15 s: max_s)
    out, _ = _segment(levels, short_pause_s=None)                    # the 1.2.0 rule, for comparison
    assert out[0][2] == "max_length" and out[0][3] - 34 > 100


def test_long_utterance_still_closes_after_short_pause():
    levels = [-70.0] * 30 + [-20.0] * 15 + [-70.0] * 50
    out, _ = _segment(levels)
    assert out[0][2] == "pause" and out[0][3] - out[0][1] == 4       # 0.4 s pause, as before


def test_short_hesitation_does_not_split():
    # 0.5 s of speech, a 0.5 s pause, more speech: one utterance (min_s keeps short pieces together under short_pause_s)
    levels = [-70.0] * 30 + [-20.0] * 5 + [-70.0] * 5 + [-20.0] * 10 + [-70.0] * 30
    out, _ = _segment(levels)
    assert len(out) == 1 and out[0][:2] == (30, 49)


# ---- 2. transcript_rewritten: the whole heard span or none of it -------------------------------------------------------------

def _words(text, t0=0.0, tags=None):
    out = []
    for k, w in enumerate(text.split()):
        d = dict(text=w, start=round(t0 + 0.4 * k, 3), end=round(t0 + 0.4 * k + 0.3, 3))
        if tags and tags[k]:
            d["tag"], d["tag_begin"] = tags[k]
        out.append(d)
    return out


def _rewrite(text, heard_idx, entity, tags=None):
    from arc_audio.api import rewrite
    ws = _words(text, tags=tags)
    rec = dict(state="linked", entity=entity, words=[dict(w=ws[k]["text"], start=ws[k]["start"], end=ws[k]["end"]) for k in heard_idx])
    return rewrite([dict(start=0.0, end=10.0, text=text, words=ws)], [rec])[0]["text"]


def test_rewrite_partial_link_takes_whole_span():
    # "Priya" of "Priya Raymond" linked to "Priya Raman" gave "Priya Raman Raymond"
    assert _rewrite("Hi, it's Priya Raymond, I need help.", [2], "Priya Raman") == "Hi, it's Priya Raman, I need help."
    tags = [None, None, ("ENTITY_PERSON", True), ("ENTITY_PERSON", False), None, None, None]
    assert _rewrite("Hi, it's priya raymond, I need help.", [2], "Priya Raman", tags) == "Hi, it's Priya Raman, I need help."


def test_rewrite_leaves_span_when_rest_is_not_the_entity():
    # two extra heard words for one missing entity word: not a reading of the entity -> no rewrite (never a mix)
    tags = [None, ("ENTITY_PERSON", True), ("ENTITY_PERSON", False), ("ENTITY_PERSON", False), None]
    out = _rewrite("Ask priya van raymond now.", [1], "Priya Raman", tags)
    assert out == "Ask priya van raymond now."


def test_rewrite_unchanged_cases():
    assert _rewrite("Delaney to Theo Brandt.", [3], "Theo Brandt") == "Delaney to Theo Brandt."
    assert _rewrite("Pass it to Brandt now.", [3], "Theo Brandt") == "Pass it to Theo Brandt now."
    # a capitalised word that does not sound like the missing entity word stays ("Monday" is not "Raman")
    assert _rewrite("I saw Priya Monday morning.", [2], "Priya Raman") == "I saw Priya Raman Monday morning."
    # a capitalised sentence start before a surname is not taken as the first name
    assert _rewrite("Okay. Ask Raman about it.", [2], "Priya Raman") == "Okay. Ask Priya Raman about it."


# ---- 10. kb_snap: command words, ranked candidates --------------------------------------------------------------------------

BOOKS = [dict(name="The Pilgrim's Progress by John Bunyan Every Child Can Read", kind="work_product"),
         dict(name="Frankenstein", kind="work_product"), dict(name="Dracula", kind="work_product")]


def test_kb_snap_ignores_command_words():
    from arc_audio.kbsnap import KB, add_spans
    kb = KB(BOOKS)
    ws = [dict(text="Read", start=0.0, end=0.3), dict(text="Frankenstein.", start=0.35, end=1.0)]
    n = iter(range(100))
    out = add_spans(ws, [], kb, lambda: f"nm{next(n)}")
    assert [(r["heard"], r["entity"]) for r in out] == [("Frankenstein.", "Frankenstein")]
    assert kb.ranked("Read", 3) == []                                 # a command word alone matches nothing
    ws = [dict(text="Read", start=0.0, end=0.3)]
    assert add_spans(ws, [], kb, lambda: "nm") == []


def test_kb_snap_ranked_candidates():
    from arc_audio.kbsnap import KB, link
    kb = KB(BOOKS)
    r = kb.ranked("Read Frankenstein", 3)
    assert r[0][1] == "Frankenstein" and r[0][0] == 1.0 and len(r) == 3 and r[0][0] >= r[1][0] >= r[2][0]
    rec = dict(heard="Read Frankenstein", state="not_in_list")
    assert link(rec, kb) and rec["entity"] == "Frankenstein"
    assert rec["snap_candidates"][0] == dict(entity="Frankenstein", score=1.0)


def test_rescorer_returns_ranked_list():
    from arc_audio.rescorer import NameRescorer
    kb = [dict(name=n) for n in ("Marcus Delaney", "Theo Brandt", "Owen Castellano")]
    rs = NameRescorer(kb, dict(thr=0.5))
    z = np.array([0.0, -1e4, 1.0, 3.0, -1.0])
    r = rs.score(1.0, z, np.arange(3), "Theo Brandt")
    assert [row for row, _ in r["ranked"]] == [1, 0, 2]
    assert abs(sum(p for _, p in r["ranked"]) + r["p_new"] - 1.0) < 1e-3


# ---- 12. the list check before not_in_list; live hold-back ------------------------------------------------------------------

ROSTER = [dict(name="De'Marco Hale", kind="person"), dict(name="Riverside Comets", kind="team"),
          dict(name="Sam Okoro", kind="person"), dict(name="Sam Lindqvist", kind="person"), dict(name="Evan Ruiz", kind="person")]


def test_alias_index():
    from arc_audio.kbsnap import AliasIndex
    ix = AliasIndex(ROSTER)
    assert ix.lookup("DeMarco") == (0, "partial")
    assert ix.lookup("DeMarco Hale") == (0, "alias")
    assert ix.lookup("De Marco Hale's") == (0, "alias")
    assert ix.lookup("the Comets") == (1, "partial")
    assert ix.lookup("Comets", allowed={0, 2, 3, 4}) is None      # wrong kind for the span
    assert ix.lookup("Sam") is None                                   # two entries: ambiguous
    assert ix.lookup("Okoro") == (2, "partial")
    assert ix.lookup("Harbor City") is None


def _fake_engine(kb, n_cands=5):
    from arc_audio.kbsnap import AliasIndex
    return types.SimpleNamespace(kb=kb, n_cands=n_cands, alias=AliasIndex(kb))


def test_not_in_list_checked_against_aliases():
    from arc_audio.engine import Engine
    eng = _fake_engine(ROSTER)
    rec = dict(heard="Comets", state="not_in_list", entity=None, p=0.81)
    r = dict(pred=None, p=0.81, link=False, ranked=[(1, 0.15), (4, 0.04)], p_new=0.81)
    Engine._name_extras(eng, rec, r, None, np.array([1]))
    assert (rec["state"], rec["entity"], rec["via"], rec["linker_state"]) == ("linked", "Riverside Comets", "partial", "not_in_list")
    assert rec["candidates"][0] == dict(entity="Riverside Comets", p=0.15) and rec["p_not_in_list"] == 0.81
    rec = dict(heard="Sam", state="not_in_list", entity=None, p=0.7)
    Engine._name_extras(eng, rec, None, None, np.array([0, 2, 3, 4]))
    assert rec["state"] == "not_in_list"
    off = types.SimpleNamespace(kb=ROSTER, n_cands=0, alias=None)
    rec = dict(heard="DeMarco", state="not_in_list", entity=None, p=0.7)
    Engine._name_extras(off, rec, r, None, np.array([0]))
    assert rec["state"] == "not_in_list" and "candidates" not in rec


# ---- 16. command_complete hold-off ------------------------------------------------------------------------------------------

def _cc_run(hold, frames):
    """frames: list of (cc_fires, is_speech); returns the released events"""
    from arc_audio.engine import Engine
    eng = types.SimpleNamespace(cc_hold=hold, cc_pend=None, out=dict(events=[]))
    eng._cc_step = types.MethodType(Engine._cc_step, eng)
    eng._cc_release = types.MethodType(Engine._cc_release, eng)
    for f, (fire, speech) in enumerate(frames):
        if fire:
            ev = dict(tag="discourse.assistant.command_complete", t=f / 10, p=0.5, window="at")
            if eng.cc_pend is None:
                eng.cc_pend = dict(ev=ev, quiet=0)
            else:
                eng.cc_pend["ev"].update(t=ev["t"])
                eng.cc_pend["quiet"] = 0
        if eng.cc_pend is not None:
            eng._cc_step(f, speech)
    return eng.out["events"]


def test_command_complete_hold_off():
    # "set an alarm at nine am | on next Sunday": fires at "am", 0.3 s pause, speech resumes, then silence
    fr = [(False, True)] * 10 + [(True, True)] + [(False, False)] * 3 + [(False, True)] * 8 + [(False, False)] * 10
    ev = _cc_run(5, fr)
    assert len(ev) == 1 and ev[0]["t"] == 1.0 and ev[0]["released_t"] == 2.7 and ev[0]["hold_off_s"] == 0.5
    ev = _cc_run(2, fr)                                              # a hold-off shorter than the pause releases at "am"
    assert ev[0]["released_t"] == 1.3


# ---- model tests (Arc Mini) -------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wav():
    from arc_audio import load_audio
    return load_audio(str(CLIP))


@pytest.fixture(scope="module")
def first_word(arc, wav):
    u = arc.transcribe(wav[:int(7.4 * SR)])[0]
    w = u["words"][0]                                                  # "Welcome"
    a, b = int(max(0.0, w["start"] - 0.05) * SR), int((w["end"] + 0.05) * SR)
    return wav[a:b]


def _quiet(n, seed=0):
    return (np.random.default_rng(seed).standard_normal(n) * 1e-3).astype(np.float32)


def test_stream_short_word_final_after_pause(arc, first_word):
    x = np.concatenate([_quiet(SR // 2), first_word, _quiet(20 * SR)])
    end = (SR // 2 + len(first_word)) / SR
    s = arc.stream(domain="assistant", background=False)
    t_final = None
    for i in range(0, len(x), FR):
        o = s.feed(x[i:i + FR])
        if t_final is None and o["transcript"]:
            t_final = (i + FR) / SR - end
    s.flush()
    assert t_final is not None and t_final < 1.5, t_final                 # was ~14.5 s


def test_end_turn_closes_without_ending_stream(arc, wav):
    x = wav[:int(7.3 * SR)]
    s = arc.stream(domain="nba", background=True)
    got = []
    for i in range(0, len(x), FR):
        got += s.feed(x[i:i + FR])["transcript"]
    assert not any("Delaney" in u["text"] for u in got)                  # no pause after the sentence yet: not final
    o = s.end_turn()
    assert len(o["transcript"]) == 1 and "Delaney" in o["transcript"][0]["text"]
    assert o["transcript"][0]["reason"] == "end_turn"
    s.set_context(names=["Theo Brandt"], previous_text="fourth quarter")   # per-turn context on the running stream
    y = wav[int(7.3 * SR):int(13.5 * SR)]
    for i in range(0, len(y), FR):
        got += s.feed(y[i:i + FR])["transcript"]
    assert s.tr.ctx and s.tr.ctx["kb"] == ["Theo Brandt"] and s.tr.ctx["prev"] == "fourth quarter"
    f = s.flush()
    assert any("drives" in u["text"] for u in got + f["transcript"])


def test_switch_interval_opt_in_and_restored(arc):
    before = sys.getswitchinterval()
    s = arc.stream(domain="assistant", background=True)
    assert sys.getswitchinterval() == before                             # default: untouched
    s.feed(np.zeros(FR, np.float32))
    s.flush()
    with arc.stream(domain="assistant", background=True, switch_interval_s=True) as s2:
        assert sys.getswitchinterval() < before
        s2.feed(np.zeros(FR, np.float32))
    assert sys.getswitchinterval() == before                             # restored by the context manager (no flush)
    s3 = arc.stream(domain="assistant", background=True, switch_interval_s=0.001)
    s3.feed(np.zeros(FR, np.float32))
    assert sys.getswitchinterval() == min(before, 0.001)
    del s3
    gc.collect()
    assert sys.getswitchinterval() == before                             # restored when an unflushed stream is collected


def test_output_masks_assistant(arc, wav):
    x = wav[:int(7.3 * SR)]
    ids = lambda **k: {t["id"] for t in arc.tags("assistant", include_experimental=True, **k)}
    assert not any(t.startswith("safety.profanity.") for t in ids())
    assert any(t.startswith("safety.profanity.") for t in ids(masks=None))
    assert any(t.startswith("safety.profanity.") for t in {t["id"] for t in arc.tags("talk", include_experimental=True)})
    o = arc.predict(x, domain="assistant", include_experimental=True, events="all", return_frames=False)
    assert o["bio_events"] == []
    assert not any(e["label"].startswith("bio_ev.") for e in o.get("audio_events") or [])
    tags = [k for u in o["utterances"] for k in u["tags"]] + [s["tag"] for s in o["spans"]] + [e["tag"] for e in o["events"]]
    assert not any(t.startswith("safety.profanity.") for t in tags)
    assert arc.output_masks("assistant", masks={"assistant": ["bio."]}) == ("bio.",)
    assert arc.output_masks("nba") == ()


def test_device_directed_fields(arc, wav):
    o = arc.predict(wav[:int(7.3 * SR)], domain="assistant", return_frames=False, events=None)
    dd = o["transcript"][0]["device_directed"]
    assert dd["source"] == "head" and set(dd) >= {"p", "directed", "threshold", "status"}
    assert "discourse.assistant.device_directed" in o["utterances"][0]["tags"]


def test_name_candidates_and_live_hold(arc, wav):
    names = ["Marcus Delaney", "Theo Brandt", "Owen Castellano"]
    o = arc.predict(wav, domain="nba", names=names, return_frames=False, events=None)
    linked = [r for r in o["names"] if r.get("state") == "linked" and r.get("candidates")]
    assert linked and all(r["candidates"][0]["p"] >= r["candidates"][-1]["p"] for r in linked)
    o = arc.predict(wav, domain="nba", names=names, return_frames=False, events=None, name_candidates=0)
    assert not any("candidates" in r for r in o["names"])
    s = arc.stream(domain="nba", names=names, background=False, live_hold=dict(name_p=1.01, min_span_s=0.0))
    shown = []
    for i in range(0, int(13.5 * SR), FR):
        lv = s.feed(wav[i:i + FR]).get("live") or {}
        shown += [sp["name_guess"] for sp in lv.get("open_spans") or [] if sp.get("name_guess") and sp["name_guess"].get("state") == "linked"]
    s.flush()
    assert shown == []                                                   # every linked first guess held back (p < 1.01)
    with pytest.raises(TypeError):
        arc.stream(domain="nba", no_such_option=1)
