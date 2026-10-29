"""Offline tests of the external-ASR adapters (no network, no keys, no model download): message parsing, is_final handling,
timing, utterance grouping and the word store. Fixtures are hand-written in the public message formats of Deepgram (pre-recorded
REST + live websocket) and Verbit (realtime captions / transcript)."""
import json

from arc_audio.tidy import group_words, tidy_name, tidy_number, dedupe_names, clean_text
from arc_audio.words import WordStore
from arc_transcriber import asr as X

DG_PRERECORDED = {
    "metadata": {"request_id": "00000000-0000-0000-0000-000000000000", "duration": 9.6, "channels": 1},
    "results": {"channels": [{"alternatives": [{
        "transcript": "Marcus Delaney brings the ball up. That puts the Hawks ahead, 102 to 98.",
        "confidence": 0.98,
        "words": [
            {"word": "marcus", "start": 0.48, "end": 0.9, "confidence": 0.99, "punctuated_word": "Marcus"},
            {"word": "delaney", "start": 0.9, "end": 1.4, "confidence": 0.97, "punctuated_word": "Delaney"},
            {"word": "brings", "start": 1.4, "end": 1.7, "confidence": 0.99, "punctuated_word": "brings"},
            {"word": "the", "start": 1.7, "end": 1.8, "confidence": 0.99, "punctuated_word": "the"},
            {"word": "ball", "start": 1.8, "end": 2.1, "confidence": 0.99, "punctuated_word": "ball"},
            {"word": "up", "start": 2.1, "end": 2.4, "confidence": 0.99, "punctuated_word": "up."},
            {"word": "that", "start": 5.0, "end": 5.2, "confidence": 0.99, "punctuated_word": "That"},
            {"word": "puts", "start": 5.2, "end": 5.4, "confidence": 0.99, "punctuated_word": "puts"},
            {"word": "the", "start": 5.4, "end": 5.5, "confidence": 0.99, "punctuated_word": "the"},
            {"word": "hawks", "start": 5.5, "end": 5.9, "confidence": 0.95, "punctuated_word": "Hawks"},
            {"word": "ahead", "start": 5.9, "end": 6.3, "confidence": 0.99, "punctuated_word": "ahead,"},
            {"word": "102", "start": 6.5, "end": 7.1, "confidence": 0.93, "punctuated_word": "102"},
            {"word": "to", "start": 7.1, "end": 7.3, "confidence": 0.99, "punctuated_word": "to"},
            {"word": "98", "start": 7.3, "end": 7.9, "confidence": 0.94, "punctuated_word": "98."}]}]}]}}

DG_LIVE = [
    {"type": "Metadata", "request_id": "00000000-0000-0000-0000-000000000000"},
    {"type": "Results", "is_final": False, "speech_final": False, "start": 0.0, "duration": 1.5,
     "channel": {"alternatives": [{"transcript": "marcus", "words": [{"word": "marcus", "start": 0.48, "end": 0.9, "punctuated_word": "Marcus"}]}]}},
    {"type": "Results", "is_final": True, "speech_final": True, "start": 0.0, "duration": 2.5,
     "channel": {"alternatives": [{"transcript": "Marcus Delaney brings the ball up.", "words": [
         {"word": "marcus", "start": 0.48, "end": 0.9, "punctuated_word": "Marcus"},
         {"word": "delaney", "start": 0.9, "end": 1.4, "punctuated_word": "Delaney"},
         {"word": "up", "start": 2.1, "end": 2.4, "punctuated_word": "up."}]}]}},
    {"type": "UtteranceEnd", "last_word_end": 2.4},
    {"type": "Results", "is_final": True, "channel": {"alternatives": [{"transcript": "", "words": []}]}},
]

VB_CAPTIONS = {"response": {"type": "captions", "id": "c1", "alternatives": [{"items": [
    {"kind": "text", "value": "Theo", "start": 3.0, "end": 3.3},
    {"kind": "text", "value": "Brandt", "start": 3.3, "end": 3.8},
    {"kind": "punct", "value": "."},
    {"kind": "text", "value": "He", "start": 4.0, "end": 4.1},
    {"kind": "text", "value": "scores", "start": 4.1, "end": 4.6},
    {"kind": "punct", "value": "!"}]}]}}
VB_TRANSCRIPT_INTERIM = {"response": {"type": "transcript", "is_final": False, "alternatives": [{"items": [
    {"kind": "text", "value": "theo", "start": 3.0, "end": 3.3}]}]}}
VB_TRANSCRIPT_FINAL = {"response": {"type": "transcript", "is_final": True, "alternatives": [{"items": [
    {"kind": "text", "value": "Theo", "start": 3.0, "end": 3.3}]}]}}
VB_HEARTBEAT = {"event": "heartbeat"}


def test_deepgram_prerecorded():
    w = X.deepgram_words(DG_PRERECORDED)
    assert len(w) == 14 and all(e["is_final"] for e in w)
    assert [e["text"] for e in w[:3]] == ["Marcus", "Delaney", "brings"]          # punctuated_word preferred
    assert w[0]["start"] == 0.48 and w[-1]["end"] == 7.9 and w[-1]["text"] == "98."


def test_deepgram_live_final_and_interim():
    got = [X.deepgram_live_words(json.dumps(m)) for m in DG_LIVE]
    assert got[0] == ([], False)                                                  # Metadata ignored
    w, fin = got[1]
    assert not fin and w[0]["text"] == "Marcus" and w[0]["is_final"] is False
    w, fin = got[2]
    assert fin and [e["text"] for e in w] == ["Marcus", "Delaney", "up."] and all(e["is_final"] for e in w)
    assert got[3] == ([], False)                                                  # UtteranceEnd ignored
    assert got[4] == ([], True)                                                   # empty final


def test_verbit_captions_and_transcript():
    w, fin = X.verbit_words(json.dumps(VB_CAPTIONS), offset_s=10.0)
    assert fin and [e["text"] for e in w] == ["Theo", "Brandt.", "He", "scores!"]  # punctuation attached to the previous word
    assert w[0]["start"] == 13.0 and w[1]["end"] == 13.8                          # offset on the audio clock
    w, fin = X.verbit_words(VB_TRANSCRIPT_INTERIM)
    assert not fin and w[0]["is_final"] is False and w[0]["text"] == "theo"
    assert X.verbit_words(VB_TRANSCRIPT_FINAL) == ([], False)                     # its words arrive as captions
    assert X.verbit_words(VB_HEARTBEAT) == ([], False)


def test_word_store_interim_replaced_by_final():
    ws = WordStore()
    ws.add(X.deepgram_live_words(DG_LIVE[1])[0])
    assert len(ws.interim) == 1 and not ws.finals
    ws.add(X.deepgram_live_words(DG_LIVE[2])[0])
    assert not ws.interim and [w["w"] for w in ws.finals] == ["Marcus", "Delaney", "up."]
    assert ws.covered == 2.4


def test_simulated_messages_round_trip():
    words = [dict(text=e["text"], start=e["start"], end=e["end"]) for e in X.deepgram_words(DG_PRERECORDED)]
    for kind in ("deepgram", "verbit"):
        got = []
        sim = X.Simulated(kind, words, lambda ev: got.extend(e for e in ev if e["is_final"]))
        sim.advance(3.0)
        early = len(got)
        sim.advance(9.6, final=True)
        assert 0 < early < len(words) or kind == "verbit"
        assert [e["text"] for e in got] == [w["text"] for w in words]
        assert all(abs(a["start"] - b["start"]) < 1e-9 for a, b in zip(got, words))


def test_utterance_grouping_like_builtin():
    words = X.deepgram_words(DG_PRERECORDED)
    spans = [(0.4, 2.5), (4.9, 8.0)]                                            # what the utterance segmenter finds on the audio
    U = group_words(words, spans)
    assert len(U) == 2
    assert U[0]["text"] == "Marcus Delaney brings the ball up." and U[1]["text"].endswith("102 to 98.")
    assert U[0]["words"][0] == dict(text="Marcus", start=0.48, end=0.9, is_final=True)
    assert U[0]["start"] == 0.48 and U[1]["end"] == 7.9
    U = group_words(words, [])                                                    # no spans: split at pauses
    assert len(U) == 2


def test_heard_tidy_and_dedupe():
    r = tidy_name(dict(heard="Marcus Delaney brings", entity="Marcus Delaney", start=0.48, end=1.7, words=[
        dict(w="Marcus", start=0.48, end=0.9), dict(w="Delaney", start=0.9, end=1.4), dict(w="brings", start=1.4, end=1.7)]))
    assert r["heard"] == "Marcus Delaney" and r["end"] == 1.4
    assert tidy_name(dict(heard="Theo Brandt.", words=[]))["heard"] == "Theo Brandt"
    assert tidy_number(dict(heard="three minutes left."))["heard"] == "three minutes"
    assert clean_text("102 to 98.") == "102 to 98"
    a = dict(start=3.0, end=3.8, entity="Theo Brandt", heard="Theo Brandt", state="linked", via="exact")
    b = dict(start=3.3, end=3.8, entity="Theo Brandt", heard="Brandt", state="linked", via="fallback")
    assert dedupe_names([b, a]) == [a]


def test_spelled_score_pair():
    from arc_audio import textlayer as TL
    v, _ = TL.text_value("score", ["one", "hundred", "and", "two", "to", "ninety", "eight."])
    assert (v["v"], v["v2"]) == (102, 98)
