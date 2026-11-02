"""Offline tests of arc_audio.integrations (no keys, no network beyond localhost, no model download): message parsing of
Deepgram / AssemblyAI / Speechmatics streaming results into Arc external words, coverage, the full adapter loop against a
local mock websocket server (finalise, end of stream, reconnect with resend on the same clock, give-up on a bad key),
the fallback to Arc's own decoder, and key scrubbing. Fixtures follow the providers' documented message formats.

Live test (opt-in, needs keys and network): ARC_LIVE_ASR=deepgram,assemblyai,speechmatics pytest -q tests/test_integrations.py
"""
import asyncio
import json
import logging
import os

import numpy as np
import pytest

from arc_audio.integrations import Options, ProviderError, get, scrub
from arc_audio.integrations import assemblyai as AA, deepgram as DG, speechmatics as SM
from arc_audio.integrations.base import ScrubFilter
from arc_audio.words import WordStore

# ---------------------------------------------------------------- fixtures (documented formats; times as returned)
DG_MSGS = [
    {"type": "Results", "start": 0.0, "duration": 1.0, "is_final": False, "speech_final": False,
     "channel": {"alternatives": [{"transcript": "Why was the", "words": [
         {"word": "why", "start": 0.32, "end": 0.56, "confidence": 0.998, "punctuated_word": "Why"},
         {"word": "was", "start": 0.56, "end": 0.8, "confidence": 1.0, "punctuated_word": "was"},
         {"word": "the", "start": 0.8, "end": 0.88, "confidence": 0.99, "punctuated_word": "the"}]}]}},
    {"type": "Results", "start": 0.0, "duration": 2.5, "is_final": True, "speech_final": False,
     "channel": {"alternatives": [{"transcript": "Why was the sacrament", "words": [
         {"word": "why", "start": 0.32, "end": 0.64, "confidence": 1.0, "punctuated_word": "Why"},
         {"word": "was", "start": 0.64, "end": 0.88, "confidence": 1.0, "punctuated_word": "was"},
         {"word": "the", "start": 0.88, "end": 1.04, "confidence": 0.999, "punctuated_word": "the"},
         {"word": "sacrament", "start": 1.04, "end": 1.52, "confidence": 0.99, "punctuated_word": "sacrament?"}]}]}},
    {"type": "UtteranceEnd", "last_word_end": 1.52},
    {"type": "Results", "start": 2.5, "duration": 0.6, "is_final": True, "from_finalize": True,
     "channel": {"alternatives": [{"transcript": "", "words": []}]}},
    {"type": "Metadata", "request_id": "00000000-0000-0000-0000-000000000000", "duration": 3.1},
]

AA_MSGS = [
    {"type": "Begin", "id": "00000000-0000-0000-0000-000000000000", "expires_at": 0,
     "configuration": {"model": "universal-streaming-english"}},
    {"type": "Turn", "turn_order": 0, "turn_is_formatted": False, "end_of_turn": False, "transcript": "",
     "words": [{"start": 720, "end": 800, "text": "why", "confidence": 0.77, "word_is_final": False}]},
    {"type": "Turn", "turn_order": 0, "turn_is_formatted": False, "end_of_turn": False, "transcript": "why",
     "words": [{"start": 720, "end": 800, "text": "why", "confidence": 0.77, "word_is_final": True},
               {"start": 880, "end": 960, "text": "was", "confidence": 0.95, "word_is_final": False}]},
    {"type": "Turn", "turn_order": 0, "turn_is_formatted": False, "end_of_turn": True, "transcript": "why was it",
     "words": [{"start": 720, "end": 800, "text": "why", "confidence": 0.77, "word_is_final": True},
               {"start": 880, "end": 960, "text": "was", "confidence": 0.96, "word_is_final": True},
               {"start": 1040, "end": 1120, "text": "it", "confidence": 0.98, "word_is_final": True}]},
    {"type": "Turn", "turn_order": 1, "turn_is_formatted": False, "end_of_turn": True, "transcript": "", "words": []},
    {"type": "Termination", "audio_duration_seconds": 3, "session_duration_seconds": 4},
]

SM_MSGS = [
    {"message": "RecognitionStarted", "id": "00000000-0000-0000-0000-000000000000",
     "language_pack_info": {"language_description": "English"}},
    {"message": "Info", "type": "recognition_quality", "reason": "Running recognition using a broadcast model quality."},
    {"message": "AudioAdded", "seq_no": 1},
    {"message": "AddPartialTranscript", "metadata": {"start_time": 0.3, "end_time": 0.9, "transcript": "why was"},
     "results": [{"type": "word", "start_time": 0.32, "end_time": 0.6, "alternatives": [{"content": "why", "confidence": 0.5}]},
                 {"type": "word", "start_time": 0.6, "end_time": 0.88, "alternatives": [{"content": "was"}]}]},
    {"message": "AddTranscript", "metadata": {"start_time": 0.3, "end_time": 1.6, "transcript": "Why was the sacrament?"},
     "results": [{"type": "word", "start_time": 0.32, "end_time": 0.6, "alternatives": [{"content": "Why", "confidence": 0.99}]},
                 {"type": "word", "start_time": 0.6, "end_time": 0.88, "alternatives": [{"content": "was", "confidence": 0.98}]},
                 {"type": "word", "start_time": 0.88, "end_time": 1.04, "alternatives": [{"content": "the", "confidence": 1.0}]},
                 {"type": "word", "start_time": 1.04, "end_time": 1.52, "alternatives": [{"content": "sacrament", "confidence": 0.97}]},
                 {"type": "punctuation", "start_time": 1.52, "end_time": 1.52, "attaches_to": "previous", "is_eos": True,
                  "alternatives": [{"content": "?", "confidence": 1.0}]}]},
    {"message": "EndOfUtterance", "metadata": {"start_time": 1.6, "end_time": 1.6}},
    {"message": "EndOfTranscript"},
]


def finals(msgs, adapter):
    got = []
    adapter.on_words = lambda ev: got.extend(ev)
    for m in msgs:
        adapter.handle(m)
    return got


# ---------------------------------------------------------------- parsing
def test_deepgram_parse():
    p = DG.parse(DG_MSGS[0])
    assert not p.final and [w["text"] for w in p.words] == ["Why", "was", "the"] and p.covered is None
    assert p.words[0] == dict(text="Why", start=0.32, end=0.56, is_final=False, confidence=0.998)
    p = DG.parse(json.dumps(DG_MSGS[1]))
    assert p.final and p.covered == 2.5 and p.words[-1]["text"] == "sacrament?" and all(w["is_final"] for w in p.words)
    assert DG.parse(DG_MSGS[2]).endpoint and not DG.parse(DG_MSGS[2]).words
    p = DG.parse(DG_MSGS[3])
    assert p.final and p.covered == pytest.approx(3.1) and p.endpoint and not p.words
    assert not DG.parse(DG_MSGS[4]).done and DG.parse(DG_MSGS[4], ending=True).done
    e = DG.parse({"type": "Error", "err_code": "INVALID_AUTH", "err_msg": "bad credentials"})
    assert e.error and not e.retry


def test_assemblyai_parse_finals_once():
    turns = {}
    out = [AA.parse(m, turns) for m in AA_MSGS]
    assert out[0].info["model"] == "universal-streaming-english"
    assert [(w["text"], w["is_final"]) for w in out[1].words] == [("why", False)]
    assert [(w["text"], w["is_final"]) for w in out[2].words] == [("why", True), ("was", False)]
    assert out[2].covered == 0.8
    assert [(w["text"], w["is_final"]) for w in out[3].words] == [("was", True), ("it", True)]   # "why" not repeated
    assert out[3].endpoint and out[3].covered == 1.12 and out[3].words[0]["start"] == 0.88       # ms -> s
    assert out[4].endpoint and not out[4].words
    assert out[5].done


def test_session_limits_are_retried_bad_keys_are_not():
    p = AA.parse({"error": "Unauthorized Connection: Too many concurrent sessions"})
    assert p.error and p.retry and p.wait_s >= 1
    assert not AA.parse({"error": "Invalid API key"}).retry
    a = AA.AssemblyAI(on_words=None)
    assert a.classify_close(1008, "Too many concurrent sessions")[0] and not a.classify_close(1008, "Unauthorized")[0]
    assert SM.parse({"message": "Error", "type": "quota_exceeded", "reason": "Concurrent Quota Exceeded"}).wait_s >= 5


def test_assemblyai_formatted_turns_wait_for_the_formatted_copy():
    turns = {}
    m = dict(AA_MSGS[3])
    assert not AA.parse(m, turns, formatted=True).final                       # unformatted end of turn: interim only
    f = dict(m, turn_is_formatted=True, words=[dict(w, text=t) for w, t in zip(m["words"], ["Why", "was", "it?"])])
    p = AA.parse(f, turns, formatted=True)
    assert p.final and [w["text"] for w in p.words] == ["Why", "was", "it?"]


def test_speechmatics_parse():
    assert SM.parse(SM_MSGS[0]).info["language_pack"] == "English"
    assert SM.parse(SM_MSGS[1]).words == [] and SM.parse(SM_MSGS[2]).words == []
    p = SM.parse(SM_MSGS[3])
    assert not p.final and [w["text"] for w in p.words] == ["why", "was"] and p.words[0]["confidence"] is None
    p = SM.parse(SM_MSGS[4])
    assert p.final and p.covered == 1.6 and [w["text"] for w in p.words] == ["Why", "was", "the", "sacrament?"]
    assert SM.parse(SM_MSGS[5]).endpoint and SM.parse(SM_MSGS[6]).done
    e = SM.parse({"message": "Error", "type": "not_authorised", "reason": "bad key"})
    assert e.error and not e.retry
    assert SM.parse({"message": "Error", "type": "quota_exceeded", "reason": "x"}).retry


@pytest.mark.parametrize("provider,msgs,text", [
    ("deepgram", DG_MSGS, ["Why", "was", "the", "sacrament?"]),
    ("assemblyai", AA_MSGS, ["why", "was", "it"]),
    ("speechmatics", SM_MSGS, ["Why", "was", "the", "sacrament?"])])
def test_into_word_store(provider, msgs, text):
    """the adapter's events go straight into Arc's WordStore: interims replaced by finals, coverage moves on."""
    ws = WordStore()
    a = get(provider)(on_words=ws.add, on_covered=ws.mark_covered)
    for m in msgs:
        a.handle(m)
    assert [w["w"] for w in ws.finals] == text and not ws.interim
    assert ws.covered >= ws.finals[-1]["end"]
    assert a.n_final_msgs >= 1


def test_offset_after_reconnect():
    """after a reconnect the provider clock restarts at 0: words are moved onto the audio clock, and finals the first
    session already gave are not repeated."""
    got = []
    a = DG.Deepgram(on_words=got.extend)
    a.handle(DG_MSGS[1])
    a.retries, a.offset = 1, 1.0                    # the new session starts at 1.0 s (the first had finalised to 2.5 s)
    a.handle({"type": "Results", "start": 0.0, "duration": 2.2, "is_final": True, "channel": {"alternatives": [{"words": [
        {"word": "sacrament", "start": 0.04, "end": 0.52, "punctuated_word": "sacrament"},
        {"word": "of", "start": 1.6, "end": 1.7, "punctuated_word": "of"}]}]}})
    assert [w["text"] for w in got] == ["Why", "was", "the", "sacrament?", "of"]
    assert got[-1]["start"] == 2.6 and a.covered == pytest.approx(3.2)


def test_error_message_raises_and_is_scrubbed():
    a = DG.Deepgram(on_words=lambda ev: None, key="dg_secret_value_123456")
    with pytest.raises(ProviderError) as e:
        a.handle({"type": "Error", "err_code": "INVALID_AUTH", "err_msg": "token dg_secret_value_123456 is not valid"})
    assert "dg_secret_value_123456" not in str(e.value) and not e.value.retry


def test_scrub():
    s = scrub("> Authorization: Token abcdef0123456789\n> Host: api.example.com", [])
    assert "abcdef0123456789" not in s and "Host: api.example.com" in s
    assert "s3cr3tvalue99" not in scrub("Authorization: Bearer s3cr3tvalue99")
    assert scrub("key=zzzzzzzzzzzz here", ["zzzzzzzzzzzz"]).count("zzzz") == 0
    rec = logging.LogRecord("websockets.client", logging.DEBUG, __file__, 1, "> Authorization: %s", ("rawkey1234567",), None)
    os.environ["ARC_TEST_FAKE_KEY"] = "rawkey1234567"
    ScrubFilter(["ARC_TEST_FAKE_KEY"]).filter(rec)
    assert "rawkey1234567" not in rec.getMessage()


def test_registry_and_options():
    assert get("Deepgram") is DG.Deepgram and get("assemblyai") is AA.AssemblyAI and get("speechmatics") is SM.Speechmatics
    with pytest.raises(ValueError):
        get("nope")
    u = DG.Deepgram(on_words=None, options=Options(language="es", endpoint_ms=200)).url()
    assert "model=nova-3" in u and "language=es" in u and "endpointing=200" in u and "interim_results=true" in u
    assert "Token" not in u and "key" not in u.lower()
    u = AA.AssemblyAI(on_words=None, options=Options(language="es")).url()
    assert "speech_model=universal-streaming-multilingual" in u and "sample_rate=16000" in u
    a = AA.AssemblyAI(on_words=None, format_turns=True)
    assert "format_turns=true" in a.url() and "format_turns=true" in a.url()            # stable across reconnects
    s = json.loads(SM.Speechmatics(on_words=None, options=Options(language="de")).start_message())
    assert s["transcription_config"]["language"] == "de" and s["audio_format"] == dict(type="raw", encoding="pcm_s16le", sample_rate=16000)


def test_missing_key():
    from arc_audio.integrations import MissingKey
    old = os.environ.pop("SPEECHMATICS_API_KEY", None)
    try:
        with pytest.raises(MissingKey):
            SM.Speechmatics(on_words=None).key
    finally:
        if old is not None:
            os.environ["SPEECHMATICS_API_KEY"] = old


# ---------------------------------------------------------------- the adapter loop against a mock server
websockets = pytest.importorskip("websockets")
from websockets.asyncio.server import serve  # noqa: E402


class MockDeepgram:
    """a Deepgram-shaped server: one final word per second of audio received; Finalize -> a from_finalize final;
    CloseStream -> Metadata and close. drop_after: close the first connection (1011) after that many frames.
    reject: refuse the handshake with HTTP 401."""

    def __init__(self, drop_after=None, reject=False):
        self.drop_after, self.reject, self.conns, self.auth = drop_after, reject, 0, []

    async def handler(self, ws):
        self.conns += 1
        self.auth.append(ws.request.headers.get("Authorization"))
        n, sent, t = 0, 0, 0.0
        async for m in ws:
            if isinstance(m, bytes):
                n += 1
                t += len(m) / 2 / 16000
                if self.drop_after and self.conns == 1 and n >= self.drop_after:
                    await ws.close(1011, "internal")
                    return
                if int(t) > sent:
                    sent = int(t)
                    await ws.send(json.dumps({"type": "Results", "is_final": True, "start": sent - 1.0, "duration": 1.0,
                                              "channel": {"alternatives": [{"words": [
                                                  {"word": f"w{sent}", "start": sent - 0.6, "end": sent - 0.4,
                                                   "punctuated_word": f"w{sent}", "confidence": 0.9}]}]}}))
                continue
            c = json.loads(m)
            if c["type"] == "Finalize":
                await ws.send(json.dumps({"type": "Results", "is_final": True, "from_finalize": True, "start": float(sent),
                                          "duration": t - sent, "channel": {"alternatives": [{"words": [
                                              {"word": "end", "start": t - 0.3, "end": t - 0.1, "punctuated_word": "end"}]}]}}))
            elif c["type"] == "CloseStream":
                await ws.send(json.dumps({"type": "Metadata", "request_id": "x"}))
                await ws.close()
                return

    async def process_request(self, conn, request):
        if self.reject:
            return conn.respond(401, "unauthorized\n")


async def run_mock(server, seconds=3.0, **opt):
    async with serve(server.handler, "127.0.0.1", 0, process_request=server.process_request) as srv:
        port = srv.sockets[0].getsockname()[1]
        got, cov, errs = [], [], []
        a = DG.Deepgram(on_words=got.extend, on_covered=cov.append, on_error=errs.append, key="test-key-not-real",
                        options=Options(backoff_s=0.01, final_timeout_s=10.0, stall_timeout_s=None, **opt),
                        url=f"ws://127.0.0.1:{port}/v1/listen")                  # no stall check: CI machines can be slow
        async with a:
            pcm = np.zeros(1600, np.float32)
            for _ in range(int(seconds * 10)):
                await a.send(pcm)
                await asyncio.sleep(0.01)                                      # 10 x real time
            got_final = await a.finalize()
            await a.close()
        return a, got, cov, errs, got_final


def test_mock_session_finalize_and_close():
    srv = MockDeepgram()
    a, got, cov, errs, fin = asyncio.run(run_mock(srv))
    assert fin and not errs and a.failed is None and a.retries == 0
    assert [w["text"] for w in got] == ["w1", "w2", "w3", "end"]
    assert srv.auth == ["Token test-key-not-real"]
    assert cov[-1] >= 3.0                                                      # end of input: everything covered


def test_mock_reconnect_resends_on_the_same_clock():
    srv = MockDeepgram(drop_after=15)                                          # the first connection dies at 1.5 s
    a, got, cov, errs, fin = asyncio.run(run_mock(srv, seconds=4.0))
    assert srv.conns == 2 and a.retries == 1 and not errs
    T = [w["text"] for w in got]
    assert T[0] == "w1" and T[-1] == "end" and len(T) == len(set((w["text"], w["start"]) for w in got))
    st = [w["start"] for w in got]
    assert st == sorted(st) and st[-2] < 4.0                                   # no word twice, on the audio clock


def test_mock_bad_key_gives_up_without_retrying():
    srv = MockDeepgram(reject=True)
    a, got, cov, errs, fin = asyncio.run(run_mock(srv))
    assert len(errs) == 1 and a.failed is not None and a.retries == 0 and srv.conns == 0
    assert "test-key-not-real" not in str(errs[0]) and not got


# ---------------------------------------------------------------- the session fallback (a stand-in for Arc)
class FakeStream:
    def __init__(self, mode):
        self.mode, self.words, self.n, self.closed = mode, WordStore(), 0, False

    def add_words(self, ev, spans=None):
        return self.words.add(ev, "external")

    def feed(self, pcm):
        t0 = self.n / 16000
        self.n += len(pcm)
        if self.mode == "builtin":                                             # a word per chunk: what Arc "heard"
            return dict(transcript=[dict(start=t0, end=t0 + 0.05, text="arc", words=[dict(text="arc", start=t0, end=t0 + 0.05)])])
        return {}

    def flush(self):
        out = dict(transcript=[dict(start=w["start"], end=w["end"], text=w["w"]) for w in self.words.finals])
        return out

    def end_turn(self):
        return {}

    def close(self):
        self.closed = True


class FakeArc:
    def __init__(self):
        self.streams = []

    def stream(self, transcript="builtin", **kw):
        s = FakeStream(transcript)
        self.streams.append(s)
        return s


def test_session_falls_back_to_arc_decoder():
    from arc_audio.integrations import Session

    async def go():
        srv = MockDeepgram(reject=True)
        async with serve(srv.handler, "127.0.0.1", 0, process_request=srv.process_request) as ws_srv:
            port = ws_srv.sockets[0].getsockname()[1]
            arc = FakeArc()
            os.environ["DEEPGRAM_API_KEY"] = os.environ.get("DEEPGRAM_API_KEY") or "test-key-not-real"
            async with Session(arc, "deepgram", url=f"ws://127.0.0.1:{port}/v1/listen", backoff_s=0.01) as s:
                for _ in range(5):
                    await s.feed(np.zeros(1600, np.float32))
                return await s.finish(), arc
    out, arc = asyncio.run(go())
    assert out["asr"]["status"] == "fallback" and out["asr"]["fallback_from"] == 0.0
    assert [s.mode for s in arc.streams] == ["external", "builtin"]
    assert len(out["words"]) == 5 and all(w["source"] == "arc" for w in out["words"])     # one per chunk, none twice
    assert arc.streams[1].n == 5 * 1600


def test_session_falls_back_mid_stream_from_the_first_unfinalised_audio():
    """the provider finalises the first second, then the connection dies for good: Arc's decoder takes the audio from
    there (no chunk twice), and the provider's words stay."""
    from arc_audio.integrations import Session

    async def go():
        srv = MockDeepgram(drop_after=15)
        async with serve(srv.handler, "127.0.0.1", 0, process_request=srv.process_request) as ws_srv:
            port = ws_srv.sockets[0].getsockname()[1]
            arc = FakeArc()
            os.environ["DEEPGRAM_API_KEY"] = os.environ.get("DEEPGRAM_API_KEY") or "test-key-not-real"
            async with Session(arc, "deepgram", url=f"ws://127.0.0.1:{port}/v1/listen", backoff_s=0.01, max_retries=0,
                               stall_timeout_s=None) as s:
                for _ in range(30):
                    await s.feed(np.zeros(1600, np.float32))
                    await asyncio.sleep(0.01)
                return await s.finish(), arc
    out, arc = asyncio.run(go())
    a = out["asr"]
    assert a["status"] == "fallback" and a["fallback_from"] == pytest.approx(1.0)
    src = [w["source"] for w in out["words"]]
    assert src[0] == "external" and "arc" in src and out["words"][0]["text"] == "w1"
    assert arc.streams[1].n == 20 * 1600                                     # the audio from 1.0 s on, once
    starts = [w["start"] for w in out["words"] if w["source"] == "arc"]
    assert starts == sorted(starts) and len(starts) == len(set(starts)) and starts[0] >= 0.0


# ---------------------------------------------------------------- live (opt-in)
LIVE = [p for p in os.environ.get("ARC_LIVE_ASR", "").split(",") if p]


@pytest.mark.skipif(not LIVE, reason="set ARC_LIVE_ASR=deepgram,assemblyai,speechmatics (and the keys) for the live test")
@pytest.mark.parametrize("provider", LIVE or ["none"])
def test_live_provider(provider):
    """a short fictional TTS line (examples/example_synth.wav, first 6 s) at real-time pace: words come back, final,
    with times inside the audio, and the last final within 3 s of the end of the audio."""
    import time
    from arc_audio.audio import load_audio
    wav = load_audio(os.path.join(os.path.dirname(__file__), "..", "examples", "example_synth.wav"))[:6 * 16000]

    async def go():
        got = []
        a = get(provider)(on_words=got.extend)
        async with a:
            t0 = time.monotonic()
            for k, i in enumerate(range(0, len(wav), 1600)):
                await a.send(wav[i:i + 1600])
                await asyncio.sleep(max(0, t0 + (k + 1) * 0.1 - time.monotonic()))
            t_end = time.monotonic()
            await a.finalize()
            await a.close()
        return a, got, t_end
    a, got, t_end = asyncio.run(go())
    fin = [w for w in got if w["is_final"]]
    assert a.failed is None and len(fin) >= 5
    assert all(0 <= w["start"] <= w["end"] <= 6.5 for w in fin)
    assert a.last_final_rx is not None and a.last_final_rx - t_end < 3.0
