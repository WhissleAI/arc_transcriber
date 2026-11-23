"""arc_transcriber.serve: routes, request parsing and response shapes, on CPU with a stand-in model (no weights)."""
import base64, copy, io, json, os, threading, urllib.error, urllib.request, wave

import numpy as np
import pytest

from arc_transcriber import serve

HERE = os.path.dirname(__file__)
GOLDEN = json.load(open(os.path.join(HERE, "golden", "mini.json")))["output"]


class FakeArc:
    dev = "cpu"

    def __init__(self):
        self.calls = []

    def predict(self, wav, **kw):
        self.calls.append(dict(kw, n=len(wav)))
        out = copy.deepcopy(GOLDEN)
        out["diarization"] = [{"speaker": "S1", "start": 0.0, "end": 40.0}]
        return out


def wav_bytes(seconds=1.0, rate=16000):
    t = np.arange(int(seconds * rate)) / rate
    pcm = (0.1 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2").tobytes()
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)
    return b.getvalue()


@pytest.fixture(scope="module")
def server():
    svc = serve.Service("arc-transcriber", max_mb=1, queue_timeout=5)
    svc.arc = FakeArc()
    httpd = serve.ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(svc, access_log=False))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", svc
    httpd.shutdown()


def call(url, path, body=None, ctype=None, headers=None, method=None):
    req = urllib.request.Request(url + path, data=body, method=method or ("POST" if body is not None else "GET"))
    if ctype:
        req.add_header("Content-Type", ctype)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.headers.get("Content-Type"), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read().decode()


def multipart(fields, file_bytes):
    bnd = "arcboundary"
    parts = [f'--{bnd}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
    parts.append(f'--{bnd}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
                 + file_bytes + b"\r\n")
    return b"".join(parts) + f"--{bnd}--\r\n".encode(), f"multipart/form-data; boundary={bnd}"


def test_health_and_models(server):
    url, svc = server
    for p in ("/health", "/v1/health/live", "/v1/health/ready", "/ping"):
        assert call(url, p)[0] == 200, p
    st, _, body = call(url, "/v1/models")
    m = json.loads(body)
    assert st == 200 and m["object"] == "list" and m["data"][0]["id"] == "arc-transcriber"
    assert call(url, "/v1/models/arc-transcriber")[0] == 200
    assert call(url, "/v1/models/other")[0] == 404
    assert json.loads(call(url, "/execution-parameters")[2])["MaxConcurrentTransforms"] == 1
    assert json.loads(call(url, "/nope")[2])["error"]["code"] == 404


def test_not_ready_until_loaded():
    svc = serve.Service("m")
    assert svc.arc is None
    with pytest.raises(serve.ApiError) as e:
        svc.transcribe(np.zeros(160, np.float32), {})
    assert e.value.status == 503
    svc.load(lambda: (_ for _ in ()).throw(RuntimeError("no weights")))
    assert svc.error and "no weights" in svc.error


def test_json_and_text(server):
    url, svc = server
    body, ct = multipart({"domain": "nba", "names": '["Theo Brandt", "Marcus Delaney"]'}, wav_bytes())
    st, _, out = call(url, "/v1/audio/transcriptions", body, ct)
    j = json.loads(out)
    assert st == 200 and j["text"].startswith("Welcome back, folks.") and j["model"] == "arc-transcriber"
    assert j["usage"] == {"type": "duration", "seconds": 1} and "wall_ms" not in j
    kw = svc.arc.calls[-1]
    assert kw["domain"] == "nba" and kw["names"] == ["Theo Brandt", "Marcus Delaney"] and kw["n"] == 16000
    st, ctype, out = call(url, "/v1/audio/transcriptions?response_format=text", wav_bytes(), "audio/wav")
    assert st == 200 and ctype.startswith("text/plain") and out.endswith("\n")


def test_verbose_json_shape(server):
    url, _ = server
    st, _, out = call(url, "/v1/audio/transcriptions?response_format=verbose_json", wav_bytes(), "audio/wav")
    j = json.loads(out)
    assert st == 200 and j["task"] == "transcribe" and j["language"] == "english" and j["duration"] == 1.0
    seg = j["segments"][1]
    assert seg["id"] == 1 and seg["speaker"] == "S1" and seg["device_directed"] == 0.0001
    assert seg["tags"][0] == {"tag": "ENTITY_PERSON", "text": "Marcus Delaney", "word_begin": 0, "word_end": 1}
    assert seg["words"][0] == {"word": "Marcus", "start": 2.27, "end": 2.85, "tag": "ENTITY_PERSON", "tag_begin": True}
    assert all(w["segment"] is not None and "word" in w for w in j["words"])
    assert j["names"][0]["entity"] == "Marcus Delaney" and j["numbers"] and j["events"][0]["label"]
    assert "speakers" in j and j["diarization"] and "wall_ms" in j


def test_arc_json_srt_vtt(server):
    url, _ = server
    j = json.loads(call(url, "/v1/audio/transcriptions?response_format=arc_json&domain=nba", wav_bytes(), "audio/wav")[2])
    assert j["domain"] == "nba" and j["transcript"] and "frames" not in j
    st, ct, srt = call(url, "/v1/audio/transcriptions?response_format=srt", wav_bytes(), "audio/wav")
    assert st == 200 and ct.startswith("application/x-subrip") and srt.startswith("1\n00:00:00,600 --> ")
    st, ct, vtt = call(url, "/v1/audio/transcriptions?response_format=vtt", wav_bytes(), "audio/wav")
    assert ct.startswith("text/vtt") and vtt.startswith("WEBVTT")


def test_raw_pcm_and_resample(server):
    url, svc = server
    pcm = np.zeros(8000, "<i2").tobytes()      # 1 s at 8 kHz
    assert call(url, "/v1/audio/transcriptions?sample_rate=8000", pcm, "audio/pcm")[0] == 200
    assert svc.arc.calls[-1]["n"] == 16000
    assert call(url, "/v1/audio/transcriptions", wav_bytes(rate=44100), "audio/wav")[0] == 200
    assert svc.arc.calls[-1]["n"] == 16000


def test_json_body_base64(server):
    url, _ = server
    a = base64.b64encode(wav_bytes()).decode()
    st, _, out = call(url, "/v1/audio/transcriptions", json.dumps({"audio": "data:audio/wav;base64," + a, "response_format": "text"}).encode(),
                      "application/json")
    assert st == 200 and out.startswith("Welcome back")


def test_errors(server):
    url, _ = server
    for q, body, ct, code in [("?response_format=xml", wav_bytes(), "audio/wav", 400),
                              ("?language=fr", wav_bytes(), "audio/wav", 400),
                              ("?diarization=bogus", wav_bytes(), "audio/wav", 400),
                              ("", b"not audio", "audio/mpeg", 415),
                              ("", b"", "audio/wav", 400),
                              ("", b"x" * (2 * 1024 * 1024), "audio/wav", 413)]:
        st, _, out = call(url, "/v1/audio/transcriptions" + q, body, ct)
        assert st == code, (q, ct, out)
        assert json.loads(out)["error"]["code"] == code
    st, _, out = call(url, "/v1/audio/transcriptions", json.dumps({"audio": "https://example.com/a.wav"}).encode(), "application/json")
    assert st == 400


def test_sagemaker_invocations(server):
    url, svc = server
    st, _, out = call(url, "/invocations", wav_bytes(), "audio/wav", {"X-Amzn-SageMaker-Custom-Attributes": "domain=nhl;response_format=json"})
    assert st == 200 and set(json.loads(out)) == {"text", "model", "usage"} and svc.arc.calls[-1]["domain"] == "nhl"
    st, _, out = call(url, "/invocations", wav_bytes(), "audio/wav")
    assert json.loads(out)["task"] == "transcribe"                       # default verbose_json
    st, _, out = call(url, "/invocations", wav_bytes(), "audio/wav", {"Accept": "text/plain"})
    assert out.startswith("Welcome back")
    a = base64.b64encode(wav_bytes()).decode()
    st, _, out = call(url, "/invocations", json.dumps({"instances": [{"audio": a, "response_format": "text"}, {"audio": ""}]}).encode(),
                      "application/json")
    j = json.loads(out)
    assert st == 200 and j["model"] == "arc-transcriber" and j["results"][0].startswith("Welcome") and j["results"][1]["error"]["code"] == 400
    lines = (json.dumps({"audio": a}) + "\n" + json.dumps({"audio": a, "response_format": "json"}) + "\n").encode()
    st, ct, out = call(url, "/invocations", lines, "application/jsonlines")
    rows = [json.loads(x) for x in out.splitlines()]
    assert st == 200 and len(rows) == 2 and rows[0]["task"] == "transcribe" and "task" not in rows[1]
    assert call(url, "/invocations", b"hello", "text/plain")[0] == 415


def test_options_reach_predict(server):
    url, svc = server
    body, ct = multipart({"events": "false", "diarization": "voice", "terms": "Hawks, Harbor City", "topic": "basketball",
                          "link_names": "false", "numbers": "0"}, wav_bytes())
    j = json.loads(call(url, "/v1/audio/transcriptions?response_format=verbose_json", body, ct)[2])
    kw = svc.arc.calls[-1]
    assert kw["events"] is None and kw["diarization"] == "voice" and kw["context"]["kb"] == ["Hawks", "Harbor City"]
    assert kw["context"]["dom"] == "basketball" and j["names"] == [] and j["numbers"] == []
