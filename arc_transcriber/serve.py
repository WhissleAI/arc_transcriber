"""arc-serve-py: Arc (arc_audio) behind the batch routes of deploy/openapi.yaml, for a GPU (or CPU) host.

    python -m arc_transcriber.serve --backbone full --device cuda          # Arc Transcriber (full) on an NVIDIA GPU
    python -m arc_transcriber.serve --backbone mini --device cpu --port 8081

Routes (same request and response shapes as the C server, deploy/server): POST /v1/audio/transcriptions (json, text,
verbose_json, srt, vtt, arc_json), GET /health, /v1/health/live, /v1/health/ready, /v1/models[/{id}], and SageMaker's
GET /ping, POST /invocations, GET /execution-parameters. Not here: the realtime WebSocket, Arc Tagger (/v1/tags) and the
Snowflake rows route. One model, one request at a time (a GPU serialises the work anyway); the rest queue.

Environment (flags win): HOST, PORT (8080), ARC_MODEL or ARC_MODEL_DIR (Hugging Face repo or local folder), ARC_REVISION,
ARC_BACKBONE (full), ARC_DEVICE (auto), ARC_THREADS, ARC_MODEL_ID, ARC_DOMAIN (unknown), ARC_MAX_BODY_MB (100),
ARC_QUEUE_TIMEOUT_S (120), ARC_ACCESS_LOG (1). Options the C server honours but this one does not: one_utterance, beams
(accepted, ignored); words / trunk / link_names / numbers = false drop those fields but do not save time.
--fetch-only downloads the weights (both Arc and, for full, the Parakeet encoder) into the Hugging Face cache and exits.
Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0 with the Whissle terms; see LICENSE.
"""
import argparse, base64, io, json, math, os, sys, threading, time
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, urlsplit

from . import DEFAULT_MODEL, DEFAULT_REVISION, __version__

SR = 16000
FORMATS = ("json", "text", "verbose_json", "srt", "vtt", "arc_json")
DIARIZATION = ("auto", "slots", "fused", "voice")
ARC_JSON_KEYS = ("transcript", "transcript_written", "names", "numbers", "audio_events", "speakers", "diarization")


class ApiError(Exception):
    def __init__(self, status, message, kind="invalid_request_error"):
        super().__init__(message)
        self.status, self.kind = status, kind


def _bool(v, default):
    if v is None or v == "":
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() not in ("0", "false", "no", "off")


def _list(v):
    """a JSON array, or items separated by newlines / commas"""
    if v is None or v == "":
        return None
    if isinstance(v, list):
        return v
    s = str(v).strip()
    if s.startswith("["):
        try:
            return json.loads(s)
        except ValueError:
            raise ApiError(400, "names / terms: invalid JSON array")
    sep = "\n" if "\n" in s else ","
    return [x.strip() for x in s.split(sep) if x.strip()]


def _r(x, d=3):
    return None if x is None else round(float(x), d)


def _jsonable(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


# ------------------------------------------------------------------ audio
def decode_audio(body, ctype, sample_rate=None):
    """request body -> float32 16 kHz mono. raw PCM16 for audio/pcm (or audio/l16); anything soundfile reads otherwise."""
    import numpy as np
    from arc_audio.audio import load_audio
    if not body:
        raise ApiError(400, "empty audio")
    ct = (ctype or "").split(";")[0].strip().lower()
    if ct in ("audio/pcm", "audio/l16", "audio/x-raw"):
        if len(body) % 2:
            raise ApiError(400, "raw PCM16 body has an odd number of bytes")
        x = np.frombuffer(body, "<i2").astype(np.float32) / 32768.0
        return load_audio(x, int(sample_rate or SR))
    try:
        import soundfile as sf
        x, sr = sf.read(io.BytesIO(body), dtype="float32", always_2d=False)
    except Exception:
        raise ApiError(415, "unsupported audio: send WAV / FLAC / Ogg (anything libsndfile reads) or raw PCM16 (audio/pcm)")
    return load_audio(x, sr)


def b64_audio(s):
    if not isinstance(s, str) or not s:
        raise ApiError(400, "audio: base64 string or data: URI expected")
    if s.startswith(("http://", "https://")):
        raise ApiError(400, "audio URLs are refused (the server makes no network calls)")
    ctype = None
    if s.startswith("data:"):
        head, _, s = s.partition(",")
        ctype = head[5:].split(";")[0] or None
    try:
        return base64.b64decode(s, validate=False), ctype
    except Exception:
        raise ApiError(400, "audio: invalid base64")


def parse_multipart(body, ctype):
    """multipart/form-data -> (fields {name: str}, file (bytes, content_type) or None)"""
    msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + ctype.encode("latin-1") + b"\r\n\r\n" + body)
    if not msg.is_multipart():
        raise ApiError(400, "invalid multipart body")
    fields, file = {}, None
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        data = part.get_payload(decode=True) or b""
        if name in ("file", "audio") and (part.get_filename() is not None or name == "file"):
            file = (data, part.get_content_type())
        elif name:
            name = name[:-2] if name.endswith("[]") else name
            val = data.decode("utf-8", "replace")
            if name in fields and name in ("names", "terms"):
                prev = fields[name] if isinstance(fields[name], list) else [fields[name]]
                fields[name] = prev + [val]
            else:
                fields[name] = val
    return fields, file


# ------------------------------------------------------------------ responses (the C server's shapes)
def _label(v):
    if isinstance(v, dict) and v.get("label") is not None:
        return {"label": v["label"], "p": _r(v.get("p"), 4)}
    if isinstance(v, str):
        return {"label": v, "p": None}
    return None


def _p(v):
    if isinstance(v, dict):
        v = v.get("p")
    return _r(v, 4)


def _seg_tags(u):
    ws, out = u.get("words") or [], []
    for t in u.get("tags") or []:
        idx = [k for k, w in enumerate(ws) if t.get("start") is not None and
               t["start"] - 0.05 <= (w["start"] + w["end"]) / 2 <= t["end"] + 0.05]
        out.append({"tag": t.get("tag"), "text": t.get("text") or "",
                    "word_begin": idx[0] if idx else -1, "word_end": idx[-1] if idx else -1})
    return out


def _word(w, key="word", seg=None):
    o = {key: w.get("text", w.get("w")), "start": _r(w["start"]), "end": _r(w["end"])}
    if w.get("tag"):
        o["tag"] = w["tag"]
        if seg is None:
            o["tag_begin"] = bool(w.get("tag_begin", False))
    if seg is not None:
        o["segment"] = seg
    return o


def full_text(out):
    return " ".join(u["text"] for u in out.get("transcript") or [] if u.get("text"))


def render_response(out, fmt, model_id, audio_s, wall_ms, domain):
    """an arc_audio predict() result -> (body str, content type), as deploy/server/main.c renders it"""
    from .output import lines_of, render
    if fmt == "text":
        return full_text(out) + "\n", "text/plain; charset=utf-8"
    if fmt in ("srt", "vtt"):
        return render(out, fmt), ("text/vtt" if fmt == "vtt" else "application/x-subrip") + "; charset=utf-8"
    if fmt == "arc_json":
        o = {"domain": domain}
        o.update({k: out[k] for k in ARC_JSON_KEYS if out.get(k) is not None})
    else:
        o = {}
        if fmt == "verbose_json":
            o.update(task="transcribe", language="english", duration=_r(audio_s))
        o["text"] = full_text(out)
        if fmt == "verbose_json":
            tr = out.get("transcript") or []
            spk = None
            if out.get("diarization") is not None or out.get("speakers") is not None:
                lines = lines_of(out) if tr and "words" in tr[0] else []
                spk = [ln["speaker"] for ln in lines] if len(lines) == len(tr) else [None] * len(tr)
            segs, words = [], []
            for i, u in enumerate(tr):
                s = {"id": i, "start": _r(u["start"]), "end": _r(u["end"]), "text": u.get("text") or "", "tags": _seg_tags(u),
                     "intent": _label(u.get("intent")), "act": _label(u.get("act")),
                     "device_directed": _p(u.get("device_directed")), "command_complete": _p(u.get("command_complete")),
                     "directed_head": _p(u.get("directed_head")), "words": [_word(w) for w in u.get("words") or []]}
                alts = u.get("alts") or []
                if len(alts) > 1:
                    s["alternatives"] = [{"text": a.get("text"), "score": _r(a.get("p", a.get("score")), 4)} for a in alts]
                if spk is not None:
                    s["speaker"] = spk[i]
                segs.append(s)
                words += [_word(w, seg=i) for w in u.get("words") or []]
            o.update(segments=segs, words=words, names=out.get("names") or [], numbers=out.get("numbers") or [],
                     events=[{"label": e.get("label"), "start": _r(e.get("start")), "end": _r(e.get("end")), "p": _r(e.get("p"), 4)}
                             for e in out.get("audio_events") or []])
            for k in ("speakers", "diarization"):
                if out.get(k) is not None:
                    o[k] = out[k]
    o["model"] = model_id
    o["usage"] = {"type": "duration", "seconds": math.ceil(audio_s)}
    if fmt != "json":
        o["wall_ms"] = round(wall_ms, 1)
    return json.dumps(o, default=_jsonable, separators=(",", ":")), "application/json"


# ------------------------------------------------------------------ the service
class Service:
    """one loaded Arc model; transcribe() serialises requests on it"""

    def __init__(self, model_id, domain="unknown", max_mb=100, queue_timeout=120.0, backbone="full"):
        self.arc, self.error, self.model_id, self.domain = None, None, model_id, domain
        self.max_bytes, self.queue_timeout, self.backbone = int(max_mb * 1024 * 1024), queue_timeout, backbone
        self.lock, self.started = threading.Lock(), int(time.time())

    def load(self, loader):
        try:
            self.arc = loader()
        except Exception as e:  # noqa: BLE001  (readiness stays 503 and says why)
            self.error = f"{type(e).__name__}: {e}"
            print(f"arc-serve-py: model load failed: {self.error}", file=sys.stderr, flush=True)

    @property
    def device(self):
        return str(getattr(self.arc, "dev", "")) or None

    def transcribe(self, wav, opts, default_format="json"):
        fmt = opts.get("response_format") or default_format
        if fmt not in FORMATS:
            raise ApiError(400, "response_format: " + " | ".join(FORMATS))
        lang = opts.get("language")
        if lang and not str(lang).lower().startswith("en"):
            raise ApiError(400, "Arc transcribes English only (language=en)")
        diar = opts.get("diarization") or "auto"
        if diar not in DIARIZATION:
            raise ApiError(400, "diarization: " + " | ".join(DIARIZATION))
        domain = opts.get("domain") or self.domain
        names = _list(opts.get("names"))
        terms = _list(opts.get("terms"))
        ctx = None
        if terms or opts.get("topic") or opts.get("prompt"):
            from arc_audio.api import context_dict
            ctx = context_dict(terms=terms, topic=opts.get("topic"), previous_text=opts.get("prompt"))
        trunk = _bool(opts.get("trunk"), True)
        if self.arc is None:
            raise ApiError(503, "the model is still loading" if not self.error else "the model failed to load", "server_busy")
        if not self.lock.acquire(timeout=self.queue_timeout):
            raise ApiError(503, "the transcription worker is busy; retry", "server_busy")
        try:
            t0 = time.perf_counter()
            out = self.arc.predict(wav, domain=domain, names=names, diarization=diar, context=ctx, return_frames=False,
                                   events="default" if _bool(opts.get("events"), trunk) else None)
            wall = (time.perf_counter() - t0) * 1000
        except ApiError:
            raise
        except ValueError as e:
            raise ApiError(400, str(e))
        except Exception as e:  # noqa: BLE001
            raise ApiError(500, f"transcription failed: {type(e).__name__}: {e}", "server_error")
        finally:
            self.lock.release()
        if not _bool(opts.get("link_names"), trunk):
            out["names"] = []
        if not _bool(opts.get("numbers"), trunk):
            out["numbers"] = []
        if not _bool(opts.get("words"), True):
            for u in out.get("transcript") or []:
                u["words"] = []
        return render_response(out, fmt, self.model_id, len(wav) / SR, wall, domain)

    def model_card(self):
        return {"id": self.model_id, "object": "model", "created": self.started, "owned_by": "whissle",
                "capabilities": ["transcriptions"], "backbone": self.backbone, "device": self.device,
                "server": "arc_transcriber.serve", "server_version": __version__}


def make_handler(svc, access_log=True):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"arc-serve-py/{__version__}"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):
            if access_log:
                sys.stderr.write("%s %s\n" % (self.address_string(), fmt % a))

        def send(self, status, body, ctype="application/json"):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def send_json(self, status, obj):
            self.send(status, json.dumps(obj, default=_jsonable))

        def error(self, e):
            self.send_json(e.status, {"error": {"message": str(e), "type": e.kind, "code": e.status}})

        def body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > svc.max_bytes:
                while n > 0:                     # drain (discarded) so the client reads the 413 instead of a reset
                    k = len(self.rfile.read(min(n, 1 << 16)))
                    if not k:
                        break
                    n -= k
                raise ApiError(413, f"payload larger than {svc.max_bytes // (1024 * 1024)} MB")
            return self.rfile.read(n) if n else b""

        def ready(self):
            ok = svc.arc is not None
            st = {"status": "ready" if ok else ("error" if svc.error else "loading")}
            if svc.error:
                st["error"] = svc.error
            self.send_json(200 if ok else 503, st)

        def do_GET(self):
            path = urlsplit(self.path).path.rstrip("/") or "/"
            if path in ("/health", "/v1/health/live"):
                return self.send_json(200, {"status": "ok"})
            if path in ("/v1/health/ready", "/ping"):
                return self.ready()
            if path == "/v1/models":
                return self.send_json(200, {"object": "list", "data": [svc.model_card()]})
            if path.startswith("/v1/models/"):
                if path[len("/v1/models/"):] == svc.model_id:
                    return self.send_json(200, svc.model_card())
                return self.error(ApiError(404, "model not found"))
            if path == "/execution-parameters":
                return self.send_json(200, {"MaxConcurrentTransforms": 1, "BatchStrategy": "SINGLE_RECORD",
                                            "MaxPayloadInMB": max(1, svc.max_bytes // (1024 * 1024))})
            self.error(ApiError(404, f"no route {path}"))

        do_HEAD = do_GET

        def do_POST(self):
            u = urlsplit(self.path)
            path = u.path.rstrip("/")
            try:
                query = dict(parse_qsl(u.query))
                if path == "/v1/audio/transcriptions":
                    return self.transcriptions(query, "json")
                if path == "/invocations":
                    return self.invocations(query)
                raise ApiError(404, f"no route {path}")
            except ApiError as e:
                self.error(e)
            except Exception as e:  # noqa: BLE001
                self.log_message("error: %s: %s", type(e).__name__, e)
                self.error(ApiError(500, f"{type(e).__name__}: {e}", "server_error"))

        def request_audio(self, query, body, default_format):
            ct = (self.headers.get("Content-Type") or "application/octet-stream")
            base = ct.split(";")[0].strip().lower()
            opts = dict(query)
            if base == "multipart/form-data":
                fields, file = parse_multipart(body, ct)
                opts.update(fields)
                if not file:
                    raise ApiError(400, "multipart: a 'file' part is required")
                data, fct = file
                wav = decode_audio(data, fct, opts.get("sample_rate"))
            elif base == "application/json":
                try:
                    j = json.loads(body or b"{}")
                except ValueError:
                    raise ApiError(400, "invalid JSON body")
                return None, j
            else:
                wav = decode_audio(body, ct, opts.get("sample_rate"))
            return svc.transcribe(wav, opts, default_format), None

        def json_item(self, item, query, default_format):
            if not isinstance(item, dict):
                raise ApiError(400, "each request is a JSON object with 'audio'")
            data, ct = b64_audio(item.get("audio") or item.get("file"))
            opts = dict(query, **{k: v for k, v in item.items() if k not in ("audio", "file")})
            return svc.transcribe(decode_audio(data, item.get("content_type") or ct, opts.get("sample_rate")), opts, default_format)

        def transcriptions(self, query, default_format):
            res, j = self.request_audio(query, self.body(), default_format)
            if j is not None:
                res = self.json_item(j, query, default_format)
            self.send(200, *res)

        def invocations(self, query):
            attrs = self.headers.get("X-Amzn-SageMaker-Custom-Attributes") or ""
            for kv in attrs.split(";"):
                k, _, v = kv.partition("=")
                if k.strip() and v:
                    query.setdefault(k.strip(), v.strip())
            accept = self.headers.get("Accept") or ""
            default = "text" if accept.lower().startswith("text/plain") else "verbose_json"
            ct = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            body = self.body()
            if ct in ("application/jsonlines", "application/x-jsonlines", "application/jsonl"):
                lines = []
                for line in body.decode("utf-8").splitlines():
                    if line.strip():
                        try:
                            lines.append(self.json_item(json.loads(line), query, "verbose_json")[0])
                        except ApiError as e:
                            lines.append(json.dumps({"error": {"message": str(e), "type": e.kind, "code": e.status}}))
                        except ValueError:
                            lines.append(json.dumps({"error": {"message": "invalid JSON line", "type": "invalid_request_error", "code": 400}}))
                return self.send(200, "\n".join(lines) + "\n", "application/jsonlines")
            if ct == "application/json":
                try:
                    j = json.loads(body or b"{}")
                except ValueError:
                    raise ApiError(400, "invalid JSON body")
                batch = j if isinstance(j, list) else (j.get("instances") or j.get("inputs")) if isinstance(j, dict) else None
                if isinstance(batch, list):
                    results = []
                    for item in batch:
                        try:
                            body, ctype = self.json_item(item, query, "verbose_json")
                            results.append(json.loads(body) if ctype == "application/json" else body)
                        except ApiError as e:
                            results.append({"error": {"message": str(e), "type": e.kind, "code": e.status}})
                    return self.send_json(200, {"model": svc.model_id, "results": results})
                return self.send(200, *self.json_item(j, query, default))
            if ct and not (ct.startswith("audio/") or ct in ("application/octet-stream", "multipart/form-data")):
                raise ApiError(415, f"unsupported Content-Type {ct}")
            res, _ = self.request_audio(query, body, default)
            self.send(200, *res)

    return Handler


def fetch_weights(model, revision, backbone):
    """download Arc's weights (and, for full, the Parakeet encoder it names) into the Hugging Face cache"""
    from huggingface_hub import snapshot_download
    from arc_audio import Arc
    from arc_audio.server import fetch_backbone
    sub = Arc.BACKBONES[backbone]
    subs = [sub, Arc.LEGACY[sub]]
    path = model if os.path.isdir(model) else snapshot_download(
        model, revision=revision, allow_patterns=[f"{d}/{p}" for d in subs for p in Arc.MODEL_FILES] + ["LICENSE", "NOTICE"])
    for d in subs:
        cfg = os.path.join(path, d, "config.json")
        if os.path.exists(cfg):
            bb = json.load(open(cfg)).get("backbone") or {}
            if bb.get("type") == "parakeet":
                print("encoder:", fetch_backbone(bb), flush=True)
            print("arc:", os.path.dirname(cfg), flush=True)
            return
    raise SystemExit(f"no {sub}/config.json under {path}")


def main(argv=None):
    env = os.environ.get
    ap = argparse.ArgumentParser(prog="python -m arc_transcriber.serve", description=__doc__.split("\n")[0])
    ap.add_argument("--host", default=env("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(env("PORT", "8080")))
    ap.add_argument("--model", default=env("ARC_MODEL") or env("ARC_MODEL_DIR") or DEFAULT_MODEL, help="Hugging Face repo or local folder")
    ap.add_argument("--revision", default=env("ARC_REVISION", DEFAULT_REVISION))
    ap.add_argument("--backbone", default=env("ARC_BACKBONE", "full"), help="full (Arc Transcriber) or mini (Arc Mini)")
    ap.add_argument("--device", default=env("ARC_DEVICE", "auto"), help="cuda | cpu | mps | auto (cuda > mps > cpu)")
    ap.add_argument("--threads", type=int, default=int(env("ARC_THREADS", "0")) or None, help="CPU threads")
    ap.add_argument("--model-id", default=env("ARC_MODEL_ID"))
    ap.add_argument("--domain", default=env("ARC_DOMAIN", "unknown"))
    ap.add_argument("--max-mb", type=float, default=float(env("ARC_MAX_BODY_MB", "100")))
    ap.add_argument("--queue-timeout", type=float, default=float(env("ARC_QUEUE_TIMEOUT_S", "120")))
    ap.add_argument("--fetch-only", action="store_true", help="download the weights and exit")
    a = ap.parse_args(argv)
    rev = None if os.path.isdir(a.model) else a.revision
    if a.fetch_only:
        return fetch_weights(a.model, rev, a.backbone)
    model_id = a.model_id or ("arc-transcriber" if a.backbone in ("full", "parakeet", "server") else "arc-mini")
    svc = Service(model_id, a.domain, a.max_mb, a.queue_timeout, a.backbone)
    from . import load
    threading.Thread(target=svc.load, args=(lambda: load(a.model, rev, a.device, a.threads, a.backbone),), daemon=True).start()
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(svc, _bool(env("ARC_ACCESS_LOG"), True)))
    print(f"arc-serve-py {__version__}: {model_id} ({a.backbone}, {a.device}) on http://{a.host}:{a.port}; loading the model",
          file=sys.stderr, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
