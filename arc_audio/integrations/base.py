"""Common base for cloud streaming recognisers whose words feed Arc's EXTERNAL stream mode.

An adapter streams 16 kHz mono PCM16 over the provider's websocket and turns its partial / final results into Arc's
external-word events:

    {"text": str, "start": s, "end": s, "is_final": bool, "confidence": float | None}

with times in seconds on the audio clock (0 = the first sample sent), and reports how far the recogniser has finalised
("covered", Arc's `mark_covered` / `arc_stream_covered`), so Arc's utterances get their FINAL as soon as the recogniser is
done with them.

A provider subclass supplies four things: the websocket URL and headers, the start message (if any), the control
messages (finalise, end of stream) and `parse(message)`, a pure function from one server message to a `Parsed` result
(so it can be tested with recorded messages, no network). The base class does the rest: connect with a timeout,
send audio, reconnect with exponential backoff (re-sending the audio the recogniser had not finalised, on the same
clock), a stall timeout, finalise and close with a timeout, and an `on_error` callback once it gives up (the session
then falls back to Arc's own decoder).

API keys come from environment variables only. They go in the websocket handshake headers and nowhere else: never into
URLs, log lines, exceptions or recordings (`scrub` and `install_log_scrubber`).
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import random
import re
import time
from typing import Callable, Iterable, Optional

import numpy as np

SR = 16000
log = logging.getLogger("arc_audio.integrations")


class MissingKey(RuntimeError):
    pass


class ProviderError(RuntimeError):
    """A provider error. retry=False: do not reconnect (bad key, bad request, out of quota)."""

    def __init__(self, msg, retry=True, wait_s=0.0):
        super().__init__(msg)
        self.retry = retry
        self.wait_s = wait_s          # wait at least this long before reconnecting (quota / concurrency limits)


# ---------------------------------------------------------------- secrets
_AUTH_RE = re.compile(r"(?i)((?:authorization|x-api-key|api[_-]?key|token|jwt)[\"']?\s*[:=]\s*[\"']?)(?:(?:token|bearer)\s+)?[^\s\"',;&]+")


def scrub(text, secrets: Iterable[str] = ()):
    """text with auth header values and the given secret values replaced by '***'."""
    s = _AUTH_RE.sub(lambda m: m.group(1) + "***", str(text))
    for k in secrets:
        if k and len(k) >= 8:
            s = s.replace(k, "***")
    return s


class ScrubFilter(logging.Filter):
    """a logging filter that removes auth headers and the known key values from every record (websockets logs the
    handshake request headers at DEBUG)."""

    def __init__(self, env_names=()):
        super().__init__()
        self.env_names = tuple(env_names)

    def filter(self, record):
        keys = [os.environ.get(n, "") for n in self.env_names]
        try:
            msg = record.getMessage()
        except Exception:
            return True
        record.msg, record.args = scrub(msg, keys), ()
        return True


_installed = set()


def install_log_scrubber(env_names=()):
    """attach ScrubFilter to the websockets and integrations loggers (idempotent per key name)."""
    for name in ("websockets", "websockets.client", "arc_audio.integrations"):
        lg = logging.getLogger(name)
        for n in env_names:
            if (name, n) not in _installed:
                lg.addFilter(ScrubFilter([n]))
                _installed.add((name, n))
    for h in logging.getLogger().handlers:
        if (id(h), "h") not in _installed:
            h.addFilter(ScrubFilter(env_names))
            _installed.add((id(h), "h"))


def env_key(name):
    v = os.environ.get(name)
    if not v:
        raise MissingKey(f"set {name} in the environment")
    return v


def pcm16(chunk):
    """float32 [-1, 1] (or already int16 / bytes) -> little-endian PCM16 bytes."""
    if isinstance(chunk, (bytes, bytearray, memoryview)):
        return bytes(chunk)
    a = np.asarray(chunk)
    if a.dtype == np.int16:
        return a.astype("<i2").tobytes()
    return (np.clip(a.astype(np.float32), -1, 1) * 32767).astype("<i2").tobytes()


# ---------------------------------------------------------------- results
@dataclasses.dataclass
class Parsed:
    """what one server message means for Arc."""
    words: list = dataclasses.field(default_factory=list)   # word events (each with its own is_final)
    final: bool = False             # some of the words are final
    covered: Optional[float] = None  # the recogniser has finalised the audio up to this time (s, provider clock)
    done: bool = False              # the server ended the session (all finals delivered)
    error: Optional[str] = None     # a server error message
    retry: bool = True              # the error is transient
    wait_s: float = 0.0             # ... but reconnect no sooner than this (quota / concurrency limits)
    endpoint: bool = False          # end of an utterance / turn on the provider side
    info: Optional[dict] = None     # session metadata (request id, model), for logs


def word(text, start, end, final, confidence=None):
    return dict(text=str(text), start=round(float(start), 3), end=round(float(end), 3), is_final=bool(final),
                confidence=None if confidence is None else round(float(confidence), 4))


@dataclasses.dataclass
class Options:
    language: str = "en"
    model: Optional[str] = None           # provider default when None
    connect_timeout_s: float = 10.0
    stall_timeout_s: Optional[float] = -1.0   # no message for this long while audio goes in: reconnect (-1: the
                                              # provider's default; None: off, the websocket ping still runs)
    final_timeout_s: float = 5.0          # finalise / close: wait at most this long for the last finals
    max_retries: int = 3                  # reconnects before giving up (then: on_error, fallback)
    backoff_s: float = 0.5                # first backoff; doubles, with jitter, capped at 8 s
    endpoint_ms: Optional[int] = None     # provider end-of-utterance silence (provider default when None)
    interim: bool = True                  # interim / partial results
    extra: dict = dataclasses.field(default_factory=dict)   # extra provider parameters


class StreamingASR:
    """async context manager: `await send(chunk)` audio as it comes, `await finalize()` at the end of a turn,
    `await close()` at the end of the input. Words arrive through on_words(events), coverage through on_covered(t)."""

    name = "base"
    key_env: tuple = ()
    chunk_ms = (20, 1000)                 # accepted audio chunk length (ms)
    stall_default = 15.0                  # the provider sends results (or acks) all the time, silence included
    fatal_close_codes = (1008, 4001, 4003)  # websocket close codes that mean: do not reconnect

    def __init__(self, on_words: Callable[[list], None], on_covered: Optional[Callable[[float], None]] = None,
                 on_error: Optional[Callable[[Exception], None]] = None, options: Optional[Options] = None,
                 key: Optional[str] = None, record: Optional[Callable[[dict], None]] = None, **kw):
        self.opt = options or Options(**{k: v for k, v in kw.items() if k in Options.__dataclass_fields__})
        self.opt.extra.update({k: v for k, v in kw.items() if k not in Options.__dataclass_fields__})
        self.on_words, self.on_covered, self.on_error = on_words, on_covered or (lambda t: None), on_error
        self.record = record
        self._key = key
        self.ws = self.rx = self.ka = None
        self.n_chunks = 0                 # audio frames sent in this session
        self.last_tx = time.monotonic()
        self.offset = 0.0                 # provider clock 0 = this audio time (moves on reconnect)
        self.sent_s = 0.0                 # audio sent so far (audio clock)
        self.covered = 0.0                # audio clock
        self.buf = []                     # (t0, bytes) not yet covered, for a resend after a reconnect
        self.buf_max_s = 60.0
        self.failed = None                # the exception that made it give up
        self.closed = False
        self.ending = False               # the end-of-stream message is sent
        self.retries = 0
        self.last_rx = time.monotonic()
        self.n_final_msgs = 0
        self.last_final_rx = None         # monotonic time of the last final words / coverage
        self.done = asyncio.Event() if _has_loop() else None
        self.final_seen = None
        self.info = {}
        self.t0 = None                    # monotonic time of the first audio sent

    # ---- provider hooks
    def url(self) -> str:
        raise NotImplementedError

    def headers(self) -> dict:
        raise NotImplementedError

    def start_message(self) -> Optional[str]:
        return None

    def finalize_message(self) -> Optional[str]:
        return None

    def end_message(self) -> Optional[str]:
        return None

    def keepalive_message(self) -> Optional[str]:
        return None

    def parse(self, message) -> Parsed:
        raise NotImplementedError

    def classify_close(self, code, reason):
        """(retry?, minimum wait s) for a websocket close by the server. Limits on concurrent sessions or rate are
        transient whatever the code; the provider's fatal codes (auth, bad request) are not."""
        r = (reason or "").lower()
        if any(x in r for x in ("concurrent", "too many", "rate limit", "quota")):
            return True, 3.0
        return code not in self.fatal_close_codes, 0.0

    async def on_open(self):
        """after the start message: wait for the provider's acknowledgement if it sends one"""

    # ---- helpers
    @property
    def key(self):
        if self._key is None:
            if not self.key_env:
                return None
            self._key = env_key(self.key_env[0])
        return self._key

    def _rec(self, kind, payload):
        if self.record is not None:
            t = time.monotonic() - self.t0 if self.t0 is not None else 0.0
            self.record(dict(t=round(t, 4), kind=kind, offset=self.offset, msg=payload))

    def handle(self, message):
        """apply one server message (also used to replay recorded messages)."""
        p = self.parse(message)
        if p.info:
            self.info.update(p.info)
        if p.error:
            raise ProviderError(f"{self.name}: {scrub(p.error, [self._key or ''])}", retry=p.retry, wait_s=p.wait_s)
        if p.words:
            ev = []
            for w in p.words:
                w = dict(w, start=round(w["start"] + self.offset, 3), end=round(w["end"] + self.offset, 3))
                if w["is_final"] and w["end"] <= self.covered - 0.05 and self.retries:
                    continue                                # already final before a reconnect
                ev.append(w)
            if p.final:
                self.n_final_msgs += 1
                self.last_final_rx = time.monotonic()
            if ev:
                self.on_words(ev)
        if p.covered is not None:
            t = p.covered + self.offset
            if t > self.covered:
                self.covered = t
                self.last_final_rx = time.monotonic()
                self.buf = [(a, b) for a, b in self.buf if a + len(b) / 2 / SR > t - 1.0]
                self.on_covered(t)
        if p.final or p.covered is not None or p.endpoint:
            if self.final_seen is not None:
                self.final_seen.set()
        if p.done and self.done is not None:
            self.done.set()
        return p

    # ---- connection
    async def _connect(self):
        import websockets
        install_log_scrubber(self.key_env)
        url = self.url()
        try:
            self.ws = await asyncio.wait_for(
                websockets.connect(url, additional_headers=self.headers(), open_timeout=self.opt.connect_timeout_s,
                                   max_size=2 ** 22, ping_interval=10, ping_timeout=20),
                self.opt.connect_timeout_s + 1)
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            retry = status not in (400, 401, 402, 403, 404)
            raise ProviderError(f"{self.name}: connect failed ({type(e).__name__}{', HTTP ' + str(status) if status else ''})",
                                retry=retry, wait_s=3.0 if status == 429 else 0.0) from None
        self.n_chunks = 0
        self.done = asyncio.Event()
        self.final_seen = asyncio.Event()
        try:
            m = self.start_message()
            if m is not None:
                self._rec("tx", json.loads(m))
                await self.ws.send(m)
            await self.on_open()
        except websockets.ConnectionClosed as e:
            rcv = getattr(e, "rcvd", None)
            code, reason = getattr(rcv, "code", None), scrub(getattr(rcv, "reason", "") or "", [self._key or ""])
            retry, wait = self.classify_close(code, reason)
            raise ProviderError(f"{self.name}: connection closed at start ({code} {reason})".strip(), retry=retry, wait_s=wait) from None
        self.last_rx = time.monotonic()
        self.rx = asyncio.create_task(self._receiver(self.ws))
        if self.keepalive_message() is not None and self.ka is None:
            self.ka = asyncio.create_task(self._keepalive())

    async def _keepalive(self, every_s=4.0):
        """the provider's keep-alive while no audio goes in (a paused microphone): some close an idle socket after ~10 s."""
        while not self.closed:
            await asyncio.sleep(1.0)
            if self.ws is not None and not self.ending and time.monotonic() - self.last_tx > every_s:
                try:
                    await self.ws.send(self.keepalive_message())
                    self.last_tx = time.monotonic()
                except Exception:
                    pass

    async def _receiver(self, ws):
        import websockets
        try:
            async for msg in ws:
                self.last_rx = time.monotonic()
                if isinstance(msg, bytes):
                    continue
                try:
                    payload = json.loads(msg)
                except ValueError:
                    continue
                self._rec("rx", payload)
                self.handle(payload)
        except ProviderError as e:
            self._fail_soon(e)
        except websockets.ConnectionClosed as e:
            if not (self.closed or self.ending or self.done.is_set()):
                rcv = getattr(e, "rcvd", None)
                code, reason = getattr(rcv, "code", None), scrub(getattr(rcv, "reason", "") or "", [self._key or ""])
                retry, wait = self.classify_close(code, reason)
                self._fail_soon(ProviderError(f"{self.name}: connection closed ({code} {reason})".strip(), retry=retry, wait_s=wait))
        finally:
            if self.done is not None and (self.closed or self.ending):
                self.done.set()

    def _fail_soon(self, e):
        self.pending_error = e
        if self.final_seen is not None:
            self.final_seen.set()

    pending_error = None

    async def __aenter__(self):
        await self._with_retries(self._connect)
        return self

    async def __aexit__(self, *a):
        if not self.closed:
            try:
                await self.close()
            except Exception:
                pass
        await self._drop()

    async def _drop(self):
        if self.ka is not None and (self.closed or self.failed is not None):
            self.ka.cancel()
            self.ka = None
        if self.rx is not None:
            self.rx.cancel()
            self.rx = None
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None

    async def _with_retries(self, fn):
        while True:
            try:
                return await fn()
            except Exception as e:
                if not isinstance(e, ProviderError):
                    e = ProviderError(f"{self.name}: {type(e).__name__}: {scrub(e, [self._key or ''])}")
                if not e.retry or self.retries >= self.opt.max_retries:
                    self._give_up(e)
                    return None
                self.retries += 1
                d = self._backoff(e)
                log.warning("%s: %s; reconnecting in %.1f s (%d/%d)", self.name, e, d, self.retries, self.opt.max_retries)
                await self._drop()
                await asyncio.sleep(d)

    def _backoff(self, e):
        d = min(8.0, self.opt.backoff_s * 2 ** (self.retries - 1)) * (0.8 + 0.4 * random.random())
        return max(d, getattr(e, "wait_s", 0.0) * (1 + 0.5 * random.random()))

    def _give_up(self, e):
        if self.failed is None:
            self.failed = e
            log.warning("%s: giving up: %s", self.name, e)
            if self.on_error is not None:
                self.on_error(e)

    async def _reconnect(self):
        """a new session that starts at the first audio the recogniser had not finalised, on the same clock."""
        await self._drop()
        self.offset = self.buf[0][0] if self.buf else self.sent_s
        await self._connect()
        for _, b in list(self.buf):
            await self.ws.send(b)
            self.n_chunks += 1

    async def _check(self):
        if self.failed is not None or self.closed:
            return False
        st = self.stall_default if self.opt.stall_timeout_s == -1.0 else self.opt.stall_timeout_s
        stalled = st is not None and self.ws is not None and time.monotonic() - self.last_rx > st and self.sent_s > 2.0
        if self.pending_error is not None or self.ws is None or stalled:
            e = self.pending_error or ProviderError(f"{self.name}: no message for {st:.0f} s")
            self.pending_error = None
            if not e.retry or self.retries >= self.opt.max_retries:
                self._give_up(e)
                await self._drop()
                return False
            self.retries += 1
            d = self._backoff(e)
            log.warning("%s: %s; reconnecting in %.1f s (%d/%d)", self.name, e, d, self.retries, self.opt.max_retries)
            await asyncio.sleep(d)
            await self._with_retries(self._reconnect)
            return self.failed is None
        return True

    # ---- audio and control
    async def send(self, chunk):
        """one chunk of 16 kHz mono audio (float32 in [-1, 1], int16 or PCM16 bytes)."""
        b = pcm16(chunk)
        if self.t0 is None:
            self.t0 = time.monotonic()
        t = self.sent_s
        self.sent_s += len(b) / 2 / SR
        self.buf.append((t, b))
        while self.buf and self.sent_s - self.buf[0][0] > self.buf_max_s:
            self.buf.pop(0)
        if not await self._check():
            return False
        try:
            await self.ws.send(b)
            self.n_chunks += 1
            self.last_tx = time.monotonic()
        except Exception as e:
            self.pending_error = ProviderError(f"{self.name}: send failed ({type(e).__name__})")
            return await self._check()
        return True

    async def _control(self, m):
        if m is None or self.ws is None or self.failed is not None:
            return
        self._rec("tx", json.loads(m))
        try:
            await self.ws.send(m)
        except Exception as e:
            self.pending_error = ProviderError(f"{self.name}: send failed ({type(e).__name__})")

    async def finalize(self, timeout=None):
        """end of a turn / end of the input: ask the recogniser for its final words now, and wait (at most timeout s)
        until the next final result arrives. A connection lost meanwhile is re-opened (the unfinalised audio is sent
        again) and the finalise repeated. Returns True when a final came."""
        if self.finalize_message() is None:
            return False
        for _ in range(self.opt.max_retries + 1):
            if not await self._check() or self.ws is None:
                return False
            self.final_seen.clear()
            await self._control(self.finalize_message())
            try:
                await asyncio.wait_for(self.final_seen.wait(), self.opt.final_timeout_s if timeout is None else timeout)
            except asyncio.TimeoutError:
                return False
            if self.pending_error is None:
                return True
        return False

    async def close(self, timeout=None):
        """end of the input: send the provider's end-of-stream, wait (at most timeout s) for the remaining finals and the
        session end, then close the socket. Coverage moves to the end of the audio either way."""
        if self.closed:
            return
        await self._check()
        if self.failed is None and self.ws is not None:
            self.ending = True
            await self._control(self.end_message())
            try:
                await asyncio.wait_for(self.done.wait(), self.opt.final_timeout_s if timeout is None else timeout)
            except asyncio.TimeoutError:
                log.warning("%s: no end of session within %.1f s", self.name, self.opt.final_timeout_s)
        self.closed = True
        if self.failed is None:
            self.covered = max(self.covered, self.sent_s)
            self.on_covered(self.sent_s + 10.0)
        await self._drop()

    def __repr__(self):
        return f"<{type(self).__name__} sent={self.sent_s:.1f}s covered={self.covered:.1f}s retries={self.retries}>"


def _has_loop():
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def replay(adapter: StreamingASR, messages: Iterable[dict]):
    """apply recorded server messages (dicts in arrival order) to an adapter, no network: what tests and the bench use."""
    out = []
    for m in messages:
        out.append(adapter.handle(m))
    return out
