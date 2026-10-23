"""External ASR adapters: Deepgram (pre-recorded REST + live websocket) and Verbit (realtime websocket messages).

Arc always listens to the audio itself; the external words are its optional heard-text input. Keys come from environment
variables only and are never printed or written:
    DEEPGRAM_API_KEY                  Deepgram
    VERBIT_WS_URL, VERBIT_API_KEY     the websocket URL and bearer token of a booked Verbit live order (see Verbit's realtime API
                                      documentation; booking the order is not part of this package)
Word events passed to Arc: {"text", "start", "end", "is_final"}, seconds on the audio clock (0 = first sample sent).
"""
import json, os, urllib.request

import numpy as np

DG_HTTP = "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&punctuate=true"
DG_WS = ("wss://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&punctuate=true&interim_results=true"
         "&encoding=linear16&sample_rate=16000&channels=1")
VERBIT_QS = "format=S16LE&sample_rate=16000&sample_width=2&num_channels=1&get_captions=True&get_transcript=True"


class MissingKey(RuntimeError):
    pass


def env_key(*names):
    vals = [os.environ.get(n) for n in names]
    if not all(vals):
        raise MissingKey("set " + " and ".join(names) + " in the environment (or use --demo)")
    return vals if len(vals) > 1 else vals[0]


def pcm16(chunk):
    return (np.clip(np.asarray(chunk, np.float32), -1, 1) * 32767).astype("<i2").tobytes()


def ws_headers_kw():
    import websockets
    return "additional_headers" if int(websockets.__version__.split(".")[0]) >= 14 else "extra_headers"


# ---------------------------------------------------------------- Deepgram
def deepgram_words(response):
    """Deepgram pre-recorded response JSON -> Arc word events (final)."""
    alt = response["results"]["channels"][0]["alternatives"][0]
    return [dict(text=w.get("punctuated_word") or w["word"], start=float(w["start"]), end=float(w["end"]), is_final=True)
            for w in alt.get("words", [])]


def deepgram_live_words(message):
    """Deepgram live 'Results' message -> (word events, is_final)."""
    m = json.loads(message) if isinstance(message, (str, bytes)) else message
    if (m or {}).get("type", "Results") != "Results":
        return [], False
    fin = bool(m.get("is_final"))
    alts = ((m.get("channel") or {}).get("alternatives") or [{}])
    return [dict(text=w.get("punctuated_word") or w.get("word", ""), start=float(w["start"]), end=float(w["end"]), is_final=fin)
            for w in alts[0].get("words") or [] if "start" in w and "end" in w], fin


def deepgram_file(wav, key, timeout=600):
    """16 kHz mono audio -> Deepgram pre-recorded REST -> Arc word events. The same samples Arc gets, so the clocks agree."""
    import io, soundfile as sf
    buf = io.BytesIO(); sf.write(buf, np.asarray(wav, np.float32), 16000, format="WAV", subtype="PCM_16")
    req = urllib.request.Request(DG_HTTP, data=buf.getvalue(), method="POST",
                                 headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return deepgram_words(json.loads(r.read()))


class DeepgramLive:
    """async context: send PCM16 chunks, word events arrive through on_words(events)."""

    def __init__(self, key, on_words):
        self.key, self.on_words, self.ws, self.rx = key, on_words, None, None

    async def __aenter__(self):
        import asyncio, websockets
        self.ws = await websockets.connect(DG_WS, **{ws_headers_kw(): {"Authorization": f"Token {self.key}"}})

        async def receiver():
            async for msg in self.ws:
                if isinstance(msg, str):
                    words, _ = deepgram_live_words(msg)
                    if words:
                        self.on_words(words)
        self.rx = asyncio.create_task(receiver())
        return self

    async def send(self, chunk):
        await self.ws.send(pcm16(chunk))

    async def close(self, tail_s=3.0):
        import asyncio
        await self.ws.send(json.dumps({"type": "CloseStream"}))
        try:
            await asyncio.wait_for(self.rx, tail_s)
        except (asyncio.TimeoutError, Exception):
            self.rx.cancel()

    async def __aexit__(self, *a):
        await self.ws.close()


# ---------------------------------------------------------------- Verbit
def verbit_words(message, offset_s=0.0):
    """Verbit realtime message -> (word events, is_final). 'captions' responses carry committed (final) words; 'transcript' responses
    that are not final carry interim words; a final 'transcript' response closes a segment whose words came as captions already.
    Punctuation items are attached to the previous word."""
    env = json.loads(message) if isinstance(message, (str, bytes)) else message
    r = (env or {}).get("response") or {}
    if not r.get("alternatives"):
        return [], False
    final = r.get("type") == "captions"
    if not final and r.get("is_final"):
        return [], False
    out = []
    for it in r["alternatives"][0].get("items") or []:
        if it.get("kind") == "punct":
            if out:
                out[-1]["text"] += it.get("value", "")
            continue
        out.append(dict(text=str(it.get("value", "")), start=offset_s + float(it["start"]), end=offset_s + float(it["end"]), is_final=final))
    return out, final


class VerbitLive:
    """async context for a booked Verbit live order: send PCM16 chunks, caption / interim words arrive through on_words(events)."""

    def __init__(self, url, token, on_words):
        self.url, self.token, self.on_words, self.ws, self.rx = url, token, on_words, None, None

    async def __aenter__(self):
        import asyncio, websockets
        url = f"{self.url}{'&' if '?' in self.url else '?'}{VERBIT_QS}"
        self.ws = await websockets.connect(url, **{ws_headers_kw(): {"Authorization": f"Bearer {self.token}"}})

        async def receiver():
            async for msg in self.ws:
                if isinstance(msg, str):
                    words, _ = verbit_words(msg)
                    if words:
                        self.on_words(words)
        self.rx = asyncio.create_task(receiver())
        return self

    async def send(self, chunk):
        await self.ws.send(pcm16(chunk))

    async def close(self, tail_s=8.0):
        import asyncio
        await self.ws.send(json.dumps({"event": "EOS"}))
        try:
            await asyncio.wait_for(self.rx, tail_s)                     # Verbit captions trail the audio by a few seconds
        except (asyncio.TimeoutError, Exception):
            self.rx.cancel()

    async def __aexit__(self, *a):
        await self.ws.close()


# ---------------------------------------------------------------- offline simulations (--demo: no key, no network)
class Simulated:
    """Deepgram- or Verbit-shaped messages built from Arc's own built-in transcript of the same audio, released at the pace of the
    audio: Deepgram interims at once and finals ~1 s after each word; Verbit captions of ~5 words, lag_s after their last word."""

    def __init__(self, kind, words, on_words, lag_s=None):
        self.kind, self.words, self.on_words = kind, words, on_words
        self.lag = (1.0 if kind == "deepgram" else 3.0) if lag_s is None else lag_s
        self.sent = 0

    def _deepgram(self, ws, fin):
        return {"type": "Results", "is_final": fin, "channel": {"alternatives": [{"words": [
            {"word": w["text"].lower(), "punctuated_word": w["text"], "start": w["start"], "end": w["end"]} for w in ws]}]}}

    def _verbit(self, ws):
        return {"response": {"type": "captions", "alternatives": [{"items": [
            {"kind": "text", "value": w["text"], "start": w["start"], "end": w["end"]} for w in ws]}]}}

    def advance(self, now, final=False):
        W = self.words
        if self.kind == "deepgram":
            fin = [w for w in W[self.sent:] if final or w["end"] + self.lag <= now]
            if fin:
                self.on_words(deepgram_live_words(self._deepgram(fin, True))[0]); self.sent += len(fin)
            inter = [w for w in W[self.sent:] if w["end"] <= now]
            if inter and not final:
                self.on_words(deepgram_live_words(self._deepgram(inter, False))[0])
        else:
            while self.sent < len(W) and (final or W[min(self.sent + 4, len(W) - 1)]["end"] + self.lag <= now):
                self.on_words(verbit_words(self._verbit(W[self.sent:self.sent + 5]))[0]); self.sent += 5
