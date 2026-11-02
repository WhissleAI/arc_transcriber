"""Deepgram live streaming (`wss://api.deepgram.com/v1/listen`, nova-3) -> Arc external words.

    export DEEPGRAM_API_KEY=...
    python -m arc_audio.integrations.deepgram clip.wav

Protocol (Deepgram streaming API reference): binary PCM16 frames in; `Results` messages out, interim (`is_final` false,
replaced by the next) and final (`is_final` true: the words of `start .. start + duration` will not change); `Finalize`
flushes the audio sent so far as a final result (`from_finalize` true); `CloseStream` ends the session (a `Metadata`
message, then the server closes). Endpointing (`endpointing` ms of silence) sets `speech_final`; Arc's own segmenter
decides utterances, so it only makes finals come sooner. Auth: `Authorization: Token <key>`.
"""
from __future__ import annotations

import json
import os
from urllib.parse import urlencode

from .base import Parsed, StreamingASR, word

URL = "wss://api.deepgram.com/v1/listen"


class Deepgram(StreamingASR):
    name = "deepgram"
    key_env = ("DEEPGRAM_API_KEY",)
    stall_default = 15.0          # Results (often empty) keep coming through silence
    fatal_close_codes = (1008,)   # DATA-0000 / NET-0001 style closes with a policy reason: bad audio or bad request

    def params(self):
        o = self.opt
        p = dict(model=o.model or "nova-3", language=o.language or "en", encoding="linear16", sample_rate=16000, channels=1,
                 interim_results=str(bool(o.interim)).lower(), punctuate="true",
                 endpointing=300 if o.endpoint_ms is None else int(o.endpoint_ms), vad_events="false")
        if o.interim:
            p["utterance_end_ms"] = 1000
        p.update({k: (str(v).lower() if isinstance(v, bool) else v) for k, v in o.extra.items() if k != "url"})
        return p

    def url(self):
        return (self.opt.extra.get("url") or os.environ.get("DEEPGRAM_URL") or URL) + "?" + urlencode(self.params())

    def headers(self):
        return {"Authorization": f"Token {self.key}"}

    def finalize_message(self):
        return json.dumps({"type": "Finalize"})

    def end_message(self):
        return json.dumps({"type": "CloseStream"})

    def keepalive_message(self):
        return json.dumps({"type": "KeepAlive"})

    def parse(self, m):
        return parse(m, ending=self.ending)


def parse(m, ending=False):
    """one Deepgram server message (dict or JSON text) -> Parsed."""
    m = json.loads(m) if isinstance(m, (str, bytes)) else (m or {})
    t = m.get("type", "Results")
    if t == "Results":
        fin = bool(m.get("is_final"))
        alt = ((m.get("channel") or {}).get("alternatives") or [{}])[0]
        ws = [word(w.get("punctuated_word") or w.get("word", ""), w["start"], w["end"], fin, w.get("confidence"))
              for w in alt.get("words") or [] if "start" in w and "end" in w]
        cov = float(m.get("start", 0.0)) + float(m.get("duration", 0.0)) if fin else None
        return Parsed(words=ws, final=fin, covered=cov, endpoint=bool(m.get("speech_final") or m.get("from_finalize")))
    if t == "Metadata":
        return Parsed(info=dict(request_id=m.get("request_id"), model=(m.get("model_info") or {})), done=ending)
    if t == "UtteranceEnd":
        return Parsed(endpoint=True)
    if t == "Error" or "err_code" in m:
        code = str(m.get("err_code") or m.get("code") or "")
        return Parsed(error=f"{code} {m.get('err_msg') or m.get('description') or m.get('message') or ''}".strip(),
                      retry=not code.startswith(("INVALID", "UNAUTH", "FORBID", "INSUFFICIENT")))
    return Parsed()


if __name__ == "__main__":
    from .cli import main
    main("deepgram")
