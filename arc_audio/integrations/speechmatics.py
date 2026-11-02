"""Speechmatics real-time (`wss://eu.rt.speechmatics.com/v2`) -> Arc external words.

    export SPEECHMATICS_API_KEY=...
    python -m arc_audio.integrations.speechmatics clip.wav
    SPEECHMATICS_RT_URL=wss://us.rt.speechmatics.com/v2 ...      # another region (or the `url=` option)

Protocol (Speechmatics real-time API reference): `StartRecognition` (raw pcm_s16le, 16 kHz, the transcription config),
`RecognitionStarted`, binary audio frames (`AddAudio`, each acknowledged by `AudioAdded`); `AddPartialTranscript`
(interim, replaced by the next) and `AddTranscript` (final: the words of `metadata.start_time .. end_time`), word and
punctuation results (punctuation `attaches_to` the previous word); `max_delay` bounds how late a final comes (0.7 s
is the lowest). `ForceEndOfUtterance` finalises now; `EndOfStream` (`last_seq_no`) ends the input and the server
answers with the remaining finals and `EndOfTranscript`. Auth: `Authorization: Bearer <key>`.
"""
from __future__ import annotations

import asyncio
import json
import os

from .base import Parsed, ProviderError, StreamingASR, word

URL = "wss://eu.rt.speechmatics.com/v2"
FATAL = ("not_authorised", "invalid_model", "invalid_config", "invalid_audio_type", "not_allowed", "invalid_message",
         "protocol_error", "insufficient_funds")


class Speechmatics(StreamingASR):
    name = "speechmatics"
    key_env = ("SPEECHMATICS_API_KEY",)
    stall_default = 15.0           # AudioAdded acknowledges every frame
    fatal_close_codes = (1008, 4001, 4003)

    def url(self):
        return self.opt.extra.get("url") or os.environ.get("SPEECHMATICS_RT_URL") or URL

    def headers(self):
        return {"Authorization": f"Bearer {self.key}"}

    def start_message(self):
        o = self.opt
        tc = dict(language=o.language or "en", enable_partials=bool(o.interim),
                  max_delay=float(o.extra.get("max_delay", 0.7)), max_delay_mode=o.extra.get("max_delay_mode", "flexible"),
                  model=o.model or "enhanced")         # "operating_point" in older versions of the API
        if o.endpoint_ms is not None:
            tc["conversation_config"] = dict(end_of_utterance_silence_trigger=o.endpoint_ms / 1000)
        for k, v in o.extra.items():
            if k not in ("url", "max_delay", "max_delay_mode"):
                tc[k] = v
        return json.dumps(dict(message="StartRecognition", audio_format=dict(type="raw", encoding="pcm_s16le", sample_rate=16000),
                               transcription_config=tc))

    async def on_open(self):
        """wait for RecognitionStarted (or the error that explains why not)."""
        while True:
            try:
                raw = await asyncio.wait_for(self.ws.recv(), self.opt.connect_timeout_s)
            except asyncio.TimeoutError:
                raise ProviderError("speechmatics: no RecognitionStarted") from None
            except asyncio.CancelledError:
                raise
            m = json.loads(raw) if isinstance(raw, str) else {}
            self._rec("rx", m)
            p = self.handle(m)
            if m.get("message") == "RecognitionStarted":
                return

    def finalize_message(self):
        return json.dumps(dict(message="ForceEndOfUtterance", timestamp=round(self.sent_s - self.offset, 3)))

    def end_message(self):
        return json.dumps(dict(message="EndOfStream", last_seq_no=self.n_chunks))

    def parse(self, m):
        return parse(m)


def parse(m):
    """one Speechmatics real-time server message -> Parsed."""
    m = json.loads(m) if isinstance(m, (str, bytes)) else (m or {})
    t = m.get("message")
    if t in ("AddTranscript", "AddPartialTranscript"):
        fin = t == "AddTranscript"
        out = []
        for r in m.get("results") or []:
            alt = (r.get("alternatives") or [{}])[0]
            c = str(alt.get("content", ""))
            if r.get("type") == "punctuation":
                if out and r.get("attaches_to", "previous") in ("previous", "both"):
                    out[-1]["text"] += c
                continue
            if "start_time" not in r or not c:
                continue
            out.append(word(c, r["start_time"], r["end_time"], fin, alt.get("confidence") if fin else None))
        md = m.get("metadata") or {}
        return Parsed(words=out, final=fin, covered=float(md["end_time"]) if fin and "end_time" in md else None)
    if t == "EndOfUtterance":
        return Parsed(endpoint=True)
    if t == "EndOfTranscript":
        return Parsed(done=True)
    if t == "RecognitionStarted":
        return Parsed(info=dict(session_id=m.get("id"), language_pack=(m.get("language_pack_info") or {}).get("language_description")))
    if t == "Error":
        typ = str(m.get("type", ""))
        return Parsed(error=f"{typ} {m.get('reason', '')}".strip(), retry=typ not in FATAL,
                      wait_s=5.0 if typ in ("quota_exceeded", "job_error") else 0.0)       # the API reference: retry after 5-10 s
    return Parsed()


if __name__ == "__main__":
    from .cli import main
    main("speechmatics")
