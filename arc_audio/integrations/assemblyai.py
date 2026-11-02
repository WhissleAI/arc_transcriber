"""AssemblyAI Universal Streaming (v3, `wss://streaming.assemblyai.com/v3/ws`) -> Arc external words.

    export ASSEMBLYAI_API_KEY=...
    python -m arc_audio.integrations.assemblyai clip.wav

Protocol (AssemblyAI streaming API reference): binary PCM16 frames of 50-1000 ms in, at about real-time pace; `Begin`,
then `Turn` messages: the words of the current turn, each with `word_is_final` (a final word does not change), and
`end_of_turn` once the turn is over (all its words final). With `format_turns=true` a second, formatted copy of the
ended turn follows (`turn_is_formatted`); this adapter then takes the finals from that copy. `ForceEndpoint` ends the
current turn now; `Terminate` ends the session (`Termination`). Times are in milliseconds. Auth: `Authorization: <key>`
(no scheme). Unknown query parameters are ignored by the service, so the model in `Begin` is logged.
"""
from __future__ import annotations

import json
import os
from urllib.parse import urlencode

from .base import Parsed, StreamingASR, word

URL = "wss://streaming.assemblyai.com/v3/ws"


class AssemblyAI(StreamingASR):
    name = "assemblyai"
    key_env = ("ASSEMBLYAI_API_KEY",)
    chunk_ms = (50, 1000)
    stall_default = None           # no messages in silence: rely on the websocket ping
    fatal_close_codes = (1008, 3005, 4001, 4002, 4003, 4008, 4031)   # auth, quota, bad parameters, session limits

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.turns = {}            # turn_order -> number of its words already sent as final

    def params(self):
        o = self.opt
        lang = (o.language or "en").lower()
        model = o.model or ("universal-streaming-english" if lang.startswith("en") else "universal-streaming-multilingual")
        p = dict(sample_rate=16000, encoding="pcm_s16le", speech_model=model,
                 format_turns=str(bool(o.extra.get("format_turns", False))).lower())
        if not lang.startswith("en"):
            p["language_detection"] = "true"
        if o.endpoint_ms is not None:
            p["min_turn_silence"] = int(o.endpoint_ms)
        p.update({k: (str(v).lower() if isinstance(v, bool) else v) for k, v in o.extra.items() if k not in ("url", "format_turns")})
        self.formatted = p["format_turns"] == "true"
        return p

    def url(self):
        return (self.opt.extra.get("url") or os.environ.get("ASSEMBLYAI_STREAMING_URL") or URL) + "?" + urlencode(self.params())

    def headers(self):
        return {"Authorization": self.key}

    def finalize_message(self):
        return json.dumps({"type": "ForceEndpoint"})

    def end_message(self):
        return json.dumps({"type": "Terminate"})

    def parse(self, m):
        return parse(m, self.turns, formatted=getattr(self, "formatted", False))


def parse(m, turns=None, formatted=False):
    """one AssemblyAI v3 server message -> Parsed. turns (dict, turn_order -> finals sent) carries state between messages:
    final words go out once, as soon as they are final; the rest of the turn goes out as interim words."""
    m = json.loads(m) if isinstance(m, (str, bytes)) else (m or {})
    turns = {} if turns is None else turns
    t = m.get("type")
    if t == "Turn":
        k = int(m.get("turn_order", 0))
        W = [w for w in m.get("words") or [] if "start" in w and "end" in w]
        eot = bool(m.get("end_of_turn"))
        if formatted and eot and not m.get("turn_is_formatted"):
            eot = False                                      # the formatted copy follows: wait for it
        if formatted:
            n_fin = len(W) if eot else 0                     # formatted text can still change: finals at the end only
        else:
            n_fin = len(W) if eot else next((i for i, w in enumerate(W) if not w.get("word_is_final")), len(W))
        done = turns.get(k, 0)
        fin = [word(w["text"], w["start"] / 1000, w["end"] / 1000, True, w.get("confidence")) for w in W[done:n_fin]]
        turns[k] = max(done, n_fin)
        inter = [word(w["text"], w["start"] / 1000, w["end"] / 1000, False, w.get("confidence")) for w in W[n_fin:]]
        cov = (W[-1]["end"] / 1000 if W else None) if eot else (W[n_fin - 1]["end"] / 1000 if n_fin else None)
        return Parsed(words=fin + inter, final=bool(fin), covered=cov if (fin or eot) else None, endpoint=eot)
    if t == "Begin":
        return Parsed(info=dict(session_id=m.get("id"), model=(m.get("configuration") or {}).get("model") or m.get("model")))
    if t == "Termination":
        return Parsed(done=True, covered=float(m["audio_duration_seconds"]) if m.get("audio_duration_seconds") else None)
    if t == "Error" or "error" in m:
        msg = str(m.get("error") or m.get("message") or "error")
        limit = any(x in msg.lower() for x in ("concurrent", "too many", "rate limit"))
        return Parsed(error=msg, retry=limit, wait_s=3.0 if limit else 0.0)    # a session limit passes; a bad key does not
    return Parsed()


if __name__ == "__main__":
    from .cli import main
    main("assemblyai")
