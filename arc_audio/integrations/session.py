"""An Arc EXTERNAL stream fed by a cloud recogniser: Arc listens to the audio (tags, directedness, end of turn, acts,
events) and takes its words from the provider.

    import asyncio
    from arc_audio import Arc
    from arc_audio.integrations import Session

    arc = Arc.from_pretrained("WhissleAI/arc-v1.1")
    async def main(chunks):                      # 100 ms float32 16 kHz mono chunks, as they come
        async with Session(arc, "deepgram", domain="assistant") as s:
            for c in chunks:
                new = await s.feed(c)           # same keys as Stream.feed(): transcript, utterances, names, ...
            out = await s.finish()              # finalise at the provider, then flush Arc

Fallback: when the provider fails for good (bad key, out of retries, a timeout), the session starts Arc's own decoder
at the first audio the provider had not finalised and feeds its words into the same EXTERNAL stream, so the output
keeps every key and the tags do not restart. `out["asr"]` says what happened.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np

from .base import SR, Options
from . import registry

CHUNK = 1600
LIST_KEYS = ("transcript", "transcript_written", "names", "numbers", "utterances", "segments", "spans", "events",
             "audio_events", "device_directed", "diarization", "bio_events")


def wire(stream, adapter):
    """connect an adapter's callbacks to an Arc stream opened with transcript='external'."""
    adapter.on_words = stream.add_words
    adapter.on_covered = stream.words.mark_covered
    return adapter


class Session:
    def __init__(self, arc, provider, domain="unknown", names=None, options: Options | None = None, fallback=True,
                 record=None, stream_kw=None, **opts):
        self.arc, self.provider, self.domain, self.names = arc, provider, domain, names
        self.options = options or Options(**{k: v for k, v in opts.items() if k in Options.__dataclass_fields__})
        self.options.extra.update({k: v for k, v in opts.items() if k not in Options.__dataclass_fields__})
        self.fallback_on = fallback
        self.record = record
        self.stream_kw = dict(stream_kw or {})
        self.acc = {k: [] for k in LIST_KEYS}
        self.audio = []                  # (t0, chunk) since the provider's coverage, for a fallback
        self.total = 0
        self.fb = None
        self.fb_from = None
        self.error = None
        self.t_end = None                # monotonic time the input ended
        self.t_last_final = None         # monotonic time of the last provider final

    # ---- set-up
    async def __aenter__(self):
        kw = dict(domain=self.domain, names=self.names, transcript="external", name_source="word")
        kw.update(self.stream_kw)
        self.s = self.arc.stream(**kw)
        cls = registry.get(self.provider)
        self.asr = cls(on_words=self._words, on_covered=self._covered, on_error=self._failed, options=self.options,
                       record=self.record)
        await self.asr.__aenter__()
        return self

    async def __aexit__(self, *a):
        await self.asr.__aexit__(*a)
        if self.fb is not None:
            try:
                self.fb.close()
            except Exception:
                pass
        try:
            self.s.close()
        except Exception:
            pass

    def _words(self, ev):
        self.s.add_words(ev)
        if any(w["is_final"] for w in ev):
            self.t_last_final = time.monotonic()

    def _covered(self, t):
        self.s.words.mark_covered(t)

    def _failed(self, e):
        self.error = e
        if self.fallback_on and self.fb is None:
            self._start_fallback()

    # ---- fallback: Arc's own decoder from the first audio the provider had not finalised
    def _start_fallback(self):
        t0 = min(self.asr.covered, self.total / SR)
        self.fb_from = t0
        self.fb = self.arc.stream(domain=self.domain, names=self.names, transcript="builtin", live=False, events=None)
        pcm = [c for a, c in self.audio if a + len(c) / SR > t0]
        if pcm:
            a0 = next(a for a, c in self.audio if a + len(c) / SR > t0)
            self.fb_t0 = a0
            self._fb_take(self.fb.feed(np.concatenate(pcm)))
        else:
            self.fb_t0 = self.total / SR
        self.audio = []

    def _fb_take(self, out):
        ev = [dict(text=w["text"], start=w["start"] + self.fb_t0, end=w["end"] + self.fb_t0, is_final=True, source="arc")
              for u in out.get("transcript") or [] for w in u.get("words") or []]
        if ev:
            self.s.add_words(ev)
            self.s.words.mark_covered(max(w["end"] for w in ev))

    def _add(self, new):
        for k in LIST_KEYS:
            self.acc[k] += new.get(k) or []
        return new

    # ---- audio
    async def feed(self, chunk):
        chunk = np.asarray(chunk, np.float32).reshape(-1)
        t = self.total / SR
        self.total += len(chunk)
        if self.fb is None:
            self.audio.append((t, chunk))
            cov = self.asr.covered
            while self.audio and self.audio[0][0] + len(self.audio[0][1]) / SR < cov - 1.0:
                self.audio.pop(0)
            await self.asr.send(chunk)
        if self.fb is not None:
            self._fb_take(self.fb.feed(chunk))
        return self._add(self.s.feed(chunk))

    async def end_turn(self):
        """push-to-talk released / end of a turn: finals now at the provider and in Arc; the stream stays open."""
        if self.fb is None:
            await self.asr.finalize()
        return self._add(self.s.end_turn())

    async def finish(self):
        """end of the input: finalise at the provider (Finalize / ForceEndpoint / ForceEndOfUtterance), end its session,
        then flush Arc. Returns the whole session's output."""
        self.t_end = time.monotonic()
        if self.fb is None:
            await self.asr.finalize()
            await self.asr.close()
        if self.fb is not None:
            self._fb_take(self.fb.flush())
        final = self._add(self.s.flush())
        out = dict(self.acc, speakers=final.get("speakers"), domain=self.domain)
        out["transcript"] = sorted(out["transcript"], key=lambda u: u["start"])
        out["text"] = " ".join(u["text"] for u in out["transcript"] if u.get("text"))
        with self.s.words.lock:
            out["words"] = [dict(text=w["w"], start=w["start"], end=w["end"], source=w.get("src")) for w in self.s.words.finals]
        out["asr"] = dict(provider=self.provider, status="fallback" if self.fb is not None else ("error" if self.error else "ok"),
                          error=str(self.error) if self.error else None, fallback_from=self.fb_from, retries=self.asr.retries,
                          info=self.asr.info,
                          time_to_final_s=None if self.t_last_final is None or self.fb is not None
                          else round(max(0.0, self.t_last_final - self.t_end), 3))
        return out


async def file_chunks(wav, speed=1.0, chunk=CHUNK):
    """a file's audio in 100 ms chunks at speed x real time (0: as fast as possible)."""
    t0 = time.monotonic()
    for k, i in enumerate(range(0, len(wav), chunk)):
        yield wav[i:i + chunk]
        if speed:
            await asyncio.sleep(max(0.0, t0 + (k + 1) * chunk / SR / speed - time.monotonic()))
        else:
            await asyncio.sleep(0)


async def run_file(arc, wav, provider, speed=1.0, **kw):
    async with Session(arc, provider, **kw) as s:
        async for c in file_chunks(wav, speed):
            await s.feed(c)
        return await s.finish()
