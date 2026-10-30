"""Latency: offline decode time, and in a real-time stream when each output arrives after the speaker stops
(partial, final transcript, utterance record with act/intent/device-directed, command_complete), plus what flush()
costs when an app ends the turn itself (the Pipecat integration's workaround). See README.md.
  python evals/latency_streaming/run.py [--voices Samantha,Daniel] [--tail 6] [--threads 4]
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402

CHUNK = 1600   # 100 ms


def speech_end(x, before):
    """End of the speech in the clip (s): last 10 ms frame within 35 dB of the loudest."""
    f = 160
    n = len(x) // f
    e = 10 * np.log10(np.mean(x[: n * f].reshape(n, f) ** 2, axis=1) + 1e-12)
    idx = np.where(e > e.max() - 35)[0]
    return idx[-1] * f / C.SR + 0.01 if len(idx) else before


def run_stream(arc, x, end, paced=True):
    """Feed x in 100 ms chunks at real-time pace; return when each record type first arrives, in seconds after the
    speech end (audio clock of the chunk just fed), plus the per-feed compute times."""
    s = arc.stream(domain="assistant")
    got = dict(partial_full=None, transcript=None, utterance=None, device_directed=None)
    cc_times, partials = [], []
    final_text = None
    t_next = time.perf_counter()
    for i in range(0, len(x), CHUNK):
        o = s.feed(x[i:i + CHUNK])
        pos = (i + CHUNK) / C.SR - end
        p = o.get("transcript_partial")
        if p and p.get("text"):
            partials.append((pos, p["text"]))
        if o.get("transcript") and got["transcript"] is None:
            got["transcript"] = pos
            final_text = C.text_of(o["transcript"])
        if o.get("utterances") and got["utterance"] is None:
            got["utterance"] = pos
        if o.get("device_directed") and got["device_directed"] is None:
            got["device_directed"] = pos
        for ev in o.get("events") or []:
            if ev.get("tag") == "discourse.assistant.command_complete":
                cc_times.append((pos, ev["t"] - end))
        if paced:
            t_next += CHUNK / C.SR
            time.sleep(max(0.0, t_next - time.perf_counter()))
    t0 = time.perf_counter()
    fo = s.flush()
    flush_ms = 1000 * (time.perf_counter() - t0)
    if final_text is None:
        final_text = C.text_of(fo.get("transcript") or [])
    for pos, text in partials:   # the first partial that already has the final words
        if C.words(text) == C.words(final_text):
            got["partial_full"] = (pos, text)
            break
    feeds = [w for a, w in s.timing]
    return got, cc_times, final_text, flush_ms, feeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default="Samantha,Daniel")
    ap.add_argument("--tail", type=float, default=6.0, help="seconds of silence fed after the speech")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--clips", default=os.path.join(C.WORK, "latency_streaming"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "run.json"))
    a = ap.parse_args()
    lines = [r[1] for r in C.read_tsv(os.path.join(HERE, "data", "commands.tsv"))]
    arc = C.load_arc(a.threads)
    rows, offline, flush_rows, feeds_all = [], [], [], []
    for v in a.voices.split(","):
        for text in lines:
            w = C.load(C.render(text, v, os.path.join(a.clips, f"{v}_{C.slug(text)}.wav")))
            w = w / max(1e-4, float(np.abs(w).max())) * 0.35
            before = 0.5
            x = C.room(w, -60, level=0.35, before=before, after=a.tail, seed=len(rows))
            end = speech_end(x[: int((before + len(w) / C.SR) * C.SR)], before)
            # offline decode of the clip with a 0.4 s tail (what press-and-hold sends)
            clip = x[: int((end + 0.4) * C.SR)]
            t0 = time.perf_counter()
            arc.transcribe(clip)
            offline.append(dict(text=text, audio_s=len(clip) / C.SR, transcribe_ms=1000 * (time.perf_counter() - t0)))
            t0 = time.perf_counter()
            arc.predict(clip, domain="assistant", return_frames=False, events=None)
            offline[-1]["predict_ms"] = 1000 * (time.perf_counter() - t0)
            # real-time stream with a long silent tail
            got, cc, final_text, _, feeds = run_stream(arc, x, end)
            feeds_all += feeds
            # the app ends the turn itself: speech + 0.3 s, then flush()
            _, _, ftext, flush_ms, _ = run_stream(arc, x[: int((end + 0.3) * C.SR)], end)
            flush_rows.append(dict(text=text, flush_ms=flush_ms, text_heard=ftext))
            early = [t for pos, t in cc if t < -0.5]
            rows.append(dict(voice=v, said=text, speech_end_s=end, final_text=final_text, **{k: got[k] for k in got},
                             command_complete=cc, command_complete_early=early))
            print(f"{v:9} {text:42} transcript {got['transcript']}  utt {got['utterance']}  cc {[round(t, 2) for _, t in cc]}  flush {flush_ms:.0f} ms", file=sys.stderr)

    def stat(xs):
        xs = [x for x in xs if x is not None]
        return dict(n=len(xs), median=C.quantile(xs, 0.5), p90=C.quantile(xs, 0.9), max=max(xs) if xs else None)
    summary = dict(
        clips=len(rows), tail_s=a.tail,
        final_transcript_after_end_s=stat([r["transcript"] for r in rows]),
        final_transcript_missing_within_tail=sum(r["transcript"] is None for r in rows),
        utterance_record_after_end_s=stat([r["utterance"] for r in rows]),
        device_directed_after_end_s=stat([r["device_directed"] for r in rows]),
        partial_after_end_s=stat([r["partial_full"][0] if r["partial_full"] else None for r in rows]),
        command_complete_emitted_after_end_s=stat([cc[-1][0] for cc in (r["command_complete"] for r in rows) if cc]),
        command_complete_missing=sum(not r["command_complete"] for r in rows),
        command_complete_early_mid_command=sum(bool(r["command_complete_early"]) for r in rows),
        flush_ms=stat([r["flush_ms"] for r in flush_rows]),
        feed_ms=stat([1000 * f for f in feeds_all]),
        offline_transcribe_ms=stat([o["transcribe_ms"] for o in offline]),
        offline_predict_ms=stat([o["predict_ms"] for o in offline]),
        offline_rtf_transcribe=float(np.sum([o["transcribe_ms"] for o in offline]) / 1000 / np.sum([o["audio_s"] for o in offline])),
    )
    for k, v in summary.items():
        print(f"{k:42} {v}")
    C.save(a.out, dict(env=C.env_info(arc, a.threads), summary=summary, rows=rows, flush=flush_rows, offline=offline))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
