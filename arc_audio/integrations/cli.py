"""python -m arc_audio.integrations.<provider> FILE [options]: stream a file to a cloud recogniser at real-time pace, merge
its words into Arc's EXTERNAL stream, and print the words with Arc's utterance tags."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .base import MissingKey
from .registry import get


def _p(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else "-"


def overlap(u, v):
    return min(u["end"], v["end"]) - max(u["start"], v["start"])


def show(out, file=sys.stdout, top=3):
    utts = out.get("utterances") or []
    shown = set()
    T = out["transcript"]
    for i, u in enumerate(T):
        print(f"[{u['start']:7.2f}-{u['end']:7.2f}] {u['text']}", file=file)
        a = max(range(len(utts)), key=lambda j: overlap(u, utts[j]), default=None)
        if a is None or overlap(u, utts[a]) <= 0 or a in shown:
            continue
        nxt = T[i + 1] if i + 1 < len(T) else None             # one read-out per Arc utterance, after its last line
        if nxt is not None and max(range(len(utts)), key=lambda j: overlap(nxt, utts[j])) == a and overlap(nxt, utts[a]) > 0:
            continue
        shown.add(a)
        tags = utts[a].get("tags") or {}
        dd = tags.get("discourse.assistant.device_directed")
        groups = ", ".join(f"{g.split('.')[-1]} {x['label'].split('.')[-1]} ({x['p']:.2f})" for g, x in (utts[a].get("groups") or {}).items())
        rest = sorted(((t, p) for t, p in tags.items() if isinstance(p, (int, float)) and p >= 0.5
                       and t != "discourse.assistant.device_directed"), key=lambda x: -x[1])[:top]
        print(f"            arc [{utts[a]['start']:.1f}-{utts[a]['end']:.1f}]: directed {_p(dd)}"
              + (f", {groups}" if groups else "") + "".join(f", {t} {p:.2f}" for t, p in rest), file=file)
    for e in out.get("events") or []:
        if "command_complete" in str(e.get("tag", e.get("label", ""))):
            print(f"            end of turn at {e.get('released_t', e.get('t'))} s", file=file)
    a = out["asr"]
    print(f"\n{a['provider']}: {a['status']}" + (f" ({a['error']})" if a.get("error") else "")
          + (f", Arc's decoder from {a['fallback_from']:.2f} s" if a.get("fallback_from") is not None else "")
          + (f", time to final after the end of the audio {a['time_to_final_s']:.2f} s" if a.get("time_to_final_s") is not None else ""),
          file=sys.stderr)


def main(provider, argv=None):
    cls = get(provider)
    ap = argparse.ArgumentParser(prog=f"python -m arc_audio.integrations.{provider}",
                                 description=f"{provider} streaming words + Arc tags (Arc EXTERNAL stream). Key: {cls.key_env[0]}.")
    ap.add_argument("audio")
    ap.add_argument("--domain", default="unknown")
    ap.add_argument("--names", help="name list for linking (.txt one per line, or .json)")
    ap.add_argument("--language", default="en")
    ap.add_argument("--model", help="provider model (default: the adapter's)")
    ap.add_argument("--endpoint-ms", type=int, help="provider end-of-utterance silence (ms)")
    ap.add_argument("--speed", type=float, default=1.0, help="x real time (default 1; 0 = as fast as possible, not for assemblyai)")
    ap.add_argument("--timeout", type=float, default=5.0, help="seconds to wait for the last finals at the end (default 5)")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--no-fallback", action="store_true", help="on a provider failure, no words from Arc's own decoder")
    ap.add_argument("--backbone", default="mini")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--arc-model", default="WhissleAI/arc-v1.1")
    ap.add_argument("--json", action="store_true", help="the whole output as JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(name)s: %(message)s")
    from .base import install_log_scrubber
    install_log_scrubber(cls.key_env)
    try:
        cls(on_words=None).key
    except MissingKey as e:
        raise SystemExit(f"{provider}: {e}")
    try:
        import websockets  # noqa: F401
    except ImportError:
        raise SystemExit('needs websockets: pip install "arc-transcriber[integrations]"')
    from arc_audio import Arc, load_audio
    names = None
    if a.names:
        from arc_transcriber.cli import load_names
        names = load_names(a.names)
    arc = Arc.from_pretrained(a.arc_model, backbone=a.backbone, device=a.device, threads=a.threads)
    wav = load_audio(a.audio)
    from .session import run_file
    opts = dict(language=a.language, final_timeout_s=a.timeout, max_retries=a.retries)
    if a.model:
        opts["model"] = a.model
    if a.endpoint_ms is not None:
        opts["endpoint_ms"] = a.endpoint_ms
    out = asyncio.run(run_file(arc, wav, provider, speed=a.speed, domain=a.domain, names=names, fallback=not a.no_fallback, **opts))
    if a.json:
        json.dump(out, sys.stdout, indent=1, default=str)
        print()
    else:
        show(out)
