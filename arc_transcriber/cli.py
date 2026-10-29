"""arc-transcribe: transcribe a file (or live audio) with Arc: transcript with inline name / number tags, names linked to your
list, numbers with values, speakers, tags, sound events and device-directed speech, from one streaming model.

    arc-transcribe clip.wav --domain nba --names roster.txt                       # Arc Mini (default), offline
    arc-transcribe clip.wav --backbone full --device cuda                         # Arc Transcriber (full, more accurate)
    arc-transcribe clip.wav --context terms.txt --topic meeting                   # context: names / terms / topic for the decoder
    arc-transcribe clip.wav --asr deepgram --format srt -o clip.srt               # Deepgram words (DEEPGRAM_API_KEY)
    arc-transcribe clip.wav --asr verbit --live                                   # Verbit live order (VERBIT_WS_URL, VERBIT_API_KEY)
    arc-transcribe clip.wav --live                                                # stream the file at real-time pace
    arc-transcribe --mic --domain talk                                            # microphone (pip install sounddevice)
    arc-transcribe clip.wav --asr deepgram --demo                                 # no key: simulated Deepgram messages

The model (WhissleAI/arc-v1.1, pinned revision) is downloaded from Hugging Face on first use; the full backbone also downloads
nvidia/parakeet-tdt-0.6b-v2 (CC-BY-4.0, about 2.5 GB) on first use.
"""
import argparse, asyncio, json, os, sys, time

from . import DEFAULT_MODEL, DEFAULT_REVISION, __version__
from .output import render, _fmt_value

SR, CHUNK = 16000, 1600


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_names(path):
    """names file: .json (a list of strings or {"name", "kind", "aliases", "group"} dicts) or text (one name per line; optional
    'name<TAB>kind')."""
    if not path:
        return None
    if path.lower().endswith(".json"):
        d = json.load(open(path, encoding="utf-8"))
        return d.get("names", d) if isinstance(d, dict) else d
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        out.append(dict(name=parts[0].strip(), kind=parts[1].strip()) if len(parts) > 1 and parts[1].strip() else parts[0].strip())
    return out


def load_arc(a):
    from arc_audio import Arc
    t = time.time()
    rev = a.revision if not os.path.isdir(a.model) else None
    try:
        arc = Arc.from_pretrained(a.model, backbone=a.backbone, revision=rev, device=a.device, threads=a.threads)
    except Exception as e:                                   # gated repo / no token: say what to do
        if type(e).__name__ in ("GatedRepoError", "RepositoryNotFoundError") or "401" in str(e):
            raise SystemExit(f"arc-transcribe: cannot download {a.model}: accept the model's terms on https://huggingface.co/{a.model} "
                             "while logged in, then run `hf auth login` (or set HF_TOKEN)")
        raise
    log(f"arc-transcribe {__version__}: model {a.model}{'@' + rev if rev else ''} ({a.backbone}) on {a.device} ({time.time() - t:.1f} s)")
    if a.context or a.topic or a.auto_context:
        arc.set_context(terms=load_terms(a.context), topic=a.topic, auto_prev=a.auto_context)
    return arc


def load_terms(spec):
    """--context: a .txt file (one term per line), a .json list, or a comma-separated list of names / terms / phrases"""
    if not spec:
        return None
    if os.path.exists(spec):
        if spec.lower().endswith(".json"):
            return [str(x) for x in json.load(open(spec, encoding="utf-8"))]
        return [l.strip() for l in open(spec, encoding="utf-8") if l.strip() and not l.startswith("#")]
    return [t.strip() for t in spec.split(",") if t.strip()]


def show(new, quiet):
    if quiet:
        return
    for u in new.get("transcript") or []:
        log(f"[{u['start']:7.2f}] {u['text']}")
    for n in new.get("names") or []:
        log(f"          name   {n['heard']!r} -> {n['entity']} ({n['state']}{', ' + n['via'] if n.get('via') else ''})")
    for n in new.get("numbers") or []:
        log(f"          number {n['heard']!r} = {_fmt_value(n)} ({n.get('kind')})")


# ---------------------------------------------------------------- offline (whole file)
def run_offline(arc, a, wav, names):
    if a.asr == "builtin":
        return arc.predict(wav, domain=a.domain, names=names, transcript="builtin", return_frames=False, delay_s=a.delay, events=a.events,
                           kb_snap=not a.no_snap)
    from . import asr as X
    if a.demo:
        words = [dict(text=w["text"], start=w["start"], end=w["end"], is_final=True) for u in arc.transcribe(wav) for w in u["words"]]
        log(f"--demo: {len(words)} words from Arc's built-in transcript stand in for {a.asr}")
    elif a.asr == "deepgram":
        words = X.deepgram_file(wav, X.env_key("DEEPGRAM_API_KEY"))
        log(f"deepgram: {len(words)} words")
    else:
        raise SystemExit("verbit: live captions only (run at real-time pace); use --demo for the offline simulation")
    return arc.predict(wav, domain=a.domain, names=names, transcript="external", words=words, name_source="word",
                       strict=a.strict, return_frames=False, delay_s=a.delay, events=a.events, kb_snap=not a.no_snap)


# ---------------------------------------------------------------- live (file at real-time pace, or microphone)
async def file_chunks(wav, pace):
    t0 = time.monotonic()
    for k, i in enumerate(range(0, len(wav), CHUNK)):
        yield wav[i:i + CHUNK]
        if pace:
            await asyncio.sleep(max(0.0, t0 + (k + 1) * CHUNK / SR - time.monotonic()))
        else:
            await asyncio.sleep(0)


async def mic_chunks(stop, duration):
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("--mic needs sounddevice: pip install sounddevice")
    import numpy as np
    loop = asyncio.get_running_loop(); q = asyncio.Queue()
    cb = lambda data, frames, t, status: loop.call_soon_threadsafe(q.put_nowait, np.asarray(data[:, 0], np.float32).copy())
    t0 = time.monotonic()
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=CHUNK, callback=cb):
        log("listening on the default microphone (Ctrl-C to stop)")
        while not stop.is_set() and (not duration or time.monotonic() - t0 < duration):
            try:
                yield await asyncio.wait_for(q.get(), 0.5)
            except asyncio.TimeoutError:
                continue


def need_live_deps(a):
    """--live / --mic with a real external ASR needs websockets; --mic needs sounddevice"""
    import importlib.util
    miss = []
    if a.asr != "builtin" and not a.demo and importlib.util.find_spec("websockets") is None:
        miss.append("websockets")
    if a.mic and importlib.util.find_spec("sounddevice") is None:
        miss.append("sounddevice")
    if miss:
        raise SystemExit(f"arc-transcribe: live mode needs {' and '.join(miss)}: pip install 'arc-transcriber[live]'")


async def run_live(arc, a, wav, names):
    from . import asr as X
    ext_mode = a.asr != "builtin"
    s = arc.stream(domain=a.domain, names=names, delay_s=a.delay, transcript="external" if ext_mode else "builtin",
                   name_source="word" if ext_mode else None, interim_source=a.asr if ext_mode else None, strict=a.strict,
                   events=a.events, kb_snap=not a.no_snap)
    acc = dict(names=[], numbers=[], transcript=[], diarization=[], events=[], utterances=[], audio_events=[], device_directed=[])

    def add(new):
        for k in acc:
            acc[k] += new.get(k) or []
        show(new, a.quiet)
    stop = asyncio.Event()
    try:
        asyncio.get_running_loop().add_signal_handler(__import__("signal").SIGINT, stop.set)
    except (NotImplementedError, RuntimeError):
        pass
    on_words = lambda ev: s.add_words(ev)
    ext = sim = None
    if ext_mode and a.demo:
        if wav is None:
            raise SystemExit("--demo needs an audio file")
        sim = X.Simulated(a.asr, [w for u in arc.transcribe(wav) for w in u["words"]], on_words)
    elif a.asr == "deepgram":
        ext = X.DeepgramLive(X.env_key("DEEPGRAM_API_KEY"), on_words)
    elif a.asr == "verbit":
        url, tok = X.env_key("VERBIT_WS_URL", "VERBIT_API_KEY")
        ext = X.VerbitLive(url, tok, on_words)
    src = mic_chunks(stop, a.duration) if a.mic else file_chunks(wav, pace=not (a.demo or a.fast))
    n = 0

    async def loop_audio():
        nonlocal n
        async for chunk in src:
            if stop.is_set():
                break
            if ext is not None:
                await ext.send(chunk)
            n += len(chunk)
            if sim is not None:
                sim.advance(n / SR)
            add(s.feed(chunk))
    if ext is not None:
        async with ext:
            await loop_audio()
            await ext.close()
    else:
        await loop_audio()
    if sim is not None:
        sim.advance(n / SR, final=True)
    final = s.flush(); add(final)
    out = dict(acc, speakers=final.get("speakers"), item=final.get("item"), domain=a.domain,
               name_source="word" if ext_mode else s.nsrc, latency=s.latency_report())
    if ext_mode and not acc["transcript"]:
        with s.words.lock:
            out["transcript"] = [dict(start=w["start"], end=w["end"], text=w["w"]) for w in s.words.finals]
    elif ext_mode:
        out["transcript"] = [dict(start=u["start"], end=u["end"], text=u["text"], words=u.get("words", []), source=u.get("source"),
                                  **({"device_directed": u["device_directed"]} if u.get("device_directed") else {})) for u in acc["transcript"]]
    else:
        out["transcript"] = [dict(start=u["start"], end=u["end"], text=u["text"], words=u.get("words", [])) for u in acc["transcript"]]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="arc-transcribe", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("audio", nargs="?", help="audio file (any format soundfile reads; resampled to 16 kHz mono)")
    ap.add_argument("--domain", default="unknown", help="nba nhl nfl cricket soccer esports news talk stream_chat meeting court_appellate "
                                                         "court_trial assistant assistant_noisy short_clips unknown (default)")
    ap.add_argument("--names", help="your name list: .txt (one per line, optional TAB kind) or .json")
    ap.add_argument("--backbone", choices=["mini", "full", "edge", "parakeet", "server"], default="mini",
                    help="mini (default: Arc Mini, small, on-device) or full (Arc Transcriber: Parakeet encoder, most accurate, GPU "
                         "recommended); edge = mini and parakeet / server = full are kept as aliases")
    ap.add_argument("--context", help="names / terms / phrases the decoder should expect: .txt (one per line), .json list, or 'a,b,c'")
    ap.add_argument("--topic", help="short topic / domain hint for the decoder context (e.g. 'meeting', 'nba', 'customer support')")
    ap.add_argument("--auto-context", action="store_true", help="feed the text of earlier utterances back as decoder context (memory)")
    ap.add_argument("--events", choices=["default", "all", "none"], default="default",
                    help="sound / channel events: default = stable + beta labels, all = include experimental labels, none = off")
    ap.add_argument("--no-snap", action="store_true", help="turn off the name-list snap post-pass (fuzzy / phonetic match to --names)")
    ap.add_argument("--asr", choices=["builtin", "deepgram", "verbit"], default="builtin", help="transcript words (default: builtin)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--format", choices=["json", "txt", "srt", "vtt"], default="txt")
    ap.add_argument("-o", "--output", help="output file (default: stdout)")
    ap.add_argument("--live", action="store_true", help="stream the file through Arc at real-time pace (100 ms chunks)")
    ap.add_argument("--mic", action="store_true", help="live from the default microphone (needs sounddevice)")
    ap.add_argument("--duration", type=float, default=0, help="--mic: stop after this many seconds (default: Ctrl-C)")
    ap.add_argument("--fast", action="store_true", help="--live: feed the file as fast as possible instead of real time")
    ap.add_argument("--demo", action="store_true", help="--asr deepgram|verbit without a key: simulated messages, no network")
    ap.add_argument("--strict", action="store_true", help="external words: no audio fallback for names (half the compute)")
    ap.add_argument("--delay", type=float, default=0.3, help="label delay in s (0, 0.1, 0.2, 0.3, 0.5, 0.8)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face repo or local folder (default {DEFAULT_MODEL})")
    ap.add_argument("--revision", default=DEFAULT_REVISION, help=f"model revision (default {DEFAULT_REVISION})")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads")
    ap.add_argument("-q", "--quiet", action="store_true", help="no live progress on stderr")
    ap.add_argument("--version", action="version", version=f"arc-transcribe {__version__}")
    a = ap.parse_args(argv)
    if not a.audio and not a.mic:
        ap.error("give an audio file or --mic")
    if a.mic and a.demo:
        ap.error("--demo simulates the ASR from a file; it cannot be used with --mic")
    if a.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            ap.error("--device cuda: no CUDA device visible")
    names = load_names(a.names)
    if a.mic or a.live or (a.asr == "verbit" and not a.demo):
        need_live_deps(a)
    arc = load_arc(a)
    from arc_audio import load_audio
    wav = load_audio(a.audio) if a.audio else None
    from .asr import MissingKey
    try:
        if a.mic or a.live or (a.asr == "verbit" and not a.demo):
            out = asyncio.run(run_live(arc, a, wav, names))
        else:
            out = run_offline(arc, a, wav, names)
    except MissingKey as e:
        raise SystemExit(f"arc-transcribe: {e}")
    text = render(out, a.format)
    if a.output:
        open(a.output, "w", encoding="utf-8").write(text)
        log(f"wrote {a.output} ({a.format}; {len(out.get('names') or [])} names, {len(out.get('numbers') or [])} numbers)")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
