"""Shared helpers for the evals: loading Arc, rendering test speech, rooms and channels, scoring.

Every eval renders its own test audio from the text lists in its folder, so no audio is committed. The renderer is
macOS `say` (system voices, written straight to a file; nothing is played) converted with `afconvert` to 16 kHz mono.
On other systems pass `--clips DIR` to an eval with your own renders (the file names are the ones each script passes to `render()`), or
replace `render()` below with another TTS.
"""
import json
import os
import platform
import re
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "_work")   # rendered clips (git-ignored)

# The four macOS voices the Steer book bench used (US, UK, US, Indian English).
VOICES = ["Samantha", "Daniel", "Flo", "Aman"]


# ---------------------------------------------------------------- Arc

def load_arc(threads=4, revision=None, backbone="mini"):
    from arc_audio import Arc
    arc = Arc.from_pretrained("WhissleAI/arc-v1.1", backbone=backbone, device="cpu", threads=threads, revision=revision)
    arc.eval_revision = revision
    return arc


def env_info(arc=None, threads=None):
    import arc_audio
    import torch
    cpu = platform.processor()
    if sys.platform == "darwin":
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        except OSError:
            pass
    info = dict(date=time.strftime("%Y-%m-%d"), arc_audio=arc_audio.__version__, torch=torch.__version__,
                python=platform.python_version(), platform=platform.platform(), cpu=cpu, device="cpu", threads=threads,
                runtime="Python reference (arc_audio)")
    if arc is not None:
        info["model_dir"] = re.sub(r"^.*/hub/", "<hf cache>/", str(getattr(arc, "_mp", "")))
        info["backbone"] = getattr(arc, "backbone", None)
        rev = re.search(r"snapshots/([0-9a-f]{7,40})", str(info["model_dir"] or ""))
        info["model_snapshot"] = rev.group(1)[:12] if rev else None
        info["model_revision"] = getattr(arc, "eval_revision", None) or getattr(arc_audio, "MODEL_REVISION", f"v{arc_audio.__version__}")   # default: the package's
    return info


def text_of(utts):
    return " ".join(u.get("text", "") for u in utts).strip()


# ---------------------------------------------------------------- speech

def render(text, voice, path, rate=None):
    """Render `text` with macOS `say` to a 16 kHz mono 16-bit WAV at `path` (cached)."""
    if os.path.exists(path):
        return path
    if sys.platform != "darwin":
        raise SystemExit(f"rendering needs macOS `say`; supply clips with --clips (missing {path})")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    aiff = path[:-4] + ".aiff"
    cmd = ["say", "-v", voice, "-o", aiff]
    if rate:
        cmd += ["-r", str(rate)]
    subprocess.run(cmd + ["--", text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff, path], check=True)
    os.remove(aiff)
    return path


def load(path):
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    a = a.mean(1)
    if sr != SR:
        raise ValueError(f"{path}: {sr} Hz, expected {SR}")
    return a


def slug(s, n=40):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:n]


def voiced_db(x, frame=320):
    """Level of the voiced part (20 ms frames within 20 dB of the loudest), dBFS. As Steer's Dialogue.asked."""
    n = len(x) // frame
    if n == 0:
        return -120.0
    e = 10 * np.log10(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    v = e[e > e.max() - 20]
    return float(10 * np.log10(np.mean(10 ** (v / 10))))


def at_level(x, db):
    return (x * 10 ** ((db - voiced_db(x)) / 20)).astype(np.float32)


def silence(s):
    return np.zeros(int(s * SR), np.float32)


# ---------------------------------------------------------------- rooms and channels (numpy only)

def _fir_lowpass(fc, taps=129):
    n = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(2 * fc / SR * n) * np.hamming(taps)
    return h / h.sum()


def lowpass(x, fc):
    return np.convolve(x, _fir_lowpass(fc), mode="same").astype(np.float32)


def bandpass(x, lo, hi):
    return (lowpass(x, hi) - lowpass(x, lo)).astype(np.float32)


def room(speech, db, level=0.35, before=0.6, after=2.0, seed=0):
    """Steer's voice-bench room (SteerVoiceBench `room`): the speech peak-normalised to `level`, `before`/`after` s of
    room, and steady slightly low-passed noise at about `db` dBFS (-60 quiet, -38 a fan or a cafe, -28 a loud room)."""
    rng = np.random.default_rng(seed)
    peak = float(np.max(np.abs(speech))) or 1.0
    s = speech / max(peak, 1e-4) * level
    x = np.concatenate([silence(before), s, silence(after)])
    amp = 10 ** (db / 20) * 1.7
    w = rng.uniform(-1, 1, len(x)).astype(np.float32)
    lp = np.convolve(w, 0.3 * 0.7 ** np.arange(64))[: len(w)]   # one-pole low-pass y = 0.7 y' + 0.3 x, as the Swift bench
    return (x + lp * amp).astype(np.float32)


def band_noise(x, snr_db, seed=7):
    """Band noise (200-4000 Hz) at `snr_db` against the speech (arcsig scenarios `noise`)."""
    rng = np.random.default_rng(seed)
    n = bandpass(rng.standard_normal(len(x)).astype(np.float32), 200, 4000)
    v = x[np.abs(x) > 1e-3]
    ps = np.mean(v ** 2) if len(v) else 1e-6
    return (x + n * np.sqrt(ps / np.mean(n ** 2) / 10 ** (snr_db / 10))).astype(np.float32)


def phone(x):
    """Telephone channel: 300-3400 Hz, through 8 kHz, 8-bit mu-law."""
    b = bandpass(x, 300, 3400)
    b = np.repeat(b[::2], 2)[: len(b)]
    q = np.sign(b) * np.log1p(255 * np.abs(np.clip(b, -1, 1))) / np.log(256)
    q = np.round(q * 127) / 127
    return (np.sign(q) * (256 ** np.abs(q) - 1) / 255).astype(np.float32)


def clipped(x):
    return np.clip(x * 8, -0.3, 0.3).astype(np.float32)


def mumbled(x):
    return (lowpass(x, 900) * 0.15).astype(np.float32)


# ---------------------------------------------------------------- scoring

def words(s):
    s = s.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9' ]+", " ", s).split()


def edits(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p = d[:]
        d[0] = i
        for j in range(1, len(h) + 1):
            d[j] = min(p[j] + 1, d[j - 1] + 1, p[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)


def wer(pairs):
    e = n = 0
    for ref, hyp in pairs:
        a, b = edits(ref, hyp)
        e += a
        n += b
    return e / max(1, n)


def contains(text, phrase):
    return f" {' '.join(words(phrase))} " in f" {' '.join(words(text))} "


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    s = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return s / (len(pos) * len(neg))


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else float("nan")


def read_tsv(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line and not line.startswith("#"):
            rows.append(line.split("\t"))
    return rows


def save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
        f.write("\n")


def directed_of(out):
    """Both device-directed numbers of a predict() output, per utterance: the utterance tag
    (`utterances[].tags['discourse.assistant.device_directed']`) and the transcript's separate head
    (`transcript[].device_directed.p`)."""
    tag = [u.get("tags", {}).get("discourse.assistant.device_directed") for u in out.get("utterances") or []]
    head = [(u.get("device_directed") or {}).get("p") for u in out.get("transcript") or []]
    return [t for t in tag if t is not None], [h for h in head if h is not None]


def act_of(out):
    for u in out.get("utterances") or []:
        g = (u.get("groups") or {}).get("discourse.act")
        if g:
            return g["label"].replace("discourse.act.", ""), g["p"]
    return None, None


def intent_of(out):
    for u in out.get("utterances") or []:
        g = (u.get("groups") or {}).get("discourse.intent")
        if g:
            return g["label"].replace("discourse.intent.", ""), g["p"]
    return None, None


# ---------------------------------------------------------------- the concept-film rooms (healthcare)

def pink(n, seed=0):
    w = np.random.default_rng(seed).standard_normal(n)
    X = np.fft.rfft(w)
    f = np.arange(len(X), dtype=float)
    f[0] = 1
    y = np.fft.irfft(X / np.sqrt(f), n)
    return (y / np.sqrt((y ** 2).mean())).astype(np.float32)


def ward(n, seed=0):
    """A quiet ward at night: low air handling (-58 dBFS) and a far-off monitor beep."""
    t = np.arange(n) / SR
    y = 10 ** (-58 / 20) * pink(n, seed)
    L = int(0.12 * SR)
    for k in np.arange(1.1, n / SR, 2.4):
        i = int(k * SR)
        if i + L < n:
            y[i:i + L] += 0.0016 * np.sin(2 * np.pi * 660 * t[:L]) * np.hanning(L)
    return y.astype(np.float32)


def theatre(n, seed=0):
    """An operating room: laminar air flow (-40 dBFS) and a pulse-oximeter beep."""
    t = np.arange(n) / SR
    y = 10 ** (-40 / 20) * pink(n, seed)
    L = int(0.09 * SR)
    for k in np.arange(0.3, n / SR, 1.05):
        i = int(k * SR)
        if i + L < n:
            y[i:i + L] += 0.012 * np.sin(2 * np.pi * 880 * t[:L]) * np.hanning(L)
    return y.astype(np.float32)


def in_room(speech, room_fn, level_db, before=0.5, after=0.8, seed=0):
    """The speech at `level_db` (voiced level, dBFS) in a room."""
    x = np.concatenate([silence(before), at_level(speech, level_db), silence(after)])
    return (x + room_fn(len(x), seed)).astype(np.float32)
