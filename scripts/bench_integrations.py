"""Live bench of the cloud streaming integrations (arc_audio.integrations): provider words alone vs Arc EXTERNAL output.

    # 1. a seeded LibriSpeech test-clean set (CC BY 4.0) and the same clips with pink noise at 10 dB SNR
    python scripts/bench_integrations.py make-set /data/LibriSpeech/test-clean OUT/set --n 50
    # 2. stream every clip to a provider at real-time pace (100 ms chunks), finalise at the end of the audio, and
    #    record every server message with its arrival time (needs the provider's key in the environment)
    python scripts/bench_integrations.py capture deepgram OUT/set OUT/capture --jobs 2
    # 3. replay the recordings into Arc EXTERNAL streams on the same clock (offline, no key), plus Arc's own stream
    #    on the same audio for the tag read-out
    python scripts/bench_integrations.py replay OUT/set OUT/capture OUT/replay --providers deepgram assemblyai speechmatics
    # 4. the table
    python scripts/bench_integrations.py report OUT/set OUT/capture OUT/replay

The replay sends each recorded message to the adapter's own parser (the code path of a live session) at the audio
time it arrived, so Arc sees the words exactly as late as it would have live. WER uses Whisper's English normaliser
(`pip install whisper-normalizer jiwer`).
"""
import argparse, asyncio, glob, json, os, random, sys, time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SR, CHUNK = 16000, 1600


# ---------------------------------------------------------------- 1. set
def make_set(a):
    import soundfile as sf
    trans = {}
    for t in glob.glob(os.path.join(a.src, "*/*/*.trans.txt")):
        for line in open(t):
            k, v = line.strip().split(" ", 1)
            trans[k] = v
    rng = random.Random(a.seed)
    pick, per = [], {}
    for k in rng.sample(sorted(trans), len(trans)):
        spk, ch = k.split("-")[:2]
        p = os.path.join(a.src, spk, ch, k + ".flac")
        d = sf.info(p).duration
        if not (2.0 <= d <= 20.0) or per.get(spk, 0) >= 3:
            continue
        per[spk] = per.get(spk, 0) + 1
        pick.append((k, p))
        if len(pick) == a.n:
            break
    for c in ("clean", "noisy"):
        os.makedirs(os.path.join(a.out, c), exist_ok=True)
    nrng = np.random.default_rng(a.seed)
    with open(os.path.join(a.out, "refs.tsv"), "w") as f:
        for k, p in sorted(pick):
            x, sr = sf.read(p, dtype="float32")
            assert sr == SR
            sf.write(os.path.join(a.out, "clean", k + ".wav"), x, SR, subtype="PCM_16")
            w = nrng.standard_normal(len(x) + 1)
            X = np.fft.rfft(w); fr = np.arange(len(X)); fr[0] = 1
            nz = np.fft.irfft(X / np.sqrt(fr), len(w))[:len(x)].astype(np.float32)
            nz *= np.sqrt(np.mean(x ** 2) / (np.mean(nz ** 2) * 10 ** (a.snr / 10)))
            sf.write(os.path.join(a.out, "noisy", k + ".wav"), np.clip(x + nz, -1, 1), SR, subtype="PCM_16")
            f.write(f"{k}\t{trans[k]}\n")
    print(len(pick), "clips")


def clips(setdir, conds=("clean", "noisy")):
    refs = dict(l.rstrip("\n").split("\t", 1) for l in open(os.path.join(setdir, "refs.tsv")))
    return refs, [(c, k, os.path.join(setdir, c, k + ".wav")) for c in conds for k in sorted(refs)]


# ---------------------------------------------------------------- 2. capture
async def capture_one(provider, path, out, opts):
    from arc_audio.audio import load_audio
    from arc_audio.integrations import get, Options
    wav = load_audio(path)
    recs = []
    cls = get(provider)
    keep = lambda r: (r["msg"].get("message") != "AudioAdded") and recs.append(r)     # drop per-frame acks
    asr = cls(on_words=lambda ev: None, record=keep, options=Options(**opts))
    meta = dict(provider=provider, file=os.path.basename(path), dur=len(wav) / SR)
    try:
        async with asr:
            t0 = time.monotonic()
            for k, i in enumerate(range(0, len(wav), CHUNK)):
                await asr.send(wav[i:i + CHUNK])
                await asyncio.sleep(max(0.0, t0 + (k + 1) * CHUNK / SR - time.monotonic()))
            meta["t_end"] = round(time.monotonic() - asr.t0, 4)
            await asr.finalize()
            await asr.close()
    except Exception as e:
        meta["exception"] = type(e).__name__
    meta.update(retries=asr.retries, failed=str(asr.failed) if asr.failed else None, info=asr.info)
    with open(out, "w") as f:
        f.write(json.dumps(dict(meta=meta)) + "\n")
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return meta


async def capture_all(a):
    _, items = clips(a.set)
    sem = asyncio.Semaphore(a.jobs)
    opts = dict(language=a.language)
    if a.model:
        opts["model"] = a.model

    async def run(c, k, p):
        out = os.path.join(a.out, a.provider, c, k + ".jsonl")
        if os.path.exists(out) and not a.force and not (a.redo_failed and json.loads(open(out).readline())["meta"].get("failed")):
            return
        os.makedirs(os.path.dirname(out), exist_ok=True)
        async with sem:
            m = await capture_one(a.provider, p, out, opts)
            print(a.provider, c, k, "failed" if m.get("failed") else "ok", flush=True)
    await asyncio.gather(*(run(*it) for it in items))


# ---------------------------------------------------------------- 3. replay into Arc
def load_capture(path):
    L = [json.loads(l) for l in open(path)]
    return L[0]["meta"], L[1:]


def provider_alone(provider, recs):
    """the provider's own final words, in arrival order (what an app showing only the provider would display), and the
    arrival time of the last final."""
    from arc_audio.integrations import get
    words, last = [], None
    asr = get(provider)(on_words=lambda ev: words.extend(w for w in ev if w["is_final"]))
    for r in recs:
        if r["kind"] != "rx":
            continue
        asr.offset = r.get("offset", 0.0)
        n = len(words)
        try:
            asr.handle(r["msg"])
        except Exception:
            continue
        if len(words) > n:
            last = r["t"]
    return words, last


def arc_replay(arc, provider, wav, recs, domain):
    """an Arc EXTERNAL stream fed the audio in 100 ms chunks and the recorded messages at their arrival times."""
    from arc_audio.integrations import get, wire
    from arc_audio.integrations.session import LIST_KEYS
    # fallback=None: no shadow decoder for names (it changes name records only, not words or tags), half the compute
    s = arc.stream(domain=domain, transcript="external", name_source="word", fallback=None, live=False)
    asr = wire(s, get(provider)(on_words=None))
    acc = {k: [] for k in LIST_KEYS}
    rx = [r for r in recs if r["kind"] == "rx"]
    j = 0

    def deliver(upto):
        nonlocal j
        while j < len(rx) and rx[j]["t"] <= upto:
            asr.offset = rx[j].get("offset", 0.0)
            try:
                asr.handle(rx[j]["msg"])
            except Exception:
                pass
            j += 1
    for i in range(0, len(wav), CHUNK):
        deliver(i / SR)
        o = s.feed(wav[i:i + CHUNK])
        for k in LIST_KEYS:
            acc[k] += o.get(k) or []
    deliver(1e9)
    s.words.mark_covered(len(wav) / SR + 10)
    o = s.flush()
    for k in LIST_KEYS:
        acc[k] += o.get(k) or []
    s.close()
    return acc


def arc_own(arc, wav, domain):
    from arc_audio.integrations.session import LIST_KEYS
    s = arc.stream(domain=domain, transcript="builtin", partials=False, live=False)   # finals and tags only
    acc = {k: [] for k in LIST_KEYS}
    for i in range(0, len(wav), CHUNK):
        o = s.feed(wav[i:i + CHUNK])
        for k in LIST_KEYS:
            acc[k] += o.get(k) or []
    o = s.flush()
    for k in LIST_KEYS:
        acc[k] += o.get(k) or []
    s.close()
    return acc


def tags_of(acc):
    """the utterance tag read-out: per Arc utterance, its time span and tags (label -> p)."""
    return [dict(start=u.get("start"), end=u.get("end"), tags=u.get("tags") or {}, groups=u.get("groups") or {})
            for u in acc.get("utterances") or []]


def replay_all(a):
    from arc_audio import Arc
    from arc_audio.audio import load_audio
    import torch
    torch.set_num_threads(a.threads)
    arc = Arc.from_pretrained(a.model, backbone="mini", device="cpu", threads=a.threads)
    _, items = clips(a.set)
    items = items[a.shard::a.shards]
    for c, k, p in items:
        wav = None
        own = os.path.join(a.out, "arc", c, k + ".json")
        if a.own and not os.path.exists(own):
            wav = load_audio(p)
            acc = arc_own(arc, wav, a.domain)
            os.makedirs(os.path.dirname(own), exist_ok=True)
            json.dump(dict(text=" ".join(u["text"] for u in sorted(acc["transcript"], key=lambda u: u["start"])),
                           tags=tags_of(acc)), open(own, "w"))
        for prov in a.providers:
            cap = os.path.join(a.capture, prov, c, k + ".jsonl")
            out = os.path.join(a.out, prov, c, k + ".json")
            if not os.path.exists(cap) or os.path.exists(out):
                continue
            wav = load_audio(p) if wav is None else wav
            meta, recs = load_capture(cap)
            alone, last = provider_alone(prov, recs)
            acc = arc_replay(arc, prov, wav, recs, a.domain)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            json.dump(dict(meta=meta, provider_text=" ".join(w["text"] for w in alone),
                           merged_text=" ".join(u["text"] for u in sorted(acc["transcript"], key=lambda u: u["start"])),
                           n_provider_words=len(alone), n_utts=len(acc["transcript"]),
                           time_to_final=None if last is None or "t_end" not in meta else round(max(0.0, last - meta["t_end"]), 4),
                           tags=tags_of(acc)), open(out, "w"))
            print(prov, c, k, flush=True)


# ---------------------------------------------------------------- 4. report
def tag_parity(A, B, thr=0.5):
    """utterances of two read-outs paired by time overlap: how many have the same tags at p >= thr and the same label in
    every group (act, intent, ...), and the largest |p difference| over all tags and group labels."""
    same = n = 0
    dmax = 0.0
    for u in A:
        best = max(B, key=lambda v: min(u["end"], v["end"]) - max(u["start"], v["start"]), default=None)
        if best is None or min(u["end"], best["end"]) <= max(u["start"], best["start"]):
            n += 1
            continue
        n += 1
        ta = {t for t, p in u["tags"].items() if (p if isinstance(p, (int, float)) else 0) >= thr}
        tb = {t for t, p in best["tags"].items() if (p if isinstance(p, (int, float)) else 0) >= thr}
        ga = {g: x["label"] for g, x in (u.get("groups") or {}).items()}
        gb = {g: x["label"] for g, x in (best.get("groups") or {}).items()}
        same += ta == tb and ga == gb
        for g in set(ga) & set(gb):
            if ga[g] == gb[g]:
                dmax = max(dmax, abs(u["groups"][g]["p"] - best["groups"][g]["p"]))
        for t in set(u["tags"]) | set(best["tags"]):
            pa, pb = u["tags"].get(t, 0.0), best["tags"].get(t, 0.0)
            if isinstance(pa, (int, float)) and isinstance(pb, (int, float)):
                dmax = max(dmax, abs(pa - pb))
    return same, n, dmax


def report(a):
    import jiwer
    from whisper_normalizer.english import EnglishTextNormalizer
    norm = EnglishTextNormalizer()
    refs, _ = clips(a.set)
    rows = []
    for prov in a.providers:
        for c in ("clean", "noisy"):
            R, P, M, ttf, same, n, dmax, nfail, eq = [], [], [], [], 0, 0, 0.0, 0, 0
            for k in sorted(refs):
                f = os.path.join(a.replay, prov, c, k + ".json")
                if not os.path.exists(f):
                    continue
                j = json.load(open(f))
                if j["meta"].get("failed"):
                    nfail += 1
                r = norm(refs[k])
                if not r.strip():
                    continue
                R.append(r); P.append(norm(j["provider_text"])); M.append(norm(j["merged_text"]))
                eq += P[-1] == M[-1]
                if j["time_to_final"] is not None:
                    ttf.append(j["time_to_final"])
                own = os.path.join(a.replay, "arc", c, k + ".json")
                if os.path.exists(own):
                    s_, n_, d_ = tag_parity(j["tags"], json.load(open(own))["tags"])
                    same += s_; n += n_; dmax = max(dmax, d_)
            if not R:
                continue
            rows.append(dict(provider=prov, cond=c, n=len(R), wer_provider=100 * jiwer.wer(R, P), wer_merged=100 * jiwer.wer(R, M),
                             same_text=eq, ttf_p50=float(np.percentile(ttf, 50)) if ttf else None,
                             ttf_p90=float(np.percentile(ttf, 90)) if ttf else None, tag_same=same, tag_n=n, tag_dmax=dmax,
                             failed=nfail))
    print("| provider | audio | clips | WER provider | WER Arc EXTERNAL | same text | time to final p50 / p90 | utterance tags as Arc's own | max abs tag p diff |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        t = f"{r['ttf_p50']:.2f} / {r['ttf_p90']:.2f} s" if r["ttf_p50"] is not None else "-"
        print(f"| {r['provider']} | {r['cond']} | {r['n']} | {r['wer_provider']:.2f} % | {r['wer_merged']:.2f} % | {r['same_text']}/{r['n']} | {t} | "
              f"{r['tag_same']}/{r['tag_n']} | {r['tag_dmax']:.3f} |")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-set"); p.add_argument("src"); p.add_argument("out"); p.add_argument("--n", type=int, default=50)
    p.add_argument("--snr", type=float, default=10.0); p.add_argument("--seed", type=int, default=1234)
    p = sub.add_parser("capture"); p.add_argument("provider"); p.add_argument("set"); p.add_argument("out")
    p.add_argument("--jobs", type=int, default=2); p.add_argument("--language", default="en"); p.add_argument("--model")
    p.add_argument("--force", action="store_true"); p.add_argument("--redo-failed", action="store_true")
    p = sub.add_parser("replay"); p.add_argument("set"); p.add_argument("capture"); p.add_argument("out")
    p.add_argument("--providers", nargs="+", default=["deepgram", "assemblyai", "speechmatics"])
    p.add_argument("--domain", default="unknown"); p.add_argument("--threads", type=int, default=2)
    p.add_argument("--model", default="WhissleAI/arc-v1.1"); p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--no-own", dest="own", action="store_false", help="skip Arc's own stream (the tag reference)")
    p = sub.add_parser("report"); p.add_argument("set"); p.add_argument("capture"); p.add_argument("replay")
    p.add_argument("--providers", nargs="+", default=["deepgram", "assemblyai", "speechmatics"]); p.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "make-set":
        make_set(a)
    elif a.cmd == "capture":
        asyncio.run(capture_all(a))
    elif a.cmd == "replay":
        replay_all(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
