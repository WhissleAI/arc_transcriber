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
    # 3b / 4b. the same replay through libarc's EXTERNAL stream (c/, `make -C c`), much faster than Python:
    python scripts/bench_integrations.py export-words OUT/set OUT/capture OUT/words      # arc-stream --words-dir lines
    python scripts/bench_integrations.py c-run OUT/set OUT/words OUT/c --mini /path/arc-v1.1/mini
    python scripts/bench_integrations.py c-report OUT/set OUT/capture OUT/c

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
        if os.path.exists(out) and not a.force and not (a.redo_failed and any(json.loads(open(out).readline())["meta"].get(x) for x in ("failed", "exception"))):
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
DD = "discourse.assistant.device_directed"


def tag_parity(A, B, thr=0.5):
    """Arc utterances of two read-outs (EXTERNAL vs Arc's own transcript) paired by time overlap. Arc's trunk also reads
    the words it is given (the word channel), so the read-outs are close but not identical. Returns per pair: the same
    act, the same intent, the same tags at p >= thr, |device-directed p difference|."""
    out = []
    for u in A:
        v = max(B, key=lambda v: min(u["end"], v["end"]) - max(u["start"], v["start"]), default=None)
        if v is None or min(u["end"], v["end"]) <= max(u["start"], v["start"]):
            out.append(None)
            continue
        g = lambda x, k: ((x.get("groups") or {}).get(k) or {}).get("label")
        ts = lambda x: {t for t, p in x["tags"].items() if isinstance(p, (int, float)) and p >= thr}
        out.append(dict(act=g(u, "discourse.act") == g(v, "discourse.act"), intent=g(u, "discourse.intent") == g(v, "discourse.intent"),
                        tags=ts(u) == ts(v), dd=abs(float(u["tags"].get(DD, 0.0)) - float(v["tags"].get(DD, 0.0)))))
    return out


def report(a):
    import jiwer
    from whisper_normalizer.english import EnglishTextNormalizer
    norm = EnglishTextNormalizer()
    refs, _ = clips(a.set)
    rows = []
    for prov in a.providers:
        for c in ("clean", "noisy"):
            R, P, M, ttf, pairs, nfail, eq = [], [], [], [], [], 0, 0
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
                    pairs += tag_parity(j["tags"], json.load(open(own))["tags"])
            if not R:
                continue
            pp = [x for x in pairs if x]
            pct = lambda key: 100 * sum(x[key] for x in pp) / len(pp) if pp else float("nan")
            rows.append(dict(provider=prov, cond=c, n=len(R), wer_provider=100 * jiwer.wer(R, P), wer_merged=100 * jiwer.wer(R, M),
                             same_text=eq, ttf_p50=float(np.percentile(ttf, 50)) if ttf else None,
                             ttf_p90=float(np.percentile(ttf, 90)) if ttf else None, n_utts=len(pairs), unpaired=len(pairs) - len(pp),
                             act=pct("act"), intent=pct("intent"), tags=pct("tags"),
                             dd_mean=float(np.mean([x["dd"] for x in pp])) if pp else float("nan"),
                             dd_p90=float(np.percentile([x["dd"] for x in pp], 90)) if pp else float("nan"), failed=nfail))
    print("| provider | audio | clips | WER provider | WER Arc EXTERNAL | same text | time to final p50 / p90 |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        t = f"{r['ttf_p50']:.2f} / {r['ttf_p90']:.2f} s" if r["ttf_p50"] is not None else "-"
        print(f"| {r['provider']} | {r['cond']} | {r['n']} | {r['wer_provider']:.2f} % | {r['wer_merged']:.2f} % | {r['same_text']}/{r['n']} | {t} |")
    print()
    print("| provider | audio | Arc utterances | same act | same intent | same tags (p >= 0.5) | device-directed abs diff mean / p90 |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['provider']} | {r['cond']} | {r['n_utts']} | {r['act']:.0f} % | {r['intent']:.0f} % | {r['tags']:.0f} % | "
              f"{r['dd_mean']:.3f} / {r['dd_p90']:.3f} |")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


# ---------------------------------------------------------------- 3b / 4b. libarc (c/tools/arc_stream.c --mode external)
def export_words(a):
    """each capture -> the arc-stream replay lines {"at", "final", "words", "upto"}, through the adapter's own parser."""
    from arc_audio.integrations import get
    _, items = clips(a.set)
    for prov in a.providers:
        for c, k, p in items:
            cap = os.path.join(a.capture, prov, c, k + ".jsonl")
            if not os.path.exists(cap):
                continue
            meta, recs = load_capture(cap)
            lines, cur = [], {}
            asr = get(prov)(on_words=lambda ev: cur.setdefault("w", []).extend(ev),
                            on_covered=lambda t: cur.__setitem__("upto", t))
            for r in recs:
                if r["kind"] != "rx":
                    continue
                cur.clear()
                asr.offset = r.get("offset", 0.0)
                try:
                    asr.handle(r["msg"])
                except Exception:
                    continue
                W = cur.get("w", [])
                fin = [dict(text=w["text"], start=w["start"], end=w["end"]) for w in W if w["is_final"]]
                inter = [dict(text=w["text"], start=w["start"], end=w["end"]) for w in W if not w["is_final"]]
                if fin or "upto" in cur:
                    lines.append(dict(at=r["t"], final=1, words=fin, **({"upto": cur["upto"]} if "upto" in cur else {})))
                if W:                                                  # the interim now (empty: none left)
                    lines.append(dict(at=r["t"], final=0, words=inter))
            d = os.path.join(a.out, prov, c)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, k + ".jsonl"), "w") as f:
                for l in lines:
                    f.write(json.dumps(l) + "\n")


def c_run(a):
    """arc-stream over every clip: Arc's own stream (--mode full, the tag reference) and one EXTERNAL stream per
    provider; --jobs processes of one file list each, outputs joined per provider and condition."""
    import subprocess
    _, items = clips(a.set)
    exe = a.exe or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "c", "build", "arc-stream")
    for mode, prov in [("full", "arc")] + [("external", p) for p in a.providers]:
        for c in ("clean", "noisy"):
            out = os.path.join(a.out, prov, c + ".jsonl")
            if os.path.exists(out) and not a.force:
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            wavs = [p for cc, k, p in items if cc == c]
            procs = []
            for j in range(a.jobs):
                part = wavs[j::a.jobs]
                cmd = [exe, a.mini] + part + ["--mode", mode, "--threads", str(a.threads), "--records", "--domain", a.domain]
                if mode == "external":
                    cmd += ["--words-dir", os.path.join(a.words, prov, c)]
                procs.append((subprocess.Popen(cmd, stdout=open(f"{out}.{j}", "w")), f"{out}.{j}"))
            for pr, f in procs:
                if pr.wait() != 0:
                    raise SystemExit(f"arc-stream failed: {f}")
            with open(out, "w") as f:
                for _, part in procs:
                    f.write(open(part).read())
                    os.remove(part)
            print(prov, c, flush=True)


def c_load(path):
    """arc-stream --records output -> {clip: dict(text, utts)}; utts: the settled tags of each utterance id."""
    res, cur = {}, {}
    for l in open(path):
        try:
            j = json.loads(l)
        except ValueError:
            continue
        if "file" in j:
            res[os.path.basename(j["file"]).rsplit(".", 1)[0]] = dict(text=j.get("text", ""), utts=cur)
            cur = {}
        elif j.get("kind") in ("final", "tags", "FINAL", "TAGS"):
            u = cur.setdefault(j["id"], {})
            if j.get("kind").lower() == "final" or j.get("tags_final", 1) or "dd" not in u:
                u.update(start=j["start"], end=j["end"], dd=j.get("device_directed"), intent=j.get("intent"), act=j.get("act"))
            if j.get("kind").lower() == "final":
                u["text"] = j.get("text", "")
    return res


def c_report(a):
    import jiwer
    from whisper_normalizer.english import EnglishTextNormalizer
    norm = EnglishTextNormalizer()
    refs, _ = clips(a.set)
    rows = []
    for prov in a.providers:
        for c in ("clean", "noisy"):
            f = os.path.join(a.c, prov, c + ".jsonl")
            fo = os.path.join(a.c, "arc", c + ".jsonl")
            if not os.path.exists(f):
                continue
            X, O = c_load(f), c_load(fo) if os.path.exists(fo) else {}
            R, P, M, ttf, eq, tag = [], [], [], [], 0, []
            for k in sorted(refs):
                cap = os.path.join(a.capture, prov, c, k + ".jsonl")
                if k not in X or not os.path.exists(cap):
                    continue
                meta, recs = load_capture(cap)
                alone, last = provider_alone(prov, recs)
                R.append(norm(refs[k])); P.append(norm(" ".join(w["text"] for w in alone))); M.append(norm(X[k]["text"]))
                eq += P[-1] == M[-1]
                if last is not None and "t_end" in meta:
                    ttf.append(max(0.0, last - meta["t_end"]))
                for i, u in X[k]["utts"].items():
                    v = max((O.get(k) or {}).get("utts", {}).values(), default=None,
                            key=lambda v: min(u["end"], v["end"]) - max(u["start"], v["start"]))
                    if v is None or min(u["end"], v["end"]) <= max(u["start"], v["start"]):
                        continue
                    tag.append((u["act"] == v["act"], u["intent"] == v["intent"], abs((u["dd"] or 0) - (v["dd"] or 0))))
            if not R:
                continue
            rows.append(dict(provider=prov, cond=c, n=len(R), wer_provider=100 * jiwer.wer(R, P), wer_merged=100 * jiwer.wer(R, M),
                             same_text=eq, ttf_p50=float(np.percentile(ttf, 50)), ttf_p90=float(np.percentile(ttf, 90)),
                             n_utts=len(tag), act=100 * np.mean([t[0] for t in tag]) if tag else float("nan"),
                             intent=100 * np.mean([t[1] for t in tag]) if tag else float("nan"),
                             dd_max=max([t[2] for t in tag], default=float("nan"))))
    print("| provider | audio | clips | WER provider | WER Arc EXTERNAL (libarc) | same text | time to final p50 / p90 | Arc utterances | same act / intent as Arc's own | max device-directed diff |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['provider']} | {r['cond']} | {r['n']} | {r['wer_provider']:.2f} % | {r['wer_merged']:.2f} % | {r['same_text']}/{r['n']} | "
              f"{r['ttf_p50']:.2f} / {r['ttf_p90']:.2f} s | {r['n_utts']} | {r['act']:.0f} % / {r['intent']:.0f} % | {r['dd_max']:.4f} |")
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
    p = sub.add_parser("export-words"); p.add_argument("set"); p.add_argument("capture"); p.add_argument("out")
    p.add_argument("--providers", nargs="+", default=["deepgram", "assemblyai", "speechmatics"])
    p = sub.add_parser("c-run"); p.add_argument("set"); p.add_argument("words"); p.add_argument("out"); p.add_argument("--mini", required=True)
    p.add_argument("--providers", nargs="+", default=["deepgram", "assemblyai", "speechmatics"]); p.add_argument("--exe")
    p.add_argument("--threads", type=int, default=1); p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--domain", default="unknown"); p.add_argument("--force", action="store_true")
    p = sub.add_parser("c-report"); p.add_argument("set"); p.add_argument("capture"); p.add_argument("c")
    p.add_argument("--providers", nargs="+", default=["deepgram", "assemblyai", "speechmatics"]); p.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "make-set":
        make_set(a)
    elif a.cmd == "capture":
        asyncio.run(capture_all(a))
    elif a.cmd == "replay":
        replay_all(a)
    elif a.cmd == "report":
        report(a)
    else:
        dict(**{"export-words": export_words, "c-run": c_run, "c-report": c_report})[a.cmd](a)


if __name__ == "__main__":
    main()
