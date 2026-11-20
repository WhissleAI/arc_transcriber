"""Conformance checks for arc-serve: every route, the error contract, realtime streaming, and parity with the Python
reference (arc_audio) on the same audio.

    python deploy/conformance/conformance.py --url http://127.0.0.1:8080 --model-dir /path/arc-v1.1/mini \
        [--python /path/to/python-with-arc_audio] [--skip-reference] [--combined] [--report out.json]

Parity: c/tools/dump_names.py runs arc_audio's Arc.predict on each clip; the server's response_format=arc_json writes the
same keys; c/tools/compare_names.py compares transcript text, every word (text, inline tag, start / end to 1 ms), every
name / number (heard, entity, state, kind, values, a date's date / iso; times to 1.5 ms; probabilities to 2e-3) and the
speakers (ids, role label, p to 2e-3), the diarization turns (speaker, start / end: the same 100 ms frame) and each utterance's
speaker. Clips: the repository's examples/example_synth.wav (nba, examples/example_names.json) plus clips synthesized here
with the OS text-to-speech (macOS `say`, or espeak-ng; written to files, never played): single-voice commands, two- and
three-voice dialogues (run in "unknown" = slot diarization, "meeting" = the TrackBank on the voice embedding, "nhl" =
TrackBank fused with the slots), a tone and silence.

Standard library only (the reference run needs arc_audio's own dependencies: torch, transformers, ...).
Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0 with the Whissle terms; see LICENSE.
"""
import argparse
import base64
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import http_client as H  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
EXAMPLE = os.path.join(ROOT, "examples", "example_synth.wav")
EXAMPLE_NAMES = os.path.join(ROOT, "examples", "example_names.json")

TTS_TEXTS = [
    "Call Priya Raman at four fifteen tomorrow and tell her the budget is twelve thousand dollars.",
    "Set a timer for twenty five minutes, then play the new album by Nora Ellis in the kitchen.",
    "Remind me to send the contract to Daniel Brooks on March third. The meeting moved to room two oh four.",
]
TTS_NAMES = [
    {"name": "Priya Raman", "kind": "person"},
    {"name": "Nora Ellis", "kind": "person"},
    {"name": "Daniel Brooks", "kind": "person", "aliases": ["Dan Brooks"]},
]

results = []


def check(name, ok, detail=""):
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    return ok


def read_wav_pcm16(path):
    with wave.open(path) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, path
        return w.getframerate(), w.readframes(w.getnframes())


def write_wav(path, rate, samples):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", max(-32767, min(32767, int(s * 32767)))) for s in samples))


def synth_clips(tmp):
    """OS text-to-speech to 16 kHz mono WAV files (nothing is played)."""
    out = []
    say, afconvert, espeak = shutil.which("say"), shutil.which("afconvert"), shutil.which("espeak-ng") or shutil.which("espeak")
    for i, t in enumerate(TTS_TEXTS):
        wav = os.path.join(tmp, f"tts_{i}.wav")
        try:
            if say and afconvert:
                aiff = os.path.join(tmp, f"tts_{i}.aiff")
                subprocess.run([say, "-o", aiff, t], check=True, capture_output=True, timeout=120)
                subprocess.run([afconvert, "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff, wav], check=True, capture_output=True, timeout=120)
            elif espeak:
                raw = os.path.join(tmp, f"tts_{i}_raw.wav")
                subprocess.run([espeak, "-w", raw, t], check=True, capture_output=True, timeout=120)
                if shutil.which("sox"):
                    subprocess.run(["sox", raw, "-r", "16000", "-c", "1", "-b", "16", wav], check=True, capture_output=True)
                else:
                    wav = raw   # 22.05 kHz: the server resamples
            else:
                break
            out.append(wav)
        except Exception as e:  # noqa: BLE001
            print(f"note: TTS failed ({e}); continuing without synthesized clips", flush=True)
            break
    return out


DIALOGUES = [   # (voice, text) turns; macOS voices, else espeak-ng variants
    [("Samantha", "Good morning everyone, thanks for joining the weekly planning meeting. Let's start with the budget review."),
     ("Daniel", "Thanks. The budget is on track, we spent about forty two thousand dollars last quarter."),
     ("Samantha", "Great. And what about the hiring plan for the new office in Denver?"),
     ("Daniel", "We have three open roles. Interviews start next Tuesday and should finish by the end of March.")],
    [("Daniel", "Can you call Priya Raman about the contract?"),
     ("Karen", "Sure, I will call her at four fifteen tomorrow."),
     ("Fred", "And remind Daniel Brooks that the demo moved to Friday."),
     ("Daniel", "Thanks, I will put both on the calendar.")],
]
ESPEAK_VOICES = {"Samantha": "en-us+f3", "Daniel": "en-gb+m3", "Karen": "en+f4", "Fred": "en-us+m1"}


def synth_dialogues(tmp):
    """multi-speaker clips: each turn synthesized in its own voice, joined with 0.6 s of silence (nothing is played)"""
    say, afconvert, espeak = shutil.which("say"), shutil.which("afconvert"), shutil.which("espeak-ng") or shutil.which("espeak")
    out = []
    for i, turns in enumerate(DIALOGUES):
        try:
            data = b""
            for k, (voice, text) in enumerate(turns):
                part = os.path.join(tmp, f"dlg_{i}_{k}.wav")
                if say and afconvert:
                    aiff = part[:-4] + ".aiff"
                    subprocess.run([say, "-v", voice, "-o", aiff, text], check=True, capture_output=True, timeout=120)
                    subprocess.run([afconvert, "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff, part], check=True, capture_output=True, timeout=120)
                elif espeak and shutil.which("sox"):
                    raw = part[:-4] + "_raw.wav"
                    subprocess.run([espeak, "-v", ESPEAK_VOICES[voice], "-w", raw, text], check=True, capture_output=True, timeout=120)
                    subprocess.run(["sox", raw, "-r", "16000", "-c", "1", "-b", "16", part], check=True, capture_output=True)
                else:
                    return out
                rate, pcm = read_wav_pcm16(part)
                data += (b"\x00\x00" * int(0.6 * rate) if k else b"") + pcm
            wav = os.path.join(tmp, f"dialogue_{i}.wav")
            with wave.open(wav, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(data)
            out.append(wav)
        except Exception as e:  # noqa: BLE001
            print(f"note: dialogue TTS failed ({e}); continuing without it", flush=True)
            break
    return out


def run_reference(py, mini, names, domain, clips, out):
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    cmd = ["nice", "-n", "19", py, os.path.join(ROOT, "c", "tools", "dump_names.py"), mini, names, domain, out] + clips
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=3600)
    if p.returncode:
        print(p.stdout[-2000:], p.stderr[-4000:], sep="\n")
    return p.returncode == 0


def server_arc_json(url, clip, names_path, domain):
    fields = {"response_format": "arc_json", "domain": domain, "events": "false"}
    if names_path:
        fields["names"] = open(names_path).read()
    r = H.post_multipart(url, "/v1/audio/transcriptions", fields, {"file": (os.path.basename(clip), open(clip, "rb").read(), "audio/wav")})
    if r.status != 200:
        raise RuntimeError(f"{clip}: HTTP {r.status} {r.text[:300]}")
    rec = r.json()
    out = {"clip": os.path.basename(clip), "domain": domain, "transcript": rec["transcript"], "names": rec["names"], "numbers": rec["numbers"]}
    for k in ("speakers", "diarization"):
        if k in rec:
            out[k] = rec[k]
    return out


def parity(url, args, tmp, sets):
    """sets: [(clips, names_path, domain)]"""
    ok_all = True
    for k, (clips, names_path, domain) in enumerate(sets):
        if not clips:
            continue
        ref, cand = os.path.join(tmp, f"ref_{k}.jsonl"), os.path.join(tmp, f"cand_{k}.jsonl")
        t0 = time.time()
        if not run_reference(args.python, args.model_dir, names_path, domain, clips, ref):
            ok_all &= check(f"reference: arc_audio on set {k} ({domain})", False, "dump_names.py failed")
            continue
        t_ref = time.time() - t0
        with open(cand, "w") as f:
            for c in clips:
                f.write(json.dumps(server_arc_json(url, c, names_path, domain)) + "\n")
        p = subprocess.run([sys.executable, os.path.join(ROOT, "c", "tools", "compare_names.py"), ref, cand], capture_output=True, text=True)
        refs = [json.loads(l) for l in open(ref)]
        n_words = sum(len(u["words"]) for r in refs for u in r["transcript"])
        n_names = sum(len(r["names"]) for r in refs)
        n_nums = sum(len(r["numbers"]) for r in refs)
        detail = (f"{len(clips)} clips, {n_words} words, {n_names} names, {n_nums} numbers; reference {t_ref:.0f} s; "
                  + (p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""))
        ok = p.returncode == 0
        ok_all &= check(f"parity vs arc_audio: set {k} ({domain}, {os.path.basename(names_path)})", ok, detail)
        if not ok:
            print(p.stdout[-4000:], p.stderr[-2000:])
    return ok_all


def realtime_speakers(url, clip, domain):
    """a dialogue through WS /v1/realtime: finals carry speaker, session.closed the stream's speakers / diarization, which
    should be the offline transcription's"""
    rate, pcm = read_wav_pcm16(clip)
    ws = H.WS(url, f"/v1/realtime?domain={domain}&events=0")
    ws.recv_json()
    for i in range(0, len(pcm), 6400):
        ws.send(pcm[i:i + 6400], binary=True)
    ws.send_json({"type": "input_audio_buffer.finish"})
    finals, closed = [], None
    t0 = time.time()
    while time.time() - t0 < 600:
        m = ws.recv_json()
        if m is None:
            break
        if m["type"] == "transcript.final":
            finals.append(m)
        if m["type"] == "session.closed":
            closed = m
            break
    ws.close()
    off = H.post_multipart(url, "/v1/audio/transcriptions", {"response_format": "verbose_json", "domain": domain, "events": "false"},
                           {"file": ("d.wav", open(clip, "rb").read(), "audio/wav")}).json()
    name = f"{os.path.basename(clip)}, {domain}"
    check(f"WS /v1/realtime speakers ({name}): every final carries speaker", finals and all("speaker" in f for f in finals),
          f"{len(finals)} finals")
    sd = closed and closed.get("diarization")
    od = off.get("diarization")
    check(f"WS /v1/realtime speakers ({name}): session.closed diarization / speakers == offline", closed is not None and
          sd == od and sorted((closed.get("speakers") or {})) == sorted(off.get("speakers") or {}),
          f"stream {len(sd or [])} turns, offline {len(od or [])}" + ("" if sd == od else f"; stream {sd} vs offline {od}"))
    agree = sum(f.get("speaker") == s.get("speaker") for f, s in zip(finals, off.get("segments", [])))
    print(f"note: realtime speakers ({name}): {agree}/{len(finals)} finals name the offline segment's speaker at the time they are sent", flush=True)


def realtime(url, clip, domain="unknown"):
    rate, pcm = read_wav_pcm16(clip)
    assert rate == 16000
    ws = H.WS(url, f"/v1/realtime?domain={domain}&events=1")
    first = ws.recv_json()
    check("WS /v1/realtime: session.created", first and first.get("type") == "session.created", json.dumps(first)[:120])
    step = 3200   # 100 ms
    for i in range(0, len(pcm), step * 2):
        ws.send(pcm[i:i + step * 2], binary=True)
    ws.send_json({"type": "input_audio_buffer.finish"})
    kinds, finals = {}, []
    t0 = time.time()
    while time.time() - t0 < 600:
        m = ws.recv_json()
        if m is None:
            break
        kinds[m["type"]] = kinds.get(m["type"], 0) + 1
        if m["type"] == "transcript.final":
            finals.append(m)
        if m["type"] == "session.closed":
            break
    ws.close()
    check("WS /v1/realtime: partial / endpoint / final / closed records", all(kinds.get(k) for k in ("transcript.partial", "transcript.endpoint", "transcript.final", "session.closed")),
          json.dumps(kinds))
    off = H.post_multipart(url, "/v1/audio/transcriptions", {"response_format": "verbose_json", "domain": domain},
                           {"file": ("a.wav", open(clip, "rb").read(), "audio/wav")}).json()
    st = [f["text"] for f in finals]
    of = [s["text"] for s in off["segments"]]
    check("WS /v1/realtime: finals == offline transcript (libarc: FULL finals = arc_transcribe)", st == of,
          f"{len(st)} finals" + ("" if st == of else f"; stream {st} vs offline {of}"))
    return finals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model-dir", help="Arc Mini folder for the Python reference")
    ap.add_argument("--python", default=sys.executable, help="interpreter with arc_audio's dependencies")
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--combined", action="store_true", help="the transcriber + tagger image")
    ap.add_argument("--stage-dir", help="ARC_FILE_ROOT as the server sees it, holding example_synth.wav (Snowflake stage-file rows)")
    ap.add_argument("--report")
    a = ap.parse_args()
    url = a.url
    tmp = tempfile.mkdtemp(prefix="arc-conf-")
    wav = open(EXAMPLE, "rb").read()

    # ---- health and metadata
    for p in ("/health", "/v1/health/live", "/v1/health/ready", "/ping"):
        r = H.get(url, p)
        check(f"GET {p}", r.status == 200, f"{r.status} {r.text.strip()}")
    r = H.get(url, "/v1/models")
    models = r.json() if r.status == 200 else {}
    ids = [m["id"] for m in models.get("data", [])]
    check("GET /v1/models", r.status == 200 and models.get("object") == "list" and ids, ",".join(ids))
    check("GET /v1/models/{id}", H.get(url, f"/v1/models/{ids[0]}").status == 200 if ids else False)
    check("GET /v1/models/unknown -> 404", H.get(url, "/v1/models/nope").status == 404)
    r = H.get(url, "/execution-parameters")
    check("GET /execution-parameters (SageMaker batch)", r.status == 200 and "MaxConcurrentTransforms" in r.json(), r.text)

    # ---- /v1/audio/transcriptions: every response format
    files = {"file": ("example_synth.wav", wav, "audio/wav")}
    names = open(EXAMPLE_NAMES).read()
    r = H.post_multipart(url, "/v1/audio/transcriptions", {"model": "whisper-1"}, files)
    check("POST /v1/audio/transcriptions (json, OpenAI client defaults)", r.status == 200 and r.json().get("text"), r.text[:100])
    r = H.post_multipart(url, "/v1/audio/transcriptions", {"response_format": "text"}, files)
    check("response_format=text", r.status == 200 and r.headers["content-type"].startswith("text/plain") and len(r.text) > 20)
    r = H.post_multipart(url, "/v1/audio/transcriptions", {"response_format": "verbose_json", "names": names, "domain": "nba"}, files)
    v = r.json() if r.status == 200 else {}
    linked = [n for n in v.get("names", []) if n.get("state") == "linked"]
    valued = [n for n in v.get("numbers", []) if n.get("value") is not None]
    check("response_format=verbose_json: segments, words, names, numbers, events",
          r.status == 200 and v.get("segments") and v.get("words") and linked and valued and "events" in v,
          f"{len(v.get('segments', []))} segments, {len(v.get('words', []))} words, {len(linked)} linked names, {len(valued)} numbers with values")
    check("verbose_json: names linked to the list", {"Theo Brandt", "Marcus Delaney"} <= {n["entity"] for n in linked}, ", ".join(sorted({n['entity'] for n in linked})))
    check("verbose_json: number values (102 to 98 -> 102 / 98)", any(n.get("value") == 102 and n.get("value2") == 98 for n in valued),
          "; ".join(f"{n['heard']}={n['value']}" for n in valued[:6]))
    check("verbose_json: utterance tags (inline ENTITY_ / NUMBER_ spans)", any(s["tags"] for s in v.get("segments", [])))
    if a.combined:
        check("combined: arc_tags on the transcript", isinstance(v.get("arc_tags"), dict) and "spans" in v["arc_tags"],
              f"{len(v.get('arc_tags', {}).get('spans', []))} spans")
    for fmt, needle in (("srt", "-->"), ("vtt", "WEBVTT")):
        r = H.post_multipart(url, "/v1/audio/transcriptions", {"response_format": fmt}, files)
        check(f"response_format={fmt}", r.status == 200 and needle in r.text)
    r = H.post_multipart(url, "/v1/audio/transcriptions", {"response_format": "arc_json"}, files)
    check("response_format=arc_json (arc_audio keys)", r.status == 200 and "transcript" in r.json())
    r = H.post_multipart(url, "/v1/audio/transcriptions", {"response_format": "json", "names[]": ["Theo Brandt", "Marcus Delaney"]}, files)
    check("names[] repeated form fields", r.status == 200)
    r = H.post_multipart(url, "/v1/audio/transcriptions", {"prompt": "Basketball: Hawks vs Comets.", "terms": "Brandt,Delaney"}, files)
    check("prompt + terms (decoder context)", r.status == 200 and r.json().get("text"))

    # resampled input: 44.1 kHz stereo float... (here: 8 kHz and 44.1 kHz PCM16 from the example)
    rate, pcm16 = read_wav_pcm16(EXAMPLE)
    s = struct.unpack(f"<{len(pcm16) // 2}h", pcm16)[: 16000 * 6]
    for rr in (8000, 44100):
        n = int(len(s) * rr / 16000)
        ys = [s[min(len(s) - 1, int(i * 16000 / rr))] / 32768 for i in range(n)]
        p = os.path.join(tmp, f"r{rr}.wav")
        write_wav(p, rr, ys)
        r = H.post_multipart(url, "/v1/audio/transcriptions", {}, {"file": ("r.wav", open(p, "rb").read(), "audio/wav")})
        check(f"WAV at {rr} Hz (resampled)", r.status == 200 and "Welcome" in r.json().get("text", ""), r.json().get("text", r.text)[:60] if r.status == 200 else r.text[:100])
    # raw PCM16
    r = H.request(url, "POST", "/v1/audio/transcriptions?sample_rate=16000&response_format=text", pcm16[: 32000 * 3], {"Content-Type": "audio/pcm"})
    check("raw audio/pcm body (16 kHz PCM16)", r.status == 200 and "Welcome" in r.text, r.text.strip()[:60])
    # tone + silence: no crash, no invented words
    tone = os.path.join(tmp, "tone.wav")
    write_wav(tone, 16000, [0.2 * math.sin(2 * math.pi * 440 * i / 16000) for i in range(16000 * 2)] + [0.0] * 16000)
    r = H.post_multipart(url, "/v1/audio/transcriptions", {"response_format": "verbose_json"}, {"file": ("tone.wav", open(tone, "rb").read(), "audio/wav")})
    check("tone + silence -> 200", r.status == 200, repr(r.json().get("text")) if r.status == 200 else r.text[:100])

    # ---- error contract
    r = H.post_multipart(url, "/v1/audio/transcriptions", {"response_format": "json"}, {})
    check("missing file -> 400 + OpenAI error envelope", r.status == 400 and "error" in r.json())
    r = H.post_multipart(url, "/v1/audio/transcriptions", {}, {"file": ("x.mp3", b"ID3\x04\x00" + b"\x00" * 200, "audio/mpeg")})
    check("MP3 -> 415 (WAV / PCM only)", r.status == 415, r.json()["error"]["message"][:80] if r.status == 415 else r.text[:80])
    r = H.post_multipart(url, "/v1/audio/transcriptions", {"language": "fr"}, files)
    check("language=fr -> 400 (English only)", r.status == 400)
    r = H.post_multipart(url, "/v1/audio/transcriptions", {"response_format": "xml"}, files)
    check("bad response_format -> 400", r.status == 400)
    check("unknown route -> 404", H.get(url, "/v2/nothing").status == 404)
    check("GET on a POST route -> 405", H.get(url, "/v1/audio/transcriptions").status == 405)
    check("/v1/realtime without upgrade -> 400", H.get(url, "/v1/realtime").status == 400)

    # ---- SageMaker /invocations
    r = H.request(url, "POST", "/invocations", wav, {"Content-Type": "audio/wav", "X-Amzn-SageMaker-Custom-Attributes": "domain=nba"})
    check("POST /invocations audio/wav -> verbose_json", r.status == 200 and r.json().get("segments"))
    b64 = base64.b64encode(wav).decode()
    r = H.post_json(url, "/invocations", {"audio": b64, "names": json.loads(names), "domain": "nba"})
    check("POST /invocations JSON {audio: base64, names}", r.status == 200 and any(n.get("state") == "linked" for n in r.json().get("names", [])))
    short = base64.b64encode(open(tone, "rb").read()).decode()
    batch = [{"audio": short}, {"audio": b64, "response_format": "text"}]
    if a.combined:
        batch.append({"text": "Call Maria Lopez at 555 867 5309."})
    r = H.post_json(url, "/invocations", {"instances": batch})
    check("POST /invocations JSON batch (instances)", r.status == 200 and len(r.json().get("results", [])) == len(batch))
    r = H.request(url, "POST", "/invocations", (json.dumps({"audio": short}) + "\n" + json.dumps({"audio": short}) + "\n").encode(),
                  {"Content-Type": "application/jsonlines"})
    check("POST /invocations application/jsonlines (batch transform, SplitType=Line)", r.status == 200 and len(r.text.strip().splitlines()) == 2)
    r = H.request(url, "POST", "/invocations", wav, {"Content-Type": "audio/wav", "Accept": "text/plain"})
    check("POST /invocations Accept: text/plain", r.status == 200 and r.headers["content-type"].startswith("text/plain"))

    # ---- Snowflake service function shape
    r = H.post_json(url, "/v1/snowflake/transcribe", {"data": [[0, short], [1, b64, '{"domain": "nba", "names": ["Theo Brandt"]}']]})
    d = r.json().get("data", []) if r.status == 200 else []
    check("POST /v1/snowflake/transcribe base64 rows (SPCS service function)", r.status == 200 and [x[0] for x in d] == [0, 1] and "Brandt" in d[1][1].get("text", ""))
    if a.stage_dir:
        r = H.post_json(url, "/v1/snowflake/transcribe", {"data": [[0, a.stage_dir + "/example_synth.wav"], [1, a.stage_dir + "/../README.md"], [2, "/etc/passwd"]]})
        d = r.json().get("data", []) if r.status == 200 else []
        check("POST /v1/snowflake/transcribe stage-file rows (ARC_FILE_ROOT), traversal refused",
              r.status == 200 and "Welcome" in d[0][1].get("text", "") and "error" in d[1][1] and "error" in d[2][1],
              "; ".join(str(x[1].get("error", {}).get("message", "ok"))[:50] for x in d))
    if a.combined:
        r = H.post_json(url, "/v1/snowflake/tags", {"data": [[0, "Call Maria Lopez at 555 867 5309."]]})
        check("combined: POST /v1/snowflake/tags", r.status == 200 and r.json()["data"][0][1].get("spans"))

    if a.combined:
        r = H.post_json(url, "/v1/tags", {"text": "Hi, this is Maria Lopez. My number is 555 867 5309.", "domain": "talk"})
        check("combined: POST /v1/tags", r.status == 200 and r.json().get("spans"))
        r = H.post_json(url, "/v1/tags", {"segments": v.get("segments", [])})
        check("combined: POST /v1/tags with the transcript's segments", r.status == 200 and r.json().get("n_words", 0) > 20)

    # ---- realtime
    finals = realtime(url, EXAMPLE)
    for clip in synth_dialogues(tmp)[:1]:
        for dom in ("unknown", "meeting"):
            realtime_speakers(url, clip, dom)
    if a.combined:
        check("combined: realtime finals carry arc_tags", finals and all("arc_tags" in f for f in finals if f.get("text")))

    # ---- concurrency: 4 requests at once
    import threading
    codes = []
    def one():
        codes.append(H.post_multipart(url, "/v1/audio/transcriptions", {}, {"file": ("t.wav", open(tone, "rb").read(), "audio/wav")}).status)
    ts = [threading.Thread(target=one) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    check("4 concurrent requests", codes == [200] * 4, str(codes))
    r = H.get(url, "/health")
    check("server still healthy after the run", r.status == 200)

    # ---- parity with the Python reference
    if not a.skip_reference:
        if not a.model_dir:
            check("parity vs arc_audio", False, "--model-dir is required for the reference run (or --skip-reference)")
        else:
            tts = synth_clips(tmp)
            names_tts = os.path.join(tmp, "tts_names.json")
            json.dump(TTS_NAMES, open(names_tts, "w"))
            dlg = synth_dialogues(tmp)
            parity(url, a, tmp, [([EXAMPLE], EXAMPLE_NAMES, "nba"), (tts, names_tts, "assistant"), (tts + [tone], EXAMPLE_NAMES, "unknown"),
                                 (dlg, names_tts, "unknown"), (dlg, names_tts, "meeting"), (dlg, names_tts, "nhl")])

    n_ok = sum(r["ok"] for r in results)
    print(f"\n{n_ok}/{len(results)} checks passed")
    if a.report:
        json.dump({"url": url, "passed": n_ok, "total": len(results), "results": results}, open(a.report, "w"), indent=1)
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
