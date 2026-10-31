# evals/: Arc evaluations from building Steer

These are the evaluations run while building **Steer** (WhissleAI's reading and voice app for Mac, Windows and Android) on
Arc Mini, collected so the Arc team can rerun them. Each folder has:
- a README: what it measures, how to run it, results with date, model revision and runtime, and known limits;
- a runnable script on `arc_audio`;
- its text lists.

No audio is committed:
- The scripts render their test speech locally with macOS `say` (system voices, written straight to files) into `evals/_work/` (git-ignored).
- On Linux, render the same file names with any TTS and pass `--clips`.
- Nothing here comes from a real person's recording.
- The 200 book-request clips Steer used are macOS-voice renders, 11 MB. They aren't committed because Apple's system-voice
  output isn't ours to redistribute; `book_search/run.py` re-renders them with the same voices and file names.

| Eval | Question | Headline (Steer app path → this PR's Python rerun) |
|---|---|---|
| [commands](commands/) | Does a spoken command become the right command, by room? | Steer 92.0% right (96% quiet / 83% loud room), 39–64 ms per command → rerun 74.9% (85% quiet / 62% loud room; macOS voices), 151 ms p50 |
| [context_memory](context_memory/) | Does text context (names prompt, bias, previous text) help? Does carrying the trunk state help? | WER 30.0% → 20.2%, names right 43% → 69%; carried state hurt (directed AUC 0.418 → 0.338, act 97 → 93/100) → rerun WER 44.4% → 33.0%, names 29% → 57%; carried AUC 0.440 → 0.347 |
| [book_search](book_search/) | Spoken title/author requests over 61,476 books, and Arc's own name linker | right book first 76% → 81% with live context, catalog lookup 1.8 ms; Arc linker 34–49% → rerun right book 68% → 75% (stand-in search), linker 32–53% |
| [device_directed](device_directed/) | Commands vs people talking to each other; which wake names are heard | cross-talk scored as high as commands (nurse 0.74–0.93 vs surgeon 0.51–0.88); "Scope" missed 8/9 → rerun cross-talk AUC 0.33; Scope heard 1/9 unprimed, 7/9 primed |
| [whisper_noise](whisper_noise/) | Whispers, low level, noise, telephone, clipping | whispers are the weakest input; telephone 0.57 / clipping 0.51 fire only on their condition → rerun whisper voice 58.9% WER vs 20.8% clean; clipping 0.66; telephone silent on short clips |
| [latency_streaming](latency_streaming/) | Offline and streaming latency, when each output arrives | Steer 0.05–0.12 s per command; Pipecat final ~2.7 s late → rerun: commands under 1 s get their final at +14–15 s (segmenter `min_s` gate); utterance record +0.79 s; flush 155 ms |
| [written_form](written_form/) | Does the written transcript (`transcript_written`) score better against written references? | 1.2.1: digits found 22% → 74%, whole sentence right 18% → 53%, WER 27.2% → 24.2% on 120 number-heavy clips; LibriSpeech WER unchanged (15.5%) |
| [v121](v121/) | Does 1.2.1 hold up on real speech, both models? TypeWhisper reproduced; Earnings-22; LibriSpeech; MPS; streaming / CLI; dates | TypeWhisper reproduced exactly (340/340 clips); written form: Digits 28% → 72% (Mini), 4% → 76% (Transcriber); Earnings-22 number recall 5% → 80% (Transcriber), 0 wrong values in 60 hand-checked; CPU = MPS outputs |
| [api_issues](api_issues/) | Minimal reproductions of the API issues | 4 issues (late final for short commands, rewrite duplicates, global switch interval, two device-directed numbers) |

**[FINDINGS.md](FINDINGS.md)** lists the learnings for the Arc team, each with its evidence and an ask.

## Where the numbers come from

- **Steer (app path):** Steer's own runtime, `libarc`.
  - libarc is a dependency-free C11 port of Arc Mini, about 7,400 lines, in the private Steer apps repo. It is being proposed for this repo as `c/` on a separate branch.
  - Model: Arc Mini 1.1.1 (HF `WhissleAI/arc-v1.1` rev `v1.1.1`), on an Apple M3 Pro CPU.
  - Test speech: Steer's own TTS voices and macOS voices.
  - Steer's command parser and catalog search are app code, not public.
- **Python reference (this PR):** the scripts here, run on 2026-10-29.
  - Package: `arc_audio` 1.2.0 (this repo at `23f032b`), with `from_pretrained`'s default model revision `v1.2.0`. Its Arc Mini files are byte-identical to `v1.1.1`: the same blobs in the Hugging Face cache.
  - Hardware: CPU only, 4 threads, M3 Pro.
  - Test speech: macOS voices.
  - The app-only parts are replaced by small stand-ins (`commands/parser.py`, `book_search/catalog.py`), and the READMEs say where that changes a number.
- **Other Python runs:** the signal study (1.1.1), the Pipecat integration (1.1.1) and the linker experiments. Each table says which run a number comes from.
- **Arc v1.4 demo server:** the ASR comparison film's replay data. It can't be rerun here; it is cited in FINDINGS only.

## Setup

```bash
pip install -e .          # this repo; the evals use arc_audio plus numpy, soundfile, rapidfuzz, jellyfish (all dependencies already)
python evals/commands/run.py --quick
```

Every script takes `--threads` and `--out`. Results go to `evals/<eval>/results/run.json`, with an `env` block (date,
`arc_audio` version, model revision and snapshot, torch, CPU). Synthetic voices throughout: the real-voice session that would
settle device-directed and affect is still to be done.
