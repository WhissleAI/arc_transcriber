# evals/v121: evaluation of the 1.2.1 changes (written form, date records, MPS)

These are the checks for `arc-transcriber` 1.2.1 (PR #4, merge `120087f`), run before replying to the TypeWhisper benchmark or
tagging v1.2.1. The changes under test are:
- the written-form transcript (`transcript_written`, `text_written`, `--written`);
- date records that keep the month;
- MPS support and `device="auto"`.

Both models were run: Arc Mini 1.1.1 and Arc Transcriber 1.2.0 (full, Parakeet encoder), from model revision `v1.2.0`. The
runs used real speech that we didn't write. This PR also fixes three formatter bugs that the runs found. The fixes are covered
in [Bugs found and fixed](#bugs-found-and-fixed).

**Setup:** run on 2026-10-30, Apple M3 Pro (6 performance + 6 efficiency cores), macOS. torch 2.15.0, transformers 5.22.1,
Python 3.12. Default torch threads (6), as TypeWhisper's worker uses. "main" is `120087f`. "fix" is this branch.

**Not tested:**
- CUDA: there is no NVIDIA GPU here.
- Windows and Linux.

Every number below comes from a script in this folder. Per-run results are in `results/`. No audio is committed. The scripts
download or render their audio.

## 1. TypeWhisper's benchmark, reproduced

**Method:** we used TypeWhisper's code at their merged PR #16 (the provider `bench/src/providers/arc.ts` and the worker
`bench/scripts/arc-worker.py`):
- their 10 English suites (340 clips; the audio is in their repo);
- their settings: `Arc.predict(wav)` with package defaults, CPU, default threads, timed around `predict()` only.

`twscore.py` is a Python port of their scoring:
- `normalizeText` and the clamped WER from `bench/src/metrics`;
- `digitsHit` from `visualizer/scripts/build-data.mjs`;
- the pooling from `visualizer/src/data/metrics.ts`.

`check_port.py` recomputes their published `werNormalized` and Digits from their own published transcriptions. It matches
**340 / 340 clips** for both Arc models and for Parakeet v2, so this port scores exactly as theirs does.

**Reproduction:**
- Our 1.2.0 plain transcripts are **identical, character for character, to TypeWhisper's published transcripts on 340 / 340
  clips, for both models**. Every WER and Digits number of theirs reproduces exactly.
- 1.2.1's plain transcript is identical to 1.2.0's on all 680 clip runs, as expected for a code-only release.

WER per suite (TypeWhisper normalizer, mean over clips, %):

| Suite | Mini plain (1.2.0 = 1.2.1 = TW) | Mini written, main | Mini written, fix | Transcriber plain (1.2.0 = 1.2.1 = TW) | Transcriber written, main | Transcriber written, fix |
|---|---|---|---|---|---|---|
| clean-speech-en | 0.6 | 0.6 | 0.6 | 0.6 | 0.6 | 0.6 |
| clean-speech-librispeech-en | 9.8 | 10.1 | 10.1 | 4.0 | 4.7 | 4.7 |
| clean-speech-commonvoice-en | 42.8 | 42.8 | 42.8 | 13.8 | 13.8 | 13.3* |
| noisy-environment-en | 6.7 | 6.7 | 6.7 | 2.6 | 2.6 | 2.6 |
| whispered-lowquality-en | 4.1 | 4.1 | 4.1 | 3.2 | 2.5 | 2.5 |
| hard-en | 25.8 | 26.4 | 25.7 | 14.8 | 13.4 | 13.7 |
| code-dictation-en | 37.8 | 38.3 | 38.3 | 15.6 | 15.9 | 15.9 |
| number-formatting-en | 26.5 | 16.7 | 16.4 | 27.1 | 12.4 | 12.4 |
| punctuation-formatting-en | 12.7 | 12.7 | 12.7 | 5.8 | 8.3 | 8.3 |
| accented-speech-en | 96.9 | 96.9 | 96.9 | 96.8 | 96.8 | 97.0* |
| **WER EN w/o acc.** (their headline: core suites) | **23.4** | 22.5 | 22.5 | **9.9** | 8.8 | 8.6* |
| WER over all 10 suites | 30.5 | 29.9 | 29.8 | 17.2 | 16.4 | 16.2* |
| **Digits** (number-formatting clips with every number verbatim) | **28 % (7/25)** | 72 % (18/25) | 72 % (18/25) | **4 % (1/25)** | 76 % (19/25) | 76 % (19/25) |

"fix" for the Transcriber is measured on `07cde07` (PR #5). PR #5 changed 2 plain transcripts there; the starred cells include
those changes (below).

**Current main `07cde07` (PR #5):** PR #5 changed the segmenter, so we reran both models.
- Plain transcripts are identical to `120087f` on 334 / 340 clips (Mini) and 338 / 340 (Transcriber).
- Every change is in Common Voice or accented clips, where utterances now close differently: 3 better, 2 worse, 3 equal.
- Mini's pooled numbers are unchanged: core WER 23.4 % plain, 22.5 % written; Digits 28 % → 72 %.
- The Transcriber's improve slightly: core WER 9.9 → 9.7 % plain and 8.8 → 8.6 % written; Common Voice 13.8 → 13.3 %.
- Digits are unchanged.

- **Their headline columns:**
  - "WER EN w/o acc." (Mini 23.4 %, Transcriber 9.9 %) is the clip-pooled mean over their *core* suites. Those are all but
    accented, hard and code-dictation.
  - Their "WER EN" column (30.0 % / 17.7 %) doesn't match any pool of their published rows that we tried. All 10 suites give
    30.5 % / 17.2 %. We report the pools we can reproduce.
- **Written form:**
  - Digits rises from 28 % to 72 % (Mini) and from 4 % to 76 % (Transcriber).
  - Number-formatting WER falls from 26.5 % to 16.4 % (Mini) and from 27.1 % to 12.4 % (Transcriber).
  - The core WER falls by 0.9 points (Mini) and 1.1 points (Transcriber).
  - For comparison, TypeWhisper's published numbers: Parakeet v2 has Digits 60 %, number-formatting WER 20.4 %, core WER 5.8 %.
- **Where written scores worse:** on suites whose references spell numbers as words (LibriSpeech, punctuation-formatting,
  code-dictation). Their normalizer turns "3,000" into "three zero", "15th" stays "15th" against "fifteenth", and "1945"
  becomes "one thousand nine hundred forty five". So the written form can lose a few tenths of a point on spoken-form
  references. That is a scoring artifact, not a recognition change. Score `transcript` against spoken references and
  `transcript_written` against written ones.

**Speed:** their protocol from `bench/src/speed.ts`: 30-clip speed set, 3 warm-up clips, 3 passes, median per clip, speed
factor = audio s / processing s. CPU, 6 threads, M3 Pro. TypeWhisper used an M5 Pro with 18 cores.

| | TypeWhisper (M5 Pro) | 1.2.0 here (M3 Pro) | 1.2.1 here (M3 Pro) |
|---|---|---|---|
| Arc Mini | 30.1× | 16.4× | 16.2× |
| Arc Transcriber | 3.2× | 2.7× | 2.7× |

- 1.2.1 doesn't change CPU speed.
- The M3 Pro is an older, smaller chip than their M5 Pro, so the absolute speeds aren't comparable. We didn't time other models
  here to measure the machine gap.
- The machine was shared with other jobs (load average 3–5), so these speeds are lower bounds.

Run: `python -I run_suites.py --tw TW --backbone mini --out rows.jsonl --speed speed.json`, then
`python -I score_suites.py --tw TW --rows mini_121=rows.jsonl`.

## 2. Real numbers we didn't write: Earnings-22

**Data:** Earnings-22 (Rev.com, CC BY-SA 4.0): real earnings calls, mostly non-native English speakers. We used the public
Hugging Face copy `distil-whisper/earnings22` (`chunked`, written-form references with digits).

`fetch_earnings22.py` picks the subset:
- 26 calls, one 100-segment row group each (seed 0), segments of 2–20 s with no markup;
- every segment whose reference has a digit (528), plus 96 digit-free controls;
- 624 segments, 91 min, 900 reference numbers.

**Not used:**
- SPGISpeech is gated on Hugging Face (login and terms required), so we skipped it.
- Earnings-21 has only full calls (no segments), so we didn't use it.

**Normalizers:**
- **whisper:** `EnglishTextNormalizer` with the spelling map (`whisper-normalizer`, as the Open ASR Leaderboard uses). It
  converts number words, so it hides most format differences.
- **strict:** lower case, punctuation removed, thousands commas dropped. It does *not* convert number words, so "forty five"
  against "45" counts.

**Numbers:**
- Every reference token with a digit is reduced to its value ("$45", "45%", "21st" → 45 / 45 / 21) and matched against the
  hypothesis as a multiset. This gives recall and precision.
- TW-digits is TypeWhisper's verbatim rule per utterance.

| Earnings-22, 624 segments | Mini plain | Mini written, main | Mini written, fix | Transcriber plain | Transcriber written, main | Transcriber written, fix |
|---|---|---|---|---|---|---|
| WER, whisper normalizer | 42.14 | 42.21 | 42.16 | 17.14 | 17.02 | 17.02 |
| WER, strict normalizer | 48.46 | 42.13 | 42.12 | 30.72 | 17.94 | 17.94 |
| Number recall | 25.0 % | 58.8 % | 59.1 % | 5.4 % | **79.9 %** | 79.9 % |
| Number precision | 61.6 % | 68.0 % | 68.4 % | 89.1 % | 87.9 % | 87.9 % |
| TW-digits (utterances, all numbers verbatim) | 20 % (106/528) | 47 % (246/528) | 47 % (247/528) | 4 % (20/528) | **70 % (368/528)** | 70 % (368/528) |
| Digit-free controls, WER whisper / strict | 34.37 / 34.79 | 34.37 / 34.79 | | 15.66 / 15.84 | 15.66 / 15.84 | |

- **With the Transcriber:**
  - The written form finds 80 % of the reference numbers, up from 5 % with plain.
  - Strict WER falls from 30.7 % to 17.9 %, close to the 17.1 % whisper-normalized WER.
  - Precision stays at 88 %.
- **Mini's low precision (68 %) is recognition, not formatting.** On this accented, spontaneous speech Mini's WER is 42 %. Its
  wrong words become wrong digits ("twenty cows" → "20 cows", "seventy percent" for "7%").
- **The whisper-normalized WER doesn't move** (±0.1). The written form changes the format, not the words.
- **Controls:** on the 96 digit-free segments the written form changed 3 utterances for Mini and 0 for the Transcriber. WER is
  unchanged.

**Hand check of wrong conversions** (the key risk):
- **Method:** we read 30 random utterances per model whose written form differs from plain (seed 0), and asked whether each
  rewritten number is the value of the words Arc heard. `results/earnings22_handcheck_*.json` holds them.
  - **Mini: 0 wrong values out of 30.** One count was written year-style ("two thousand and eight" → "2008").
  - **Transcriber: 0 wrong values out of 30.** Two style points: "thirty four cent." → "34 cent.", and
    "three and five million" (meant 3.5 M) → "three and 5 million".
  - Every number that is wrong against the reference was already wrong in the words. For example, "twenty twelve" → "2012"
    when the speaker said "2021".
- **All-row scan for invented clock times** (every written `h:mm` not in plain, across all 3,000+ written outputs):
  - main has **5 invented times, all Arc Mini on Earnings-22**: "at 1:12 days" (×2, from "one hundred twelve"), "at 1:50"
    (from "one hundred fifty"), "by 2:09" (from Arc's own "200 9"), and "as at 20:18 end of the year" ("twenty eighteen" with a
    time tag).
  - On TypeWhisper's hard-en suite Mini wrote "one thousand four hundred twenty five Oak Street" as **"14:25 Oak Street"**.
  - The Transcriber wrote "one hundred and four hundred twenty five" as **"10,425"**.
  - With the fix: 0 invented times and no malformed numbers.

## 3. LibriSpeech regression (plain speech)

**Data:** 500 random utterances (seed 0) from each of test-clean and test-other (openslr.org, CC BY 4.0). Whisper
normalizer and strict normalizer, corpus WER (%).

| | Mini plain | Mini written | Transcriber plain | Transcriber written |
|---|---|---|---|---|
| test-clean, whisper | 8.81 | 8.84 | 3.56 | 3.60 |
| test-clean, strict | 9.15 | 9.32 | 3.82 | 4.21 |
| test-other, whisper | 24.43 | 24.44 | 5.90 | 5.91 |
| test-other, strict | 24.35 | 24.57 | 6.33 | 6.71 |
| utterances changed by the written form | 13 + 16 | | 20 + 20 | |

- The Transcriber is 3.56 % on test-clean here, against 4.0 % on TypeWhisper's 50 clips (their normalizer, mean over clips).
  Same model; a different sample and scoring.
- LibriSpeech references spell every number as words. So any digit the written form adds is an "error" here: it adds 0.01–0.03
  points with the whisper normalizer, and 0.2–0.4 points with strict.
- Recognition is unchanged.
- The fix changes no LibriSpeech output.

## 4. CPU vs MPS, end to end

**Method:** `device_check.py` on `07cde07`. Clips are consecutive LibriSpeech test-clean utterances of one chapter: 15 s,
60 s and 61 s. The third clip was meant to be 180 s but was capped by the chapter's length in our 500-utterance subset, so we
have two lengths, not three. For each device we ran `predict()` and `stream()` (100 ms chunks fed as fast as the stream takes
them, then `flush()`), with default threads.

**Outputs:**
- **Identical on CPU and MPS for both models, on every clip, in both predict and stream**, for the plain and the written
  transcripts (word difference 0).
- Stream output is identical to predict output on the same device.
- This is the first end-to-end run of Arc Transcriber on MPS. Before, only a stand-in backbone was tested.

**Times:** wall seconds. The machine was heavily shared during this run (load average 25–45 from other jobs), so the CPU
times are inflated and only the order of magnitude means anything.

| Model, clip | CPU predict | MPS predict | CPU stream | MPS stream |
|---|---|---|---|---|
| Mini, 15 s | 3.8 | 4.4 | 8.0 | 24.5 |
| Mini, 60 s | 12.9 | 10.7 | 30.7 | 58.6 |
| Mini, 61 s | 14.9 | 8.3 | 32.3 | 53.0 |
| Transcriber, 15 s | 21.2 | **2.3** | 7.6 | 3.7 |
| Transcriber, 60 s | 67.8 | **8.2** | 27.1 | 11.1 |
| Transcriber, 61 s | 74.5 | **8.6** | 22.7 | 13.2 |

- **Arc Transcriber on MPS** runs `predict()` at about 7× real time (8.2 s per 60 s). Even allowing for the load, MPS is the
  way to run the full model on a Mac.
- **Arc Mini** gains little from MPS in `predict()`, and its streaming is 2–3× slower on MPS than on CPU (the decoder steps one
  token at a time). Keep Mini on the CPU for live use. `device="auto"` picks MPS for both models, which is wrong for Mini's
  streaming.
- **CUDA:** untested (no GPU here).

## 5. Streaming and the CLI give the same written text

**Method:** 5 real recordings, each made by joining consecutive Earnings-22 segments of one call (26–65 s, 5.1 min in total).
`stream_cli_check.py` compares six written outputs against `predict()["transcript_written"]` (word edit distance):
- `transcribe(written=True)` `text_written`;
- the stream's finals (`transcript_written` of each `feed()` and `flush()`, 100 ms chunks);
- `flush()["transcript_written_all"]`;
- `arc-transcribe --written` in txt / srt / vtt;
- the json `transcript_written` and per-line `written`.

**Results on `120087f`, 5 recordings, both models:** word difference **0 for every path on every recording**.
- Paths compared: `transcribe(written=True)`, the stream finals, `transcript_written_all`, CLI txt / srt / vtt / json.
- The stream's plain finals are also identical to `predict()`'s.
- The checks aren't trivial. The written text differs from plain on all 5 recordings, and has 15–36 digit characters per
  recording.
- The stream gave 6–16 finals per recording, and the same number of written finals.

On `07cde07` we reran 3 of the recordings with both models: still 0 differences on every path.

One plain-transcript oddity, not from the formatter: Mini decoded "GH100,000" as "10,00,00,00,0" (MTN Ghana call). The
written form left it unchanged.

## 6. Date records

**Method:** 15 sentences (`data/dates.tsv`), rendered with two macOS voices (Samantha, US; Daniel, UK). The renders are
written to files; nothing is played. A sentence counts as right when a `numbers` record of kind `date` has the expected
`iso`.

| | Arc Mini | Arc Transcriber |
|---|---|---|
| Sentences with the expected date record | 19 / 30 | 21 / 30 |
| Month kept when Arc heard day + month ("March third", "third of March", "21st of July") | all | all |

**What works:**
- "March third" → `heard` "March third", `date` {month 3, day 3}, `iso` "--03-03".
- "the third of March" → `heard` "third of March", same date.
- "October fifteenth, twenty twenty five" → "2025-10-15".
- "nineteen eighty seven" → "1987".
- "March 3rd 2024" (US voice) → "2024-03-03".

**What doesn't** (the 9–11 misses):
- **Year after a sentence break:** "January 1st, 2020" and "February twenty ninth, 2024". Both models put a period before the
  year ("January 1. 2020"), so the year becomes its own record. `extend_date` (like the written form) doesn't join across `.`
  or across a new utterance.
- **UK reading "the 3rd of March 2024":** this gives two overlapping records, "3rd of March" (--03-03) and "March 2024"
  (2024-03). `extend_date` doesn't join a year after "of <Month>".
- **"the first of every month":** a `date` record with `heard` "first of", `date` null. A trailing "of" is kept, and it isn't
  a date.
- **"March 2021"** (no day): Mini gave no date record; the Transcriber (UK voice) gave only "2021".
- **Recognition misses:** "July 23" for "July twenty first", "February 20" for "February twenty ninth".

## 7. Current main `07cde07` (PR #5) against `120087f`

PR #5 changed the segmenter (short utterances close after 0.8 s), `rewrite()`, the switch interval, `kb_snap`, name linking,
`command_complete_hold_s` and per-domain masks. We checked that these changes don't regress the measurements above:

| Check | Result on `07cde07` |
|---|---|
| TypeWhisper suites, plain | identical on 334 / 340 (Mini) and 338 / 340 (Transcriber) clips; core WER Mini 23.4 % (same), Transcriber 9.7 % (was 9.9 %) |
| TypeWhisper suites, written | Mini 22.5 % core, Digits 72 % (same); Transcriber 8.6 % (was 8.8 %), Digits 76 % (same) |
| Earnings-22 / LibriSpeech | LibriSpeech test-clean Mini (first 150): identical. Earnings-22 got **worse**: Mini 42.14 → 42.34 % (54 / 624 changed: 24 worse, 13 better); Transcriber 16.17 → 16.72 % on the first 150 (10 changed). See below |
| CPU vs MPS | identical outputs (section 4 was run on `07cde07`) |
| Streaming and CLI written text | identical across all paths (3 recordings × 2 models, word difference 0) |
| Date records | same records on all 30 sentences for both models (Mini 19/30, Transcriber 21/30) |
| `pytest` (this branch on `07cde07`) | 82 passed, 2 skipped; with `ARC_TEST_PARAKEET=1`, the full-model golden and smoke tests pass too (13 passed) |

**Regression from the 0.8 s close on real speech:**
- Nearly every Earnings-22 change on `07cde07` is a new short utterance at the start of a segment, where a clipped word or a
  breath now closes on its own. It decodes as a confident filler or an invented phrase:
  - Transcriber: "Go ahead, sir. Good morning, ladies and gentlemen…", "Thank you, Trey. So just…", "Yes. Total cost…",
    "September. We continue…";
  - Mini: "Hey, hey, hey.", "you three.".
- 31 of Mini's 54 changed outputs have more utterances than before.
- It is the same mechanism as fillers on short non-speech clips, now reached in `predict()` on real audio.
- Possible fix: gate the short close on `min_voiced_s` or a speech-confidence check, or apply it only in live streams.

**Short-command final latency**: 12 commands × 2 macOS voices, real-time pace, 6 s of
silence after, Arc Mini, CPU.

| | `120087f` | `07cde07` |
|---|---|---|
| Finals that arrived within the 6 s tail | 15 / 24 | **24 / 24** |
| Final after the speech, median / p90 / max | 0.67 / 0.87 / 1.79 s (arrived ones only) | **0.77 / 1.18 / 1.35 s** |
| Commands under ~1 s ("Pause.", "Say that again.", "Open WhatsApp.", ...) | never (until `max_s` = 15 s or `flush()`) | 0.99–1.35 s |
| Utterance record after the speech, median | 0.79 s | 0.79 s |

So the 0.8 s close works. A short command's final now arrives about 1.0–1.3 s after the speech: the 0.8 s pause plus the decode.
Longer commands are unchanged (about 0.6–0.8 s). The machine was shared during this run, so the decode share may be smaller on
an idle machine.

## Bugs found and fixed

All three are in `arc_audio/written.py`. Tests are in `tests/test_written.py`.

1. **Numbers with "hundred" / "thousand" became clock times.** `hour_minute` read any number with "hundred" as HHMM:
   - "one thousand four hundred twenty five Oak Street" → "14:25";
   - "at one hundred twenty five" → "at 01:25";
   - "around four hundred thirty people" → "around 04:30 people";
   - "at one hundred twelve days" → "at 1:12 days" (Earnings-22).

   Fix:
   - "thousand" or a larger scale is never a time;
   - without a time tag, the hour before "hundred" must be 10–24 ("fourteen hundred"). A tagged "six hundred forty five" is
     still 06:45.
2. **Malformed scale sequences were multiplied into an invented number.** "one hundred and four hundred twenty five" →
   "10,425"; "two thousand three thousand" → 5,000. Fix: `well_formed()` rejects a "hundred" after a value ≥ 100, and a scale
   that repeats or rises. The words are kept, or split at "and" ("100 and 425").
3. **A bare "twenty X" with a time tag became 20:xx.** "as at twenty eighteen end of the year" → "20:18". Fix: two-group times
   with hour ≥ 20 need "a.m." / "p.m." / "o'clock" / "hundred"; otherwise the words stay.

Effect on the measured sets:
- TypeWhisper hard-en, Mini: 26.4 % → 25.7 %.
- TypeWhisper number-formatting, Mini: 16.7 % → 16.4 %.
- Earnings-22 Mini: 5 invented times → 0.
- No other output changed on Earnings-22 (Transcriber), LibriSpeech, or the remaining TypeWhisper suites.
- `pytest`: 82 passed, 2 skipped; the 2 full-model tests pass with `ARC_TEST_PARAKEET=1`.

Not fixed yet:
- the date-record joins in section 6;
- tagged ordinals under 10 written as digits ("fourth quarter" with a date tag → "4th quarter"; "first point" → "1st point");
- currency without a symbol mapping ("34 cent.").

## Files

| File | What it does |
|---|---|
| `twscore.py`, `check_port.py` | TypeWhisper's normalizer, WER, Digits and pooling in Python; checks the port against their published rows |
| `run_suites.py`, `score_suites.py` | their 10 suites and speed protocol; per-suite tables |
| `fetch_earnings22.py`, `pick_librispeech.py` | data subsets (audio stays local and is deleted after the run) |
| `run_set.py`, `score_set.py`, `rewrite.py` | predict over a manifest; WER (two normalizers), number recall / precision; re-render the written form with another `arc_audio.written` from the saved tagged words |
| `device_check.py` | CPU vs MPS, predict and stream, 3 clip lengths |
| `stream_cli_check.py` | written text through predict / transcribe / stream / CLI |
| `run_dates.py`, `data/dates.tsv` | date records |
| `results/*.json` | summaries, hand checks, and the device, stream and date runs (no audio; Earnings-22 text excerpts under CC BY-SA 4.0, attribution Rev.com) |
