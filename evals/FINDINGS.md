# Findings for the Arc team

What building Steer on Arc Mini taught us, each with its evidence and an ask. The sources:
- **Steer**: the app's libarc runtime, Arc Mini 1.1.1.
- **Py 1.1.1**: Python reference runs during development (the signal study, the Pipecat integration).
- **v1.4 demo**: the Arc v1.4 demo server's replay data, used in the ASR comparison film.
- **rerun**: this PR's scripts, `arc_audio` 1.2.0 with Arc Mini 1.1.1 files, CPU, 2026-10-29.
- **Mac 1.6.6**: Steer for Mac 1.6.6 test runs (libarc, Arc Mini 1.1.1), 2026-10-30. Behaviour only.
- **TypeWhisper**: the independent TypeWhisper benchmark (10 English suites, 340 clips, Mac mini M5 Pro, CPU), with `arc-transcriber`
  1.2.0 and the package defaults (`Arc.predict(wav)`, plain `transcript`).
- **1.2.1**: `arc_audio` 1.2.1 (the written-form PR), Arc Mini 1.1.1 files, M3 Pro, 2026-10-30 (`evals/written_form`).

All speech is synthetic (Steer TTS or macOS voices). A real-voice session is still owed and may change the device-directed and affect conclusions.

## Streaming and API

### 1. A short command's final transcript arrives about 15 s late
- **Evidence (rerun, `api_issues`, `latency_streaming`):** In a stream with 20 s of silence after the speech, "Pause." (0.46 s) got its final transcript at **+14.5 s** and "Open WhatsApp." (0.92 s) at **+14.1 s**. Commands over 1 s were final at +0.4 s. In the real-time run (24 clips, 6 s of silence), **9/24 finals never arrived**: every command under about 1 s. The other 15 arrived at +0.58–0.68 s.
  - The cause is in `arc_audio/transcribe.py` `UttSegmenter.feed`. An utterance closes on a 0.4 s pause only if
    `n - low_run >= min_n` (`min_s` = 1.0 s). A shorter one isn't closed until `max_s` = 15 s, or until `flush()`.
- **Earlier (Py 1.1.1, Pipecat):** Arc's own end-of-utterance final came about 2.7 s after speech ended. So the Pipecat service
  creates a new `arc.stream()` per VAD turn and calls `flush()` (`flush()` takes 155 / 201 ms, median / p90), which gives a 0.42–0.79 s end-of-speech-to-transcript time.
- **Ask:**
  - Close short utterances on a pause too: apply `min_s` when merging, not when closing, or close them after a longer pause.
  - Add `Stream.finalize()` / `end_turn()`, which closes the current utterance without tearing the stream down.
  - Add a public per-call context setter on a running stream.
- **Fixed in 1.2.1:** short utterances close after `short_pause_s` (0.8 s) of silence ("Pause." final 14.6 s -> 0.8 s after the
  speech, espeak voice, CPU); `Stream.end_turn()` / `finalize()`; `Stream.set_context()` on a running stream. In Steer's live path,
  short clips also get no device-directed or intent readout (#25).

### 2. `transcript_rewritten` can duplicate a token
- **Evidence (Py 1.1.1, Pipecat):** "Priya Raymond" was linked to "Priya Raman", but the linked span covered only "Priya". The rewrite gave "Priya Raman Raymond", so the service uses the plain transcript plus `names`.
- **Rerun (`api_issues` #2):** not reproduced on macOS-voice renders of the same sentence. Three voices heard "Priya Raman" whole; the fourth heard "Queen Raymond", which wasn't linked. So the case needs a partial link to occur. The rewrite logic is `arc_audio/api.py` `rewrite()`.
- **Ask:** rewrite the whole heard span of a linked name, or none of it.
- **Fixed in 1.2.1:** whole span or none (the case above now gives "Priya Raman").

### 3. `Arc.stream()` sets the process-wide `sys.setswitchinterval`
- **Evidence:** `arc_audio/api.py` sets it while the background worker runs and restores it on `flush()`. Rerun: 0.005 before, 0.0005 while a stream runs, 0.005 after `flush()`. An unflushed stream leaves it changed.
- **Ask:** make it opt-in. In a host process such as Pipecat, every thread is affected.
- **Fixed in 1.2.1:** opt-in (`switch_interval_s`), restored on `flush()` / `close()` / `with` / garbage collection / exit.

### 4. Two device-directed numbers per utterance
- **Evidence:** `utterances[].tags['discourse.assistant.device_directed']` and `transcript[].device_directed.p` disagree on the same clip. "Sleep in twenty minutes.": tag 0.32, head 0.0006. Across the operating-room scene, the tag was 0.35–0.91 and the head 0.000–0.014. On the context-memory sessions, the head's fresh AUC was 0.529 against the tag's 0.440. With Samantha alone it was 0.842, so the head may carry signal on a different scale. libarc exposes one number, which behaves like the tag.
- **Ask:** document which one apps should use and how each is calibrated, or merge them.
- **1.2.1:** documented in the README ("Device-directed: which number"): use the utterance tag; the head is a raw score with its own
  threshold. Neither is calibrated for always-listening use (#6).

### 5. Port the name linker and a frame-level API to C
- libarc, Steer's C port, has no name linker. Speaker and item pooling aren't ported either.
- Apps want the probability rows (frame, span, utterance, events) as a ring buffer, not only records.
- **Ask:** treat libarc (being proposed as `c/` separately) as a first-class runtime, with the linker and a frame API.

## Device-directed and wake names

### 6. Device-directed doesn't separate commands from other speech
- **Steer (libarc):** fresh-state AUC was 0.418 for commands vs speech to someone else, across 210 utterances.
  - Directed without the name ranged 0.29–0.97 on 32 commands ("Say that again" 0.29).
- **Py 1.1.1 signal study:** commands scored 0.26–0.98; narration 0.67–0.84; two-person dialogue 0.75–0.88.
- **Operating-room film (libarc):** the nurse's lines scored **0.74–0.93**, the surgeon's commands **0.51–0.88**.
- **Pipecat session (Py 1.1.1):** 0.0005–0.26 for speech addressed to the agent.
- **Rerun (`device_directed`, `context_memory`):** In the operating room, commands with the name scored 0.58–0.84 (mean 0.73) and staff-to-staff cross-talk 0.50–0.91 (mean 0.78, all 18 ≥ 0.5). AUC was **0.33** (tag) and 0.34 (head). On the context-memory sessions (4 voices), commands vs speech to someone else gave tag AUC **0.440**.
- **Mac 1.6.6 (cabin noise):** clearly addressed "Steer, …" commands read device_directed **0.27–0.45**, under the 0.5 bar.
- **Ask:**
  - Train with multi-talker negatives: narration, TV, staff talking to each other, speech to someone in the room.
  - Publish an operating point. Steer's target for always-listening is ≥ 95% precision at ≥ 80% recall.
  - Until then, Steer and the films require a device name **and** directed ≥ 0.5.

### 7. Carrying the trunk's state across utterances made things worse
- **Steer (libarc):** carried vs fresh, with 2 s room gaps:
  - device-directed AUC 0.418 → 0.338;
  - act right 97 → 93/100.
  - Over one long session, the score for speech to someone else drifts up (0.71 → 0.83).
- **Rerun (one `predict()` over the session vs per utterance):** Fresh vs carried: tag AUC **0.440 → 0.347**, head AUC 0.529 → 0.426, act right 78 → 75/80. Speech to someone else drifts up (tag mean 0.80 → 0.86). Same direction as libarc.
- **Ask:**
  - Say whether the trunk is meant to continue across separate push-to-talk clips.
  - Say how gaps should be handled (feed room silence? reset?).
  - Consider training with gaps. Steer ships with the state off; the switch is there to retest with Edge.

### 8. Some wake names are almost never heard
- **Film screen (libarc, primed):** "Scope" was missed 8/9 (heard as "Let's go"); "Nova" and "Camera" 9/9; "Atlas" and "Display" 7/9.
- **Steer:** "Steer" was heard in 9/12 clips on macOS voices (lost as "Sleep in…", "Yeah. Save…").
- **Rerun (unprimed → primed, 9 each):** Scope 1 → 7, Nova 3 → 9, Camera 0 → 6, Atlas 5 → 8, Display 3 → 4, Steer 3 → 5. Unprimed, "Scope" was heard as "Let's go", "Camera" as "Tamara", "Display" as "Just play" and "Steer" as "Yeah" / "Steel". One primed "Atlas" clip (Aman voice) came back as a long hallucinated sentence: "Here, Chloe's thumb will be suggesting Pashchini of three keys…".
- **Mac 1.6.6 (cabin noise, 180 clips):** the name came out **"Dear" 59/180** against "Steer" 51/180; the rest were "Yeah", "Here",
  "Dave", "Diaz". In noise the wake name is missed more often than it is heard.
- **Ask:**
  - Ship a name-screening tool, or guidance on which names work.
  - Make priming dependable for one-word names.
  - Consider a keyword/wake head conditioned on the name list.

## Text context and names

### 9. Text context works and is worth keeping
- **Steer (libarc):**
  - All names in the prompt, plus bias and previous text: WER **30.0% → 20.2%**, names right **42.9% → 68.7%**, continuations 37.9% → 23.2%.
  - Bias costs nothing. The names prompt costs about +200 ms per utterance.
  - Previous text of 8 words is as good as 40 and far cheaper (151 vs 346 ms).
  - Unrelated previous text doesn't hurt commands.
- **Rerun (`context_memory`):** With 4 macOS voices: WER 44.4% (none) → 33.0% (all names + bias + previous text); names right 28.6% → 56.9% (57.7% with the relevant names prompt); continuations 51.3% → 41.4%. Bias alone: 35.8% WER, 50.0% names. Same direction as Steer, at a higher error level on these voices.
- **Ask:**
  - Make the prompt cheaper. Its cost grows with its length, and live use re-decodes every 0.5 s.
  - Keep previous text and the 60 s window as public API.
  - libarc's `auto_prev` kept clip-local times, so it never fired across separate calls. Steer passes previous text itself.

### 10. Arc's name linker doesn't find book titles
- **Python, 2026-10-29 (Steer session), 152 title/author clips:**

  | Setup | Right | Wrong links |
  |---|---|---|
  | 2,000 most-read titles as the list | 34% (52/152) | 9 |
  | same, with `kb_snap` (144 clips) | 35% (51/144) | **63** |
  | top 20 live candidates as the list | 49% (title 48/120, author 26/32) | 19 |

  - Steer's own catalog search with Arc's live prompt gets 78%.
  - The linker linked "Sherlock Holmes" to the author "T. Rice Holmes".
  - `kb_snap` matched "Read Frankenstein" to *The Pilgrim's Progress by John Bunyan Every Child Can Read*, on the word "Read".
- **Rerun:** With the 2,000-title list: 32% (49/152) and 8 wrong links. With `kb_snap`: 34% (52/152) and **70** wrong links. With the top 20 live-RAG candidates from the stand-in search: 53% (title 55/120, author 25/32) and 13 wrong links. The median time was 1.9 s per clip with the 2,000-entry list, against 0.46 s with 20 entries.
- **Steer, 2026-10-30:** of the 21 book requests Steer still gets wrong, **19 never had the right title among the candidates** Arc
  was given. The linker couldn't have linked them; the loss is upstream, in what reaches the candidate list and the prompt.
- **Linker input (1.2.1, code):** names are encoded as at most `NAME_MAX_BYTES` = 40 bytes (`model.encode_names`, head kept).
  **28%** of the 2,000 most-read titles in `book_search/data/catalog.tsv.gz` are longer. Long titles are cut, and titles that share a
  40-byte head look the same to the linker. The linker also scores every name passed to it, with no shortlist (#23).
- **Ask:**
  - Train the linker on titles (`work_product`) at catalog scale, tens of thousands of entries.
  - Make `kb_snap` ignore command words and stopwords.
  - Return a ranked candidate list with scores, not only the one link.
- **1.2.1:** `kb_snap` ignores command words and stopwords; name records carry ranked `candidates` (and `snap_candidates`). Training the linker on titles is still open.

### 11. The live names prompt is the part that helps search
- **Steer:**
  - Right book first: 76% with no context → **81%** with live context plus a relisten.
  - Title words in the transcript: 56% → 70%.
  - Catalog lookup 1.8 ms. The bench's "7 ms" for the final times only the relisten step; the final decode is about 122–135 ms.
  - Arc fixed "Mr. High" → "Mr. Hyde" because the title was in its prompt.
  - A static list of 1,900 titles helped words (64% → 71%) but not the right book (76% → 76%), and costs 230 ms per request.
- **Rerun (stand-in search):** Right book first: 68% (none) → 72% (static) → 73% (live) → 75% (live+rag2). Title/author words heard: 56% → 76%. The words improve more than the search does.
- **Ask:** keep the names prompt fast and large enough for about 40 candidates. That is the pattern that works for big catalogs.

### 12. Names in the v1.4 demo replay
- **v1.4 demo, from the replay data:**
  - "Cavaliers" and "DeAndre" were flagged as not on the roster, though Cleveland Cavaliers and De'Andre Hunter are on it. "Miami" was classed as a person.
  - Confirmation is slow: median 1.6 s after the word, p95 **9.7 s**. "Sam Merrill" went through 7 revisions.
  - Odd early guesses: "100" for 19, "Hale" for Allen, "Mike" for Miami.
- **Ask:**
  - Check the not-in-list decision against aliases and partial names.
  - Bound the confirmation lag.
  - Hold back low-confidence first guesses from the live view.
- **1.2.1:** the list check (aliases, partial names, punctuation) runs before `not_in_list`; `stream(live_hold=...)` holds back low-confidence and very early guesses. The confirmation lag is not bounded yet.

## Recognition

### 13. Loud rooms and one-word commands
- **Steer (7,648 clips):**
  - Arc was 96.1% right in a quiet room and **83.0%** in a loud one (−28 dBFS). whisper tiny.en got 99.0% and 96.7% on the same clips.
  - The losses are one-word commands: "Pause" → "Oz" / "Of course"; "Resume" → "was you"; "Mute Telegram" → "Newt Telegram".
- **Rerun (`commands`):** 3,344 clips, 4 macOS voices: 74.9% right overall; 84.9% quiet, 75.1% noisy, **61.5%** loud room, 78.2% soft voice. By voice: Samantha 86.5%, Aman 82.1%, Daniel 75.8%, Flo 55.4%. "Mute Telegram" → "New Telegram" in all 32 clips.
- **Mac 1.6.6 (noise):** "chapter" came back as chat / chap / challenge / channel / check, and "stop" as "stopped".
- **Ask:** use noise augmentation weighted to short commands. The `assistant_noisy` domain exists; say whether apps should switch to it from the noise level.

### 14. Whispers and low-level speech are the weakest input, and there is no vocal-effort output
- **Bedside film (libarc):**
  - "Head up a little" came back as "Hit up" / "Bedup".
  - "Raise my head a little" once came back as "Wraith Mei had a little".
  - Short unprimed commands sometimes came back empty.
- **Rerun (`whisper_noise`):** macOS Whisper voice: **58.9% WER**, against 20.8% for the same lines in normal voices. Normal speech at −42 dBFS: 22.6%. So the trouble is the unvoiced whisper, not the low level. Examples: "Pause." → "Oh, shit."; "Read faster." → "Brad's master."
- Steer detects whispers from the level, 6 dB under the person's usual level. It whispers back at ≥ 12 dB under.
- **Ask:**
  - Add whispered and low-level speech to training and calibration.
  - Add a whisper / vocal-effort tag. Answering in kind is a product feature.

### 15. Intents are smart-home classes
- **Evidence:**
  - "Say that again" → `calendar_set` 0.87.
  - "I need the nurse" → `iot_coffee`; "show the pre-op scan" → `takeaway_order`.
  - Only "What time is it?" → `datetime_query` 0.98 is reliable for Steer.
  - v1.4 demo: assistant intent accuracy 0.45, voice-assistant replay WER 0.60.
- **Rerun (clinical lines):** "I need the nurse" → `lists_createoradd` / `calendar_set`; "Raise my head a little" → `calendar_set` / `lists_query` / `social_post`; "Show the pre-op scan" → `qa_stock` / `play_music`. All at p ≤ 0.36.
- **Ask:** a custom-intent head trained from an app's command list, or an embedding of the utterance that apps can classify themselves.

## Events

### 16. `command_complete` is good in short clips but can fire mid-command
- **Py 1.1.1 signal study:** caught 9/10 clean, 8/10 at 5 dB SNR, 9/10 on a phone line; about 1 false event per run.
- **v1.4 demo:** recall 1.0, precision 0.91. But "set alarm at nine am | on next sunday" got a complete at "am".
- **Rerun (`latency_streaming`):** In 24 real-time streams, the event came median +0.18 s after the speech ended (p90 +0.36 s). It was **missing on 5/24**, including the long "Set an alarm at nine am on next Sunday". One fired 0.51 s before the end, on the last word.
- **Ask:**
  - Calibrate against trailing modifiers ("…on next Sunday", "…for an hour").
  - Expose a hold-off parameter.
- **1.2.1:** `command_complete_hold_s` (default 0). Calibration against trailing modifiers is still open.

### 17. Labels to mask or calibrate in the `assistant` domain
- **Py 1.1.1 signal study, on clean synthetic speech:**
  - profanity (experimental) fired at p > 0.9 everywhere, silence and music included;
  - bio events fired 12–29 times per clip;
  - band noise was read as crowd, cheer, laughter and whistle;
  - diarization merged two clear voices into one speaker;
  - overlap and background speech scored 0.04 / 0.03 during a barge-in;
  - a command spoken over Steer's own reading wasn't heard at all, so barge-in needs echo cancellation before Arc.
- **Working well:**
  - telephone (0.57) and clipping (0.51) fire only on their own condition;
  - speech onset in 0.3–0.6 s with no false frames in silence or music;
  - numbers with values ("twenty minutes" → 1200 s, even on a phone line).
- **Ask:**
  - Mask profanity and bio events in `assistant`, or calibrate them.
  - Say whether the single-speaker diarization is expected in `assistant`.
  - Affect has no trained output in Mini. It is the most valuable missing signal for Steer's delivery.
- **1.2.1:** masked in `assistant` / `assistant_noisy` by default (`masks`). Calibration is still open.

## Benchmarks, output format and runtime

### 18. No written-form output, so benchmarks score Arc's numbers as words
- **TypeWhisper:** "numbers come out as words ('two thirty in the afternoon', 'plus one two one two …')". Digits EN: **Arc Mini 28%,
  Arc Transcriber 4%**, against Parakeet-TDT 0.6B v2 60%, Whisper small 64%, Apple SpeechAnalyzer 84%. On their number-formatting
  suite, WER was Mini 26.5%, Transcriber 27.1%, Parakeet 20.4%.
- **Cause:** the decoder writes the spoken form, and the package had no inverse text normalization. Arc already knows where every
  number is and what kind it is (inline NUMBER_* tags and the `numbers` values). But only names were rewritten
  (`transcript_rewritten`; `NUMWORD_RE` skips numbers).
  - A normalizer doesn't undo it either. Whisper's `EnglishTextNormalizer` turns "two thirty" into "230" and "March third" into
    "march 3rd", but "2:30" into "2 30" and "March 3" into "march 3".
- **1.2.1 (`evals/written_form`: 40 number-heavy sentences × 3 macOS voices, Arc Mini, scored against written references):**

  | | `transcript` | `transcript_written` |
  |---|---|---|
  | Digits found | 22.2% | **74.1%** (76.3% with single digits as digits) |
  | Whole sentence right | 18.3% | 52.5% |
  | WER (Whisper normalizer) | 27.2% | 24.2% |

  `transcript_written` is built from Arc's own tags. On 73 LibriSpeech clips, WER is unchanged (15.48%).
- **Most remaining misses are recognition errors, not format:** "102 to nine two"; "two point high kilograms"; "$5. $5." for "$3.99";
  "Please remind my old kids" for "Please arrive by 8 p.m.".
  - Some are tagging: "two thousand four hundred employees" was tagged NUMBER_MONEY.
  - Some are style: "the 15th of October" for "October 15"; "21st" after a month.
- **Ask:**
  - Treat the written form as a product output, and decide its style: US or UK dates, and when single digits are words.
  - Put it in the model card's benchmark recipe: score `transcript_written` against written references.
  - Consider a written-form decoder target, or a written-form head on the tags, so the C port and other runtimes get it too.
    libarc has no `arc_audio/written.py`.
  - Tell benchmarkers which field to score. TypeWhisper scored `transcript`, which was what the README suggested then.

### 19. A date record lost its month
- **Evidence (1.2.0):** "The meeting moved to 2:30 on March 3rd" (macOS voice). The transcript's NUMBER_DATE tag covered "March
  third", but the `numbers` record had `heard` "third" and value 3.
  - `tidy_number` trimmed `heard` to number words, so "March" went, and nothing kept the month.
  - "the third of March" was worse: the tag covered only "third".
- **1.2.1:** `extend_date` keeps the month (and "of March"). It adds `date` {year, month, day} and `iso` ("--03-03", "2024-03-03",
  "2024"). `value` keeps its meaning (the day, or the year) for existing clients.
- **Ask:**
  - Decide the date value in the schema: the ISO string, or year / month / day fields. One number can't hold "March 3".
  - Port the fix to libarc (`c/src`, its `tidy_number` / `extend_number`). The golden clip has no date, so `make parity` can't catch it.
  - Train NUMBER_DATE spans to include "of <Month>" when the ordinal comes first.

### 20. `device="mps"` failed, so Arc ran CPU-only on a Mac
- **TypeWhisper:** "arc_audio … fails on MPS with 'Passed CPU tensor to MPS op'". Both models ran on the CPU: Mini at 30× real time,
  Transcriber at 3.2×.
- **Causes (reproduced on an M3 Pro):**
  - Arc Mini: the pooling carry is float64 on the device (`model.zero_carry`), and MPS has no float64.
  - Event head: the tracker built its input on the CPU (`events.EventTracker.step`).
  - Arc Transcriber: `server.ServerFeatures._emit` passed CPU-pooled features to the GPU heads. This gives the reported error, and
    the CUDA path makes the same call.
- **1.2.1:** fixed, and `device="auto"` picks cuda > mps > cpu.
  - Arc Mini on MPS reproduces `tests/golden/mini.json` exactly.
  - A test with a stand-in backbone covers the Transcriber's feature path.
- **Speed (M3 Pro, 60 s of LibriSpeech, Arc Mini):**

  | | CPU, 4 threads | MPS |
  |---|---|---|
  | `predict()` | 4.3 s (13.9× real time) | 4.9 s (12.2×) |
  | Streaming, 100 ms chunks | 11.9 s | **43.8 s** |

  MPS gives Mini nothing: the model is small and the decode goes step by step.
- **Arc Transcriber** couldn't be timed: the 2.5 GB Parakeet download was over this run's disk budget.
  - Its encoder alone, with random weights of the same shape, takes 1.2 s per 38 s window on the CPU and 0.24 s on MPS. That is
    about 2.4 s per minute of audio.
  - TypeWhisper measured roughly 19 s per minute, so most of the Transcriber's CPU time is outside the encoder: the decoder, beams
    and alignment.
- **Ask:**
  - Run CI on macOS arm64 with `device="mps"`, and on CUDA for the full model.
  - Profile Arc Transcriber. If the decoder and alignment dominate, batch them or move them to the GPU. Its 3.2× against Parakeet's
    208× is the gap people see.

### 21. Mini's plain-transcript gap is recognition errors, not number format
- **TypeWhisper (WER without the accented suite):** Arc Mini **23.4%**, Arc Transcriber 9.9%, Parakeet-TDT 0.6B v2 5.8%.

  | Suite | Arc Mini | Arc Transcriber | Parakeet v2 |
  |---|---|---|---|
  | LibriSpeech | 9.8% | 4.0% | 1.5% |
  | Common Voice | 42.8% | 13.8% | 6.9% |
  | Code dictation | 37.8% | 15.6% | 5.8% |

  Arc Transcriber trails the encoder it is built on in every suite.
- **1.2.1 (Arc Mini, 73 LibriSpeech clips, 1,169 words, Whisper normalizer):** 15.5% WER, 181 errors. Numbers caused **1** error
  (0.6%). The rest:
  - other substitutions: 47%;
  - near-spellings or homophones: 29% ("kaliko" → "calico" ×5, "Leighton" → "Layton", "idylls" → "idols", "recognising" →
    "recognizing");
  - insertions: 15%, mostly short words ("a", "in", "is");
  - deletions: 8%.

  Rare words, proper nouns and short function words dominate. The written form can't close this gap.
- **On the number sentences:** most of Mini's 24% WER is mishearing. It is worst on one voice (Flo, UK): "three unread and three
  three parties in front of the register" for "352 participants registered".
- **Ask:**
  - Publish Arc Transcriber against its own base encoder on standard sets. It loses about 4 points to Parakeet on the same encoder,
    which suggests the decoder or the heads cost accuracy. Say whether that is the price of the tags.
  - For Mini: more read speech and more accents (Common Voice 42.8%), and British spellings, which the normalizer doesn't map.

## Short clips, noise and live context

### 22. Arc Mini writes a filler on very short silence, noise and key clicks
- **Mac 1.6.6:** 0.4–0.9 s clips of silence, room noise and key clicks came back "Okay." / "Yeah." / "Mm-hmm." in **20 of 24**. Arc's
  own speech frames read under 10% on the same clips.
- **1.2.1 rerun (Python, synthetic 0.4–0.9 s clips):** 13 of 24 were non-empty:
  - digital silence 0/6;
  - room noise at −45 dBFS 4/6;
  - fan noise 4/6;
  - key clicks 5/6.

  The words were "Yeah." ×8, "Oh." ×4 and "Mm-mm." ×1.
- **Impact:** push-to-talk taps and accidental presses produce a "Yeah." that an app may take as an answer.
- **Ask:**
  - Gate the final decode on the speech readout (and `min_voiced_s`) for short clips, or return an empty text with a low confidence.
  - Train with short non-speech clips whose target is empty.

### 23. The context prompt budget is small and silent about what it drops
- **Code (1.2.1, `ctxprompt.build_prompt`):** at most 96 context tokens. Names and terms get up to `kb_max` = 40; previous text gets up
  to `prev_max` = 48.
  - A name that doesn't fit is **skipped, not truncated**, in list order. Later, shorter names can still go in.
  - So which names Arc sees depends on list order and length, and nothing reports what was dropped.
- **Measured (Arc Mini tokenizer):** of 5, 10, 20 or 40 typed book titles (median 8 tokens), **4** make it into the prompt. Of 12
  two-word person names, 7 do. Steer's search wants about 40 candidates (#11).
- **Linker side:** every name passed is scored (there is no shortlist), and names are cut to 40 bytes (#10). With 2,000 titles the
  median clip took 1.9 s, against 0.46 s with 20.
- **Reddit (r/speechtech, the Arc post):** people asked how many names the context takes and what happens to the rest. The README
  doesn't say.
- **Ask:**
  - Document the budget, and return which names made it into the prompt (or rank them by relevance before cutting).
  - Raise `kb_max` for the names prompt, or let apps trade previous text for names.
  - For titles, make `NAME_MAX_BYTES` larger, or keep both head and tail.

### 24. Live-context words don't pull short questions
- **Mac 1.6.6:** "How much butter?" came back "How much longer?" / "How much better?" / "How much a banter?" across voices. This
  happened even with the page's words, "butter" included, in the context.
- **Impact:** the reading app's main use, questions about the page, fails on exactly the word the context has.
- **Ask:**
  - Say how strong the context pull is meant to be on short utterances.
  - Consider a bias weight apps can set, or a boost for context words in two- to four-word questions. Related: #9 (context helps
    longer utterances).

### 25. No device-directed or intent readout for very short clips in the live path
- **Mac 1.6.6 (libarc, live):** clips under about 1 s get a transcript, but no device-directed value and no assistant intent.
- **1.2.1 offline `predict()` (Python, `assistant` domain) does give them:**

  | Clip | Head | Utterance tag |
  |---|---|---|
  | "Pause." (0.46 s) | 0.67 | 0.63 |
  | "Stop." (0.53 s) | 0.95 | 0.16 |
  | "Open WhatsApp." (0.92 s) | 0.11 | 0.30 |

  No assistant intent passed its threshold on any of these. So the gap is in the live path, where the utterance record waits for
  the segmenter (#1). The two numbers also disagree ("Stop." 0.95 against 0.16; #4).
- **Ask:**
  - Emit the utterance readouts on `flush()` / end of turn for any clip with speech, however short, together with the final (#1).
  - Calibrate intents for one- and two-word commands.
