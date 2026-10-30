# Changelog

Package versions of `arc-transcriber` (this repository) and the model revisions of
[WhissleAI/arc-v1.1](https://huggingface.co/WhissleAI/arc-v1.1) they load by default. Model numbers are on the model card.

## 1.2.1

Code-only release: model revision stays `v1.2.0` (`arc_audio.MODEL_REVISION`; the package no longer derives the revision from its
own version). Same weights, same golden outputs.

- Written form (`arc_audio/written.py`): `transcript_written` in `predict()` (per utterance `text` + `words`), `transcribe(written=True)`
  (`text_written`, `words_written`), stream results (`transcript_written` for new utterances; `flush()` adds `transcript_written_all`),
  `arc-transcribe --written` and a `written` field per JSON line. Built from Arc's own number tags and values: times ("2:30 p.m."),
  money ("$45", "€12.50", "$2 million"), dates ("March 3", "March 3, 2024", "the 3rd of March"), percents, decimals, cardinals,
  ordinals from 10, phone numbers and IDs as digits, scores ("3–1") in sports domains. A span whose value is not certain keeps its
  words; single-digit counts / durations / ordinals stay words (`spell_small=False` to change). Linked names take the list's spelling
  word for word (`spell_names`; unlike `transcript_rewritten`, no unspoken words are added). `tests/test_written.py`,
  `evals/written_form/` (digits found: 22 % -> 74 % on 120 number-heavy clips; LibriSpeech WER unchanged).
- Date records keep the month: "March third" gives `heard` "March third", `date` {year, month, day} and `iso` "--03-03" (was `heard`
  "third", the month lost); "the third of March" is covered too. `value` is unchanged (the day, or the year).
- Apple GPU: `device="mps"` runs Arc Mini (float64 pooling sums kept on the CPU; event-head tensors on the model's device) with the
  same outputs as CPU (`tests/golden/mini.json` passes on MPS), and Arc Transcriber's `ServerFeatures` no longer passes CPU tensors to
  the GPU heads ("Passed CPU tensor to MPS op"; the CUDA path makes the same mixed-device call, untested here). `device="auto"` picks cuda > mps > cpu;
  `device="mps"` without MPS falls back to cpu with a warning. CLI: `--device mps|auto`.
- `pytest` runs against the checkout (`pythonpath = ["."]`), not an installed copy.
- `evals/FINDINGS.md`: new items 18-25 (written form and benchmarks, the date month, MPS and speed, Mini's error mix, short-clip
  fillers, the context prompt budget, live context on short questions, short-clip readouts) and new evidence for items 1, 6, 8, 10
  and 13. `evals/written_form/` (the written-form benchmark).

- `c/`: `arc_score` (teacher-forced log P(text | audio) for candidate texts, name spans scored inside Arc's name tags, encoder
  reused, shared prefixes decoded once) and the final decode's n-best (`arc_utt.alts`, `alt_scores`), with `build/arc-score`
  and `tests/test_score.c` against arc_audio (`tools/dump_score_ref.py`, `tests/data/score_ref.jsonl`).

- `c/`: libarc, a native C API (C11, no dependencies) for Arc Mini: transcript, word times, linked names, number values,
  utterance tags, events; passes `tests/golden/mini.json` (`make -C c parity`). CI builds it and runs its unit tests on Linux
  and macOS (`.github/workflows/c.yml`). Python package unchanged.

## 1.2.0

- New model revision `v1.2.0` (the default): **Arc Transcriber 1.2.0** (`--backbone full`); **Arc Mini stays 1.1.1** (same files).
- Arc Transcriber 1.2.0: word error rate 15.7 % -> 13.7 % on our held-out multi-domain dev set; more names (span recall 76.4 % ->
  80.2 %) and numbers (85.0 % -> 86.8 %) found; assistant intents 48.3 % -> 53.2 %; diarization error rate 40.8 % -> 38.8 %; fewer
  repetition loops. Short assistant commands are worse (word error rate 7.0 % -> 10.9 %).
- Arc Transcriber 1.2.0 thresholds: device-directed default 0.85 (was 0.50; false-positive rate 2.6 % -> 2.3 % at the same recall);
  audio-event thresholds re-calibrated, safety labels set for a low false-alarm rate (still experimental, off by default). Tag
  thresholds, tag offsets, labels and default-on events are unchanged.
- Same file names, tensor names and packed 4-bit format as 1.1.1 (`docs/WEIGHTS_FORMAT.md`): ports load 1.2.0 without code changes.
- `tests/golden/full.json` regenerated for Arc Transcriber 1.2.0; `tests/golden/mini.json` unchanged.
- Code behaviour, API and output schema are unchanged. `LICENSE` / `NOTICE` headers now say "Arc" (same Whissle terms).

## 1.1.2

- Licence and metadata only: Whissle Inc. named as rights holder; Whissle terms added to LICENSE (commercial licences,
  reserved rights, outputs, patents and trademarks, contributions); CONTRIBUTING.md and a pull request template. Code
  behaviour and model weights are unchanged (model revision `v1.1.2` has the same weight files as `v1.1.1`).

## 1.1.1

- Model names: **Arc Mini** (`--backbone mini`, was `edge`) and **Arc Transcriber** (`--backbone full`, was `parakeet`); the old names
  still work. Model folders on Hugging Face are now `mini/` and `full/` (revision `v1.1.1`; `v1.1.0` is unchanged).
- External ASR words (`--asr deepgram|verbit`) are grouped into utterances like the built-in transcript; name spans no longer run into
  the next word; `heard` texts have no trailing punctuation; repeated records of the same mention are merged; spelled-out scores
  ("one hundred and two to ninety eight") keep both values.
- A clear error when live mode needs `pip install "arc-transcriber[live]"`.
- `docs/DEPLOYMENT.md` (porting to MLX / Core ML / ONNX Runtime / ExecuTorch / TensorRT) and golden reference outputs for checking
  an export (`tests/golden/`, `scripts/check_parity.py`).

## 1.1.0

- Two backbones: `edge` (new on-device model, 4-bit packed weights, about 71 MB download) and `parakeet` (server model on the NVIDIA
  Parakeet-TDT 0.6B v2 encoder).
- 34 inline tag classes in the transcript (new: law, product, term, phone, email, URL, address), repetition-loop guard, per-class
  tag calibration.
- Decoder context (names, terms, topic, earlier text) and context memory (`auto_prev`).
- Name-list snap post-pass (fuzzy / phonetic), sound / channel events with a status per label, device-directed speech.
- The default model is `WhissleAI/arc-v1.1`. The API is backward compatible: 1.0 calls keep working (they use the edge backbone).

## 1.0.1

- Arc v1 inference code and the `arc-transcribe` CLI.
