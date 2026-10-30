# written_form: the written transcript against written references

**Question:** how much does `transcript_written` (numbers from Arc's own tags written as digits, names in the list's spelling) change
the scores against references written the usual way ("2:30 p.m.", "$45", "March 3"), and does it hurt plain speech?

```bash
python evals/written_form/run.py                       # 40 sentences x Samantha, Daniel, Flo; Arc Mini on the CPU
python evals/written_form/run.py --libri DIR           # also LibriSpeech-style clips: DIR/<id>.wav + DIR/refs.json {id: text}
```

- `data/sentences.txt`: 40 number-heavy sentences in written form (times, money, dates, percents, measures, ordinals, cardinals,
  scores, IDs). macOS `say` reads them aloud; the written text is the reference.
- Scores, for `transcript`, `transcript_written` and the written form with single digits as digits (`spell_small=False`):
  - **WER** after Whisper's `EnglishTextNormalizer` (from transformers; a basic lower-case fallback without it);
  - **digits**: the written numbers of the reference ("$45", "2:30", "12.5%", "3rd", "2,400") found verbatim in the raw output;
  - **exact**: the whole sentence right, ignoring case and sentence punctuation.
- LibriSpeech check: the 73 validation clips of `hf-internal-testing/librispeech_asr_dummy` (references in words).

## Results (2026-10-30, `arc_audio` 1.2.1, Arc Mini 1.1.1 files, CPU, M3 Pro)

| 120 number clips | WER | digits found (135) | sentence right |
|---|---|---|---|
| `transcript` (spoken form) | 27.2% | 22.2% | 18.3% |
| `transcript_written` | **24.2%** | **74.1%** | **52.5%** |
| written, single digits as digits | 24.2% | 76.3% | 54.2% |

| 73 LibriSpeech clips | WER |
|---|---|
| `transcript` | 15.48% |
| `transcript_written` | 15.48% |

By kind, digits found go from 10% to 71% for times, 11% to 72% for money, 12% to 58% for dates and 33% to 89% for percents
(`results/run.json`, `by_kind`).

## Limits

- The remaining misses are mostly recognition ("102 to nine two", "two point high kilograms") and some tagging ("two thousand four
  hundred employees" tagged as money). A few are style: Arc said "the 15th of October" where the reference is "October 15", and the
  written form keeps the order that was spoken.
- Single-digit counts and ordinals stay words by default ("the third floor"), which costs digits on references that write "3rd".
- Synthetic voices only; Flo (UK) is much harder for Arc Mini than Samantha and Daniel.
- Arc Transcriber (full) was not run here (its 2.5 GB encoder download was over this run's disk budget).
