# Evaluations

- [`written_form/`](written_form/): the written-form transcript (`transcript_written`) on number-heavy sentences: digits
  found, WER, whole-sentence accuracy.
- [`v121/`](v121/): the 1.2.1 evaluation: the TypeWhisper benchmark reproduced with its own scoring, Earnings-22 numbers,
  LibriSpeech regression, CPU vs MPS, streaming and CLI consistency, date records.

`common.py` holds the shared helpers (rendering with macOS voices, a simulated room, scoring).
