# Voice commands

**What it measures.** How often a spoken command to a reading assistant ends up as the right command, through
Arc Mini's transcript and a command parser, in four rooms. It also reports the word error rate and how long Arc takes per command.
This is the bench behind Steer for Mac's press-and-hold talk key (`SteerVoiceBench` in the Steer apps repo).

- **Phrases** (`data/`): 209 phrases in 21 command kinds, plus `UNKNOWN` (general questions that must not trigger a command).
  - `commands_spoken.tsv`: the phrases people say most, and the commands' own names.
  - `commands_loose.tsv`: how people actually ask.
  - `commands_loose_heldout.tsv` and `_heldout2.tsv`: new phrasings, written after the parser was tuned.
  - `commands_loose_heldout3.tsv`: 33 more, not used by either run here; kept for a later honest score.
- **Rooms:** Steer's own room model.
  - Speech is peak-normalised to 0.35.
  - It has 0.6 s of room before it and a 0.3 s tail after it (press-and-hold).
  - The noise is steady and slightly low-passed: quiet −60 dBFS, noisy −38 dBFS (a fan or a café), loud room −28 dBFS.
  - The soft-voice room uses −55 dBFS noise with the speech at 0.05 peak.
- **Arc settings, as the app runs them:** no prompt; the six messaging-app names as decoder bias. Steer also tested a names
  prompt with those names, and biasing towards the command words. Neither helped commands, so both stay off.
- **Scoring.** A clip counts as right when `parse(transcript) == expected`. It is only scored on phrases where `parse(true words) == expected`, so a parser miss is never charged to Arc.
  - Steer's parser (`VoiceCommand.swift`) is not public. `parser.py` is a small regex stand-in for it.
  - The stand-in reads 209/209 of the scored phrases right from the true words.

## Run

```bash
python evals/commands/run.py                 # 209 phrases x 4 voices x 4 rooms = 3,344 clips, about 15 min on 4 CPU cores
python evals/commands/run.py --quick         # every third phrase, 2 voices
python evals/commands/run.py --no-bias       # without the app-name bias
```

The script renders the speech with macOS `say` (Samantha, Daniel, Flo, Aman) into `evals/_work/` (git-ignored). No audio is
committed. On Linux, render the same file names with any TTS and pass `--clips DIR`.

## Results

| Run | Arc | Runtime | Voices | Clips | Right command: quiet / noisy / loud room / soft voice / all | WER all | Arc time per command |
|---|---|---|---|---|---|---|---|
| **Steer (app path)**, 2026-10-28 | Arc Mini 1.1.1 (HF rev `v1.1.1`) | libarc, the Steer app's C port; M3 Pro, CPU | Steer TTS edge female + male, 2 speeds | 7,648 (press-and-hold and tap; 5 rooms incl. "hesitant") | 96.1% / 90.9% / 83.0% / 96.1% / **92.0%** (7,032/7,648) | 3.9% quiet, 12.2% loud | p50 39–64 ms, p95 61–84 ms |
| Steer, whisper tiny.en (q5_1) for comparison, same clips | – | whisper.cpp | same | 7,648 | 99.0% / 98.3% / 96.7% / 97.7% / **97.7%** | 2.2–3.8% | p50 ~290 ms |
| **Python reference (this PR)**, see below | Arc Mini 1.1.1 (HF rev `v1.2.0`, same Mini files) | `arc_audio` 1.2.0, CPU, 4 threads, M3 Pro | macOS `say`: Samantha, Daniel, Flo, Aman | 3,344 (press-and-hold) | 84.9% / 75.1% / 61.5% / 78.2% / **74.9%** (2,506/3,344) | 17.5% quiet, 43.6% loud, 28.2% all | p50 137–159 ms, p95 300–345 ms (Python, CPU) |

Notes on the Steer row:
- From `SteerVoiceBench` output of 2026-10-28 (`full_arc.txt` / `full_whisper.txt`).
- Its phrase list is the 209 here.
- It adds a tap mode, which uses Steer's own energy endpointer, and a "hesitant" condition: the phrase split in two with a 0.85 s pause.
  - Hold/hesitant: 95.8%. Tap/hesitant: 95.6%, with 9 clips cut short by the endpointer, not by Arc.
- Release-candidate pass on two macOS voices (Samantha, Daniel; 16 commands each): 26/32 right.
  - The misses were "Course" (pause), "Keep breathing" (keep reading), "How long is Meredith" (how long is left), "music of" and "New Telegram" ×2.
  - Arc took 0.05–0.12 s per command (worst 0.37 s), in the app.

Rerun per voice (all rooms): Samantha 86.5%, Aman 82.1%, Daniel 75.8%, Flo 55.4% (Flo in the loud room: 33%). The best voice
in the quiet room (Samantha, 92%) is close to Steer's quiet number. The rerun is lower mainly because of the voices: Flo, a
light and fast macOS voice, loses most. Most frequent misses: "Mute Telegram" → "New Telegram" (all 32), "Am I free at 3?" →
"And my three of three", "Louder" → "Matter" / "Lada", "Quiet please" → "Clyde, please", "Skip" → "Skin" / empty.

The voices differ between the two rows (Steer's own TTS vs macOS voices), so compare the pattern (quiet vs loud, which words
fail) more than the absolute numbers.

## What fails

- **The errors are acoustic.** Short one-word commands lose their only word:
  - "Pause" → "Oz" / "cause" / "Of course";
  - "Resume" → "was you";
  - "Speak up" → "Sneak off";
  - "Mute Telegram" → "Newt Telegram" / "New Telegram".
- **Loud rooms are where Arc is weakest.** At −28 dBFS, Steer lost about 13 points against the quiet room. whisper tiny lost about 2.
- **Biasing towards command words didn't help** (Steer's `--bias-commands` run). Only the wake name and the app names are biased in the app.

## Known limits

- Synthetic voices only, close-miked, with synthetic room noise. A real-voice session is still to be done.
- The parser is a stand-in, so the "right command" numbers are only comparable between runs of this script.
- Tap mode and the hesitant condition are not reproduced here: they test Steer's endpointer, not Arc.
