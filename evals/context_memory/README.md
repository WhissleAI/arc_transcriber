# Context memory

**What it measures.** Two kinds of "memory" in Arc Mini, each tested two ways:

1. **Text context.** This is the decoder's trained prompt: domain, names and terms with kinds, and previous text, plus the decoder bias.
   - Does it lower the word error rate?
   - Does it get people's and product names right?
   - Does it help a sentence that continues the last one?
2. **Hidden state.** The trunk's SSM state can be carried from one utterance to the next instead of starting from zero.
   - Does that help device-directed (commands vs speech to someone else)?
   - Does it help the dialogue act on commands?

**Test sentences** (`items.py`; all names are fictional):
- 20 commands to a reading assistant;
- 10 dictation sentences with people's and product names;
- 5 sentences about a book (character names);
- 6 continuation pairs (the second sentence is scored, with the first as previous text);
- 12 sentences said to someone else in the room (the negatives for device-directed).

**Text-context conditions:**
- **none:** no prompt.
- **bias:** decoder bias towards the names relevant to the sentence's group (apps for commands, people and terms for
  dictation, characters for the book).
- **names prompt + bias:** the same names also go into the prompt, with their kinds.
- **all names (31) prompt + bias:** every name, every time.
- **previous text only.**
- **all names + bias + previous text.**

**Hidden state** (`state` part):
- Per voice, one session of the commands, dictation and other-speech sentences in a shuffled order, each after 2 s of quiet room.
- **fresh:** `predict()` on each utterance alone, with its 2 s of room before it.
- **carried:** one `predict()` over the whole session (the state runs on, as in a stream). Each utterance is matched back by time.
- The score is the AUC of device-directed for commands vs speech to someone else.
  - Both of Arc's device-directed numbers are reported: the utterance tag (`utterances[].tags['discourse.assistant.device_directed']`) and the transcript's own head (`transcript[].device_directed.p`).
  - Steer's libarc reads one number, `device_directed`, per utterance. Its AUCs match the utterance tag's here.
- Act is counted right when a command comes out as `command_request`, or as `question` for questions.

## Run

```bash
python evals/context_memory/run.py               # both parts, 4 voices, about 15 min on 4 CPU cores
python evals/context_memory/run.py --part text   # or --part state
```

## Results

### Text context

| Condition | All WER | Names right | Commands WER | Continuations WER | ms per utterance |
|---|---|---|---|---|---|
| **Steer (libarc)**, none (the app before) | 30.0% | 42.9% | 14.0% | 37.9% | 271 |
| bias only | 22.2% | 61.0% | 11.7% | 27.4% | 279 |
| names prompt + bias | 20.8% | 68.4% | 11.0% | 32.1% | 488 |
| all names (31) + bias | 21.5% | 67.1% | 12.7% | 32.1% | 502 |
| previous text only | 29.3% | 43.5% | 14.0% | 32.6% | 285 |
| all names + previous text | **20.2%** | **68.7%** | 12.7% | **23.2%** | 505 |
| **Python reference (this PR)**, none | 44.4% | 28.6% | 35.0% | 51.3% | 204 |
| bias (relevant names) | 35.8% | 50.0% | 31.7% | 44.1% | 305 |
| names prompt + bias (relevant) | 33.0% | 57.7% | 30.4% | 50.7% | 486 |
| all names (31) prompt + bias | 34.3% | 55.6% | 32.9% | 50.7% | 318 |
| previous text only | 44.0% | 29.0% | 35.0% | 48.0% | 186 |
| all names + bias + previous text | 33.0% | 56.9% | 32.9% | 41.4% | 255 |

How the Steer rows were made:
- Run on 2026-10-28 with libarc, Arc Mini 1.1.1 (HF rev `v1.1.1`), on an M3 Pro CPU.
- 295 clips: Steer TTS voices (female, male, voice_0081) and two macOS voices (Samantha, Daniel).
- The libarc rows' ms include each clip's whole pipeline, so compare ms within a row group only.

How the rerun rows were made:
- Run on 2026-10-29 with `arc_audio` 1.2.0 and model rev `v1.2.0` (Arc Mini 1.1.1 files), on the CPU with 4 threads.
- 236 clips: macOS Samantha, Daniel, Flo and Aman.
- The absolute WER is higher than Steer's because of the voices (Flo especially).
- The direction is the same: names in the bias and prompt cut WER by about 11 points and double the names that come out right, and previous text helps continuations most.

From Steer's libarc runs, previous-text length vs WER:

| Previous text | Commands WER | Continuations WER | Decode ms (commands / continuations) |
|---|---|---|---|
| none | 11.7% | 39.5% | 58 / 123 |
| 8 words | 9.2% | 34.2% | 151 / 229 |
| 16 words | 9.2% | 36.8% | 162 / 242 |
| 24 words | 9.2% | 35.5% | 209 / 400 |
| 40 words | 9.2% | 38.2% | 346 / 327 |

- Unrelated previous text (the last command) does not hurt commands: 14.0% → 13.0% WER, +7 ms.
- Steer ships text memory on: its bias list, plus the last 12 words said within 60 s as previous text.

### Hidden state

| Run | Trunk state | Device-directed AUC, commands vs other speech | Act right on commands |
|---|---|---|---|
| **Steer (libarc)**, 210 utterances, 2 s room gaps | fresh | **0.418** | **97/100** |
| | carried | 0.338 | 93/100 |
| Steer (libarc), no room gap | fresh / carried | 0.373 / 0.343 | 92 / 90 |
| Steer (libarc), one long session | carried | 0.325 (other speech drifts up 0.71 → 0.83) | 91 |
| **Python reference (this PR)**, 4 macOS voices, 168 utterances, utterance tag | fresh / carried | **0.440** / 0.347 | **78/80** / 75/80 |
| **Python reference (this PR)**, transcript head | fresh / carried | 0.529 / 0.426 (Samantha alone: 0.842 fresh in a trial run) | |

- Carrying the state made both device-directed and act worse on this data, so Steer ships it **off**.
  - libarc has `arc_memory()`, save and load behind an experimental switch.
- An AUC under 0.5 means the utterance tag ranks speech to someone else *above* commands on these synthetic voices.

## Known limits

- Synthetic voices.
- One person per session. The trunk was trained on continuous streams, and Steer's sessions are separate press-and-hold clips joined with room gaps.
- The relevant-names lists are per group, not per sentence.
