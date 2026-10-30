# Device-directed speech and wake names

**What it measures.** Can Arc tell speech meant for the device from people talking to each other in the same room? And
which device names does it hear reliably at the start of a command? This came from two places:
- Steer's ambient mode, where a command needs the name "Steer" and directed ≥ 0.5.
- Two healthcare concept films (a bedside bed and an operating-room display) made to pitch Arc to device makers.

- **A. Wake names** (`data/wake_names.tsv`):
  - Each candidate name is said before three commands (`data/or_commands.tsv`: "zoom in", "more", "show the pre-op scan, left side"; Steer's own: "pause", "keep reading", "sleep in twenty minutes") by three voices.
  - The clips are placed in an operating room: pink air-flow noise at −40 dBFS plus a pulse-oximeter beep, with speech at −24 dBFS. Steer's quiet room is used for the Steer commands.
  - The name counts as heard when it is one of the transcript's first two words.
  - Each name is run **unprimed** and **primed**: the name (and "pre-op scan") in the decoder's names prompt and bias.
- **B. Cross-talk:**
  - The commands said with the name "Nova", the same commands without it, and six lines that staff say to each other (`data/or_crosstalk.tsv`), all in the operating room.
  - Both device-directed numbers are reported: the utterance tag and the transcript's head.
  - The AUC is commands vs cross-talk. 0.5 is chance; below 0.5, cross-talk scores *higher*.

## Run

```bash
python evals/device_directed/run.py            # 3 voices (Samantha, Daniel, Aman), about 2 min
```

## Results

**Steer (libarc)**: all runs used Arc Mini 1.1.1 (HF rev `v1.1.1`) on an M3 Pro.

| Test | Result |
|---|---|
| Wake word "Steer", two macOS voices, 6 commands each | heard in 9/12; the Daniel voice lost it as "Sleep in…", "Yeah. Save…", "Yeah. Still…". Directed with the name 0.26–0.89, ≥ 0.5 in 8/9 |
| Directed without the name, 32 command clips | 0.29–0.97 ("Say that again" 0.29; "sleep in twenty minutes" 0.30–0.31) |
| Signal study (Python reference 1.1.1, Steer voices): commands vs narration / dialogue / narration over music | commands 0.26–0.98 (clean mean 0.76); narration 0.67, 0.84; two-person dialogue 0.75–0.88; narration over music 0.88 |
| Operating-room film, 3 Steer voices | surgeon's commands **0.51–0.88**, the nurse's lines to the surgeon **0.74–0.93** |
| Wake-name screen for the film (3 voices × 3 commands, primed with the name) | "Scope" missed **8/9** (heard as "Let's go" / "Go"); "Nova" and "Camera" 9/9; "Atlas" and "Display" 7/9 |
| Pipecat voice-agent session (Python reference 1.1.1) | directed 0.0005–0.26 for speech addressed to the agent (the service's per-turn `device_directed`), so the integration's `directed_threshold` is off by default |

**Python reference (this PR):** `arc_audio` 1.2.0, model rev `v1.2.0` (Arc Mini 1.1.1 files), CPU, 4 threads, 2026-10-29.

| Wake name (3 voices × 3 commands) | Unprimed | Primed | Unprimed misses, e.g. |
|---|---|---|---|
| Scope | 1/9 | 7/9 | Let's go. Zunin.; Let's go more. |
| Nova | 3/9 | 9/9 | You know about Moore.; Novak showed the free off-scan left side. |
| Camera | 0/9 | 6/9 | Tamara, Zunin.; Tamara, more. |
| Atlas | 5/9 | 8/9 | Adler, more.; Edward, show the free off-scan left side. |
| Display | 3/9 | 4/9 | Just play. Zuman.; Just play more. |
| Steer (Steer quiet room) | 3/9 | 5/9 | Yeah. Cause.; Yeah. He's reading. |

| Group (operating room) | Utterance tag: range, mean, ≥ 0.5 | AUC vs cross-talk (tag / head) |
|---|---|---|
| command with name (9) | 0.58–0.84, mean 0.73, 9/9 | 0.33 / 0.34 |
| command, no name (9) | 0.35–0.91, mean 0.71, 8/9 | 0.35 / 0.34 |
| cross-talk (staff to staff) (18) | 0.50–0.91, mean 0.78, 18/18 | – |

The transcript head was 0.000–0.014 for every clip in this scene.

## What it means

- **Device-directed doesn't separate cross-talk from commands today.** Staff talking to each other score as high as or higher than
  commands, so "ignore talk between people" can't rest on directed alone. Steer and the films use a device name **and**
  directed ≥ 0.5.
- **Wake names need screening per room and voice.** Some names are almost never heard unprimed ("Scope" → "Let's go",
  "Camera" → "Tamara", "Display" → "Just play", "Steer" → "Steel" / "Yeah"). Priming helps some names a lot ("Nova", "Scope")
  and others little ("Display").
- **The two device-directed numbers disagree.** The utterance tag sits around 0.5–0.9 for everything. The transcript head is near 0 for everything here, yet ranks commands well in one case (see `context_memory`).

## Known limits

- Synthetic voices (macOS `say` here, Steer TTS in the films), synthetic rooms, and a small set: 9 trials per name, 18 cross-talk lines.
- The film's priming used libarc's names prompt. Here it is `arc_audio`'s names prompt plus bias, which may explain why "Scope"
  does better primed here (see the table).
