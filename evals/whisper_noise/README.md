# Whispers, noise and channels

**What it measures.** Short commands said in hard conditions: a whisper, low-level speech, noisy and loud rooms, band
noise, a telephone line, clipping, "mumbled" (low-passed and quiet), and a quiet ward at night. For each condition:
- the word error rate;
- how many transcripts come back **empty**;
- the mean probability of Arc's channel and sound events over the speech (telephone, narrowband, clipping, degraded audio, SNR low/high, silence);
- the top intent on the clinical lines.

- **Lines** (`data/lines.tsv`): Steer's ten signal-study commands, the bedside film's lines ("Raise my head a little.",
  "Head up a little.", "A bit more.", "I need the nurse.") and "Show the pre-op scan, left side."
- **Voices:**
  - macOS `say` Samantha, Daniel and Aman.
  - The whisper conditions use the macOS **Whisper** voice: a real unvoiced whisper, unlike Steer TTS's `whisper` tag, which is only 5–10 dB quieter than normal speech.
- **Levels:** the speaker's usual level is −26 dBFS (voiced level). Low level is −42 dBFS, where the bedside film placed its whispers: 16 dB under.
- **Channels:** these follow Steer's signal study (`arcsig`), rewritten in numpy.
  - Band noise: 200–4000 Hz at 5 dB SNR.
  - Telephone: 300–3400 Hz, through 8 kHz, 8-bit µ-law.
  - Clipped: ×8, clipped at 0.3.
  - Mumbled: 900 Hz low-pass, ×0.15.

## Run

```bash
python evals/whisper_noise/run.py        # 12 conditions x 3 voices x 15 lines, about 5 min
```

## Results

**Steer**: the signal study used the Python reference 1.1.1 with Steer TTS voices, 10 commands per condition, on 2026-10-28:

| Condition | Transcript | Events |
|---|---|---|
| clean | right except "Save that again." | snr.high; silence ≥ 0.98 in gaps |
| band noise, 5 dB SNR | badly degraded ("Oh, it's a good fast pass") | snr.low; scene labels misfire as crowd, cheer, laughter, whistle |
| telephone | mostly right; "twenty minutes" still read as 1200 s | telephone **0.57**, narrowband **0.48** mean; ≤ 0.08 / 0.26 elsewhere |
| clipped | right | clipping **0.51** mean, ≤ 0.05 elsewhere |
| mumbled | mostly wrong | – |

Bedside concept film (libarc, Steer TTS `whisper` tag placed at −42 dBFS in a −58 dBFS ward):
- "Raise my head a little." came back as "Raised my head a little.", "raise my head a little." and "**Wraith Mei had** a little." on three seeds.
- "Head up a little" came back as "Hit up" / "Get up" / "**Bedup** a little" across 4 voices.
- Short unprimed commands ("zoom in") sometimes came back empty.
- Top intents were noise: "Raise my head" → `audio_volume_up` 0.20, "I need the nurse" → **`iot_coffee`** 0.15, "show the pre-op scan" → `takeaway_order` 0.08.

How Steer handles whispers:
- Steer's whisper rule is the app's, not Arc's: voiced level ≥ 6 dB under the person's usual level counts as a whisper.
- At ≥ 12 dB under, the reply uses Steer's `whisper` tag. In a noisy room, Steer speaks 4 dB louder.
- Arc has no whisper or vocal-effort output today. The level comes from the samples.

**Python reference (this PR):** `arc_audio` 1.2.0, model rev `v1.2.0` (Arc Mini 1.1.1 files), CPU, 2026-10-29:

| Condition | WER | Empty | telephone | narrowband | clipping | degraded | snr.low | snr.high |
|---|---|---|---|---|---|---|---|---|
| clean | 20.8% | 0/45 | 0.00 | 0.00 | 0.01 | 0.00 | 0.01 | 0.13 |
| low level (-42 dBFS) | 22.6% | 0/45 | 0.00 | 0.00 | 0.01 | 0.00 | 0.02 | 0.05 |
| whisper voice | 58.9% | 0/15 | 0.00 | 0.02 | 0.01 | 0.02 | 0.04 | 0.02 |
| whisper voice, low level | 58.9% | 0/15 | 0.00 | 0.01 | 0.01 | 0.02 | 0.06 | 0.01 |
| noisy room (-38 dBFS noise) | 41.1% | 0/45 | 0.00 | 0.00 | 0.02 | 0.01 | 0.05 | 0.01 |
| loud room (-28 dBFS noise) | 64.9% | 0/45 | 0.00 | 0.00 | 0.05 | 0.09 | 0.19 | 0.01 |
| band noise, 5 dB SNR | 81.0% | 0/45 | 0.00 | 0.01 | 0.03 | 0.12 | 0.14 | 0.00 |
| telephone (300-3400 Hz, mu-law) | 40.5% | 0/45 | 0.04 | 0.03 | 0.01 | 0.01 | 0.03 | 0.03 |
| clipped | 40.5% | 0/45 | 0.00 | 0.01 | 0.66 | 0.49 | 0.02 | 0.22 |
| mumbled (low-passed, quiet) | 106.5% | 0/45 | 0.00 | 0.01 | 0.01 | 0.01 | 0.03 | 0.05 |
| ward at night (-58 dBFS) | 23.2% | 0/45 | 0.00 | 0.00 | 0.01 | 0.00 | 0.01 | 0.07 |
| whisper voice, ward, low level | 66.1% | 0/15 | 0.00 | 0.01 | 0.01 | 0.03 | 0.08 | 0.01 |

What the rerun shows:
- **Whispers:** the macOS Whisper voice gives 58.9% WER against 20.8% for the same lines in normal voices.
  - Turning it down to −42 dBFS changes nothing (58.9%). In the ward it gives 66.1%.
  - Normal speech at −42 dBFS is barely worse than at −26 dBFS (22.6% vs 20.8%), so the problem is the unvoiced whisper, not the level.
  - No transcript came back empty, but whispered lines come back wrong: "Pause." → "Oh, shit.", "Read faster." → "Brad's master.", "Stop reading Slack for an hour." → "Stop redesigning for an hour.", "Summarise this." → "Summit Rice's."
- **Rooms:** WER goes 20.8% clean → 41.1% noisy → 64.9% loud room → 81.0% at 5 dB band noise. Mumbled: 106.5%.
- **Events:**
  - **clipping fires on clipped audio** (0.66 mean; ≤ 0.05 elsewhere), together with degraded audio (0.49).
  - **telephone and narrowband did not fire** on these short phone-channel commands (0.04 / 0.03). The signal study saw 0.57 / 0.48 on one phone clip of ten commands in a row (about half a minute), so on short clips the channel labels need more context than one command gives.
  - snr.low rises with the noise (0.01 clean → 0.19 loud room → 0.14 band noise).
- **Intents on the clinical lines** are smart-home classes with low confidence:
  - "Raise my head a little" → `calendar_set` / `lists_query` / `social_post`;
  - "I need the nurse" → `lists_createoradd` / `calendar_set`;
  - "Show the pre-op scan" → `qa_stock` / `play_music` / `calendar_set`;
  - all at p ≤ 0.36.

## Known limits

- Synthetic speech and channels.
- The macOS Whisper voice is one speaker.
- The event means are over the speech frames only.
- Several scene labels are experimental (`events="all"` includes them); only telephone, narrowband, clipping, degraded audio and silence are stable or beta.
