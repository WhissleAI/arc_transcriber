# Arc with the OS speech recogniser (macOS, Android)

[`arc_speech.swift`](arc_speech.swift) pairs Apple's on-device `SpeechAnalyzer` / `SpeechTranscriber` (macOS 26) with
an Arc stream in `ARC_STREAM_EXTERNAL` mode. Arc runs heads-only through the C API; Apple's words go in with their
audio times; Arc's utterances come out with the words, the read-out (device-directed, intent, act) and events.

```bash
macos/build.sh                                         # libarc + build/arc-speech
macos/build/arc-speech arc-v1.1/mini clip.wav [...]    # fed at real time from the file; nothing is played
```

- When Arc's segmenter closes an utterance (`ARC_OUT_ENDPOINT`), the tool asks the transcriber to finalise up to there.
  This can cost words when Arc closes on a pause inside a sentence; `--no-finalize` leaves finals to the transcriber
  (better text, later utterances).
- `--record DIR` writes the transcriber's results as replay files for
  `c/build/arc-stream --mode external --words-dir DIR`.
- No microphone or speech-recognition permission is needed for files.

## Measured on a Mac

M3 Pro, macOS 26.2, 100 LibriSpeech test-clean clips (noisy: pink noise at 10 dB SNR), Whisper's English normaliser.

| transcript | WER clean | WER noisy |
|---|---|---|
| Apple SpeechTranscriber + Arc heads (`--no-finalize`) | 1.2 % | 2.2 % |
| Arc full (`c/build/arc-stream`) | 6.2 % | not measured |

Latency and CPU: not measured yet. Raspberry Pi latencies:
[raspberry-pi/](../raspberry-pi/README.md#measured-on-a-raspberry-pi-5).

## Android

The same C calls serve Android's on-device `SpeechRecognizer` (API 31+): an engine created with `ARC_NO_DECODER`, a
stream in `ARC_STREAM_EXTERNAL` mode fed the microphone PCM, and the recogniser's `onPartialResults` / `onResults`
strings passed to `arc_stream_text(s, text, final)`. Call `stopListening()` on `ARC_OUT_ENDPOINT` to get the final
sooner.
