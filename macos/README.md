# Arc with Apple's on-device speech recogniser (macOS 26)

[`arc_speech.swift`](arc_speech.swift) pairs Apple's `SpeechAnalyzer` / `SpeechTranscriber` (on-device, offline once
the language's assets are installed) with an Arc stream in `ARC_STREAM_EXTERNAL` mode: Arc runs heads-only through
the C API (no transcript decoder loaded), Apple's words go in with their audio times (`arc_stream_words`), and Arc's
utterances come out with the words, the read-out (device-directed, intent, act) and events. When Arc's segmenter
closes an utterance (`ARC_OUT_ENDPOINT`), the tool asks the transcriber to finalise up to there
(`SpeechAnalyzer.finalize(through:)`): without that, the transcriber can hold a long phrase's final text until the end
of input.

```bash
macos/build.sh                                         # libarc + build/arc-speech
macos/build/arc-speech arc-v1.1/mini clip.wav [...]    # fed at real time from the file; nothing is played
```

`--record DIR` writes the transcriber's results as replay files for `c/build/arc-stream --mode external --words-dir DIR`,
so the same words can be merged with Arc on another machine. No microphone or speech-recognition permission is needed
for files.

Each final result also goes in as `arc_stream_covered(s, end)`: the recogniser has finalised everything up to there, so
Arc's utterances that end before it get their FINAL at once instead of waiting for more words.

Forcing a final at every ENDPOINT is a trade-off. Arc's segmenter also closes on pauses inside a sentence, and a final
forced there can cost words in the transcriber's next result (on a LibriSpeech clip, "the words of Shelley's fragment"
came back as ". of Shelley's fragment"). `--no-finalize` leaves finals to the transcriber: the text is better and Arc's
utterances wait for it.

## Measured on a Mac

M3 Pro, macOS 26.2, LibriSpeech test-clean clips (and the same clips with pink noise at 10 dB SNR), Whisper's English
normaliser. Apple's text is from `arc-speech --no-finalize` (the transcriber finalises on its own).

| | WER, 100 clean clips | WER, 100 noisy clips | time to first word / final / settled tags | CPU |
|---|---|---|---|---|
| SpeechTranscriber (+ Arc heads for tags) | 1.2 % | 2.2 % | to be measured | to be measured |
| Arc full (`c/build/arc-stream`, same output as on the Pi) | 6.2 % | to be measured | see [the Pi table](../raspberry-pi/README.md#measured-on-a-raspberry-pi-5) | to be measured |

## Android

The same C entry points serve Android's `SpeechRecognizer` (`createOnDeviceSpeechRecognizer`, API 31+): an engine
created with `ARC_NO_DECODER`, a stream in `ARC_STREAM_EXTERNAL` mode fed the microphone's PCM, and the recogniser's
`onPartialResults` / `onResults` strings passed to `arc_stream_text(s, text, final)` (no word times: the words are
spread over the utterance open at that moment). Call `stopListening()` on `ARC_OUT_ENDPOINT` to get the final result
sooner.
