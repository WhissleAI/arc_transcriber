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

## Android

The same C entry points serve Android's `SpeechRecognizer` (`createOnDeviceSpeechRecognizer`, API 31+): an engine
created with `ARC_NO_DECODER`, a stream in `ARC_STREAM_EXTERNAL` mode fed the microphone's PCM, and the recogniser's
`onPartialResults` / `onResults` strings passed to `arc_stream_text(s, text, final)` (no word times: the words are
spread over the utterance open at that moment). Call `stopListening()` on `ARC_OUT_ENDPOINT` to get the final result
sooner.
