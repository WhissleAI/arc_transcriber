# Arc Transcriber as a hosted service (`deploy/`)

`arc-serve` runs **Arc Mini** through **libarc** (`c/`, C11) behind a small HTTP + WebSocket server written in C.
No Python, no PyTorch, no ONNX Runtime, no network access at run time. One binary, one container image, the same
contract on every platform:

| route | what |
|---|---|
| `POST /v1/audio/transcriptions` | OpenAI-compatible request and response, plus Arc fields: words with times and inline tags, linked names, number values, utterance tags (intent, act, device-directed), sound / channel events |
| `WS /v1/realtime` | streaming: 16 kHz PCM16 in; partials, endpoints, finals (identical to the offline decode), settled tags, events out |
| `GET /health`, `GET /v1/health/live`, `GET /v1/health/ready` | liveness / readiness (NIM-style paths) |
| `GET /v1/models` | the loaded model, OpenAI list shape |
| `GET /ping`, `POST /invocations`, `GET /execution-parameters` | Amazon SageMaker's container contract (port 8080, `docker run IMAGE serve`) |
| `POST /v1/snowflake/transcribe` | Snowflake service-function rows (`{"data": [[row, audio], ...]}`) |
| `POST /v1/tags`, `POST /v1/snowflake/tags` | combined image only: Arc Tagger (see below) |

The full contract is [`openapi.yaml`](openapi.yaml). Platform adapters are in [`platforms/`](platforms/).

## Quick start

```bash
make -C deploy -j2                                  # deploy/build/arc-serve (builds c/build/libarc.a first)
ARC_MODEL_DIR=/path/to/arc-v1.1/mini deploy/build/arc-serve
curl -s localhost:8080/v1/audio/transcriptions -F file=@examples/example_synth.wav \
     -F names=@examples/example_names.json -F domain=nba -F response_format=verbose_json
```

OpenAI SDKs and clients work unchanged (`base_url="http://host:8080/v1"`, any `model`; the server answers with the
model it serves). Arc's extra fields come with `response_format=verbose_json`; Arc's extra inputs (`names`, `domain`,
...) are ordinary form fields:

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
r = c.audio.transcriptions.create(model="arc-mini", file=open("clip.wav", "rb"), response_format="verbose_json")

import requests   # with Arc's inputs
r = requests.post("http://localhost:8080/v1/audio/transcriptions", files={"file": open("clip.wav", "rb")},
                  data={"names": '["Theo Brandt", "Harbor City Hawks"]', "domain": "nba", "response_format": "verbose_json"}).json()
```

### Request options

Form fields, query parameters, JSON keys (for `application/json` bodies with base64 `audio`) or SageMaker custom
attributes (`X-Amzn-SageMaker-Custom-Attributes: domain=nba;response_format=text`):

| option | default | |
|---|---|---|
| `file` | | the audio: WAV (PCM 8/16/24/32-bit, float 32/64, any channels, any rate: resampled to 16 kHz mono) or raw PCM16 (`audio/pcm`, `sample_rate=`) |
| `response_format` | `json` | `json`, `text`, `verbose_json`, `srt`, `vtt`, `arc_json` (arc_audio's keys, used by the parity check) |
| `names` | | the list to link names to (and to bias the decoder toward): JSON array of strings or `{name, kind, group, aliases}`, or newline / comma separated; `names[]` repeated fields also work |
| `prompt` | | earlier text, for the decoder's context prompt (OpenAI's meaning of `prompt`) |
| `terms`, `topic` | | vocabulary and topic for the context prompt and the bias |
| `domain` | `unknown` (`ARC_DOMAIN`) | Arc's domain: `assistant`, `meeting`, `talk`, `news`, `nba`, `nfl`, `nhl`, `soccer`, `cricket`, `esports`, `stream_chat`, `court_trial`, ... |
| `language` | | only `en` (Arc Mini is English); anything else is refused with 400 |
| `events`, `words`, `trunk`, `link_names`, `numbers` | `true` | switch read-outs off to save time |
| `diarization` | `auto` | speaker turns: `auto` (by domain, as arc_audio: `meeting` / `news` / `talk` use `voice`, `nhl` / `cricket` / `esports` / `court_appellate` use `fused`, the rest `slots`), `slots`, `fused`, `voice` |
| `one_utterance` | `false` | decode the clip as one utterance (a held-key command) |
| `beams` | 4 | beam width (1 = greedy) |
| `model`, `temperature` | | accepted and ignored (OpenAI clients send them) |

Errors use OpenAI's envelope: `{"error": {"message", "type", "code"}}` with 400 / 404 / 405 / 413 / 415 / 503.

### Realtime (`WS /v1/realtime`)

```
ws://host:8080/v1/realtime?domain=assistant&partials=400&events=1
-> binary frames: 16 kHz mono PCM16 (any size), or {"type":"input_audio_buffer.append","audio":"<base64>"}
-> {"type":"session.update","session":{"names":[...],"prompt":"..."}}   {"type":"input_audio_buffer.commit"}   {"type":"input_audio_buffer.finish"}
<- session.created, transcript.partial, transcript.endpoint, transcript.final, transcript.tags, event, frame (frames=1), session.closed, error
```

Finals and tags records carry `speaker` (the diarization turn so far that the utterance overlaps most, the open turn
included); `session.closed` carries the stream's `speakers` and `diarization`, identical to a transcription of the same
audio. `diarization=` works as for files.

A session holds one engine for its lifetime. Names / numbers linking is not done in streams (libarc: linking needs
the word times of a whole utterance; use the file route for it).

### Configuration (environment)

| variable | default | |
|---|---|---|
| `PORT`, `HOST` | `8080`, `0.0.0.0` | SageMaker requires 8080; Hugging Face lets you choose |
| `ARC_MODEL_DIR` | first of `/opt/ml/model`, `/repository`, `/opt/arc/model` (also their `mini/` subfolder) | the Arc Mini folder: `config.json`, `tags.json`, `model.q4.safetensors`, `heads.safetensors`, `moonshine-streaming-tiny/` |
| `ARC_MODEL_ID` | `arc-mini` | the id `/v1/models` and responses report |
| `ARC_WORKERS` | 2 | engines (concurrent requests / realtime sessions); weights are memory-mapped and shared |
| `ARC_THREADS` | cores / workers | threads per engine |
| `ARC_MAX_AUDIO_S` | 900 | longest clip per request (memory grows with clip length: about 1 GB for 8 minutes with every read-out on) |
| `ARC_MAX_BODY_MB` | 100 | |
| `ARC_QUEUE_TIMEOUT_S` | 60 | how long a request waits for a free engine before 503 |
| `ARC_FILE_ROOT` | | Snowflake: a mounted stage; rows may then pass file paths under it |
| `ARC_MAX_CONNECTIONS`, `ARC_ACCESS_LOG` | 64, 1 | |

## Images

```bash
# bring-your-own weights: the platform mounts them (SageMaker /opt/ml/model, Hugging Face /repository)
docker buildx build --platform linux/amd64,linux/arm64 -f deploy/Dockerfile --target runtime -t arc-transcriber:0.1.0 .
# weights inside the image (Marketplace model package, Snowflake): fetch the public weights, then
HF_TOKEN=hf_... deploy/fetch_weights.sh mini
docker buildx build --platform linux/amd64,linux/arm64 -f deploy/Dockerfile --target bundled -t arc-transcriber:0.1.0-mini .
```

- Multi-stage: `debian:bookworm-slim` builds; the runtime is `gcr.io/distroless/cc-debian12:nonroot` (glibc only: no
  shell, no package manager), uid 65532, read-only weights owned by root. `HEALTHCHECK` uses the binary itself
  (`serve --healthcheck`). The binary is about 0.6 MB (macOS arm64 build); the bundled image adds the 68 MB of weights.
- One build serves amd64 and arm64: libarc picks its AVX2 (x86-64) or NEON (arm64) kernels at run time, with a
  portable C fallback.
- **Pro tier later**: the image does not care which weights it serves. A Pro image is the same Dockerfile with
  `--build-arg WEIGHTS=deploy/weights/<pro> --build-arg MODEL_ID=arc-pro` (bundled) or the same runtime image with
  other weights mounted, as long as the weights are in the Arc Mini folder format libarc reads. Private weights then
  go only into private registries / model packages; nothing in the code changes.

### Combined image: transcriber + Arc Tagger

`Dockerfile.combined` links [WhissleAI/arc-tagger-c](https://github.com/WhissleAI/arc-tagger-c) (native backend) into the
same binary: audio in, transcript plus the tagger's read-outs (PII, safety, intent, dialogue act, affect, roles,
events) out, as `arc_tags` on `verbose_json` and on realtime finals, plus `POST /v1/tags`. Both libraries link
cleanly (no symbol clashes; one cJSON 1.7.19). Build context: the folder holding both checkouts.

```bash
make -C deploy -j2 ARCTAG=/path/to/arc-tagger-c         # deploy/build/arc-serve-combined
ARC_MODEL_DIR=.../mini ARCTAG_MODEL_DIR=.../arc-tagger deploy/build/arc-serve-combined
cd .. && docker buildx build -f arc_transcriber/deploy/Dockerfile.combined --target runtime -t arc-transcriber-tagger:0.1.0 .
```

## GPU: there is no GPU path (and so no CUDA image)

libarc is CPU-only. `c/src/arc_kernels.c` (GEMM with fp32 / int8 / int4 weights, conv, attention, FFT) has AVX2+FMA
and NEON micro-kernels and a portable C fallback, run over libarc's own thread pool; there is no CUDA, Metal, OpenCL or
Vulkan code anywhere in `c/`. The optional Arc Transcriber Space backend runs its encoder through ONNX Runtime's **CPU**
provider only. The Python package's GPU use (`arc_audio`, Arc Transcriber full on CUDA) has no C counterpart yet: the
full backbone (Parakeet encoder) is not ported (`c/README.md`, "Plan: Arc Transcriber (full backbone)").

What that means for NVIDIA NIM: a NIM is a GPU-accelerated container, so Arc does not qualify today. The shortest GPU
paths: (1) Arc Transcriber Space, whose encoder and RNNT already sit behind `c/src/arc_nft2_be.h` on ONNX Runtime, can
take ONNX Runtime's CUDA / TensorRT execution providers; (2) the Parakeet encoder port, which `c/README.md` plans with a
CUDA path after the CPU one. See [`platforms/nim/`](platforms/nim/README.md).

## Model card (draft)

**Model**: Arc Mini (Arc v1.1, `WhissleAI/arc-v1.1`, folder `mini/`), served by libarc 1.4.1. Moonshine-streaming-tiny
speech encoder and decoder (4-bit packed) plus Arc's trunk and heads; 68 MB of weights.

**Does**: English transcripts, punctuated and cased, with inline tags on names, numbers and contacts and word times;
names linked to a caller-supplied list (with kinds, groups, aliases; also biases the decoder); numbers with values
(scores, durations, times of day, money, ordinals, dates, ...); per-utterance device-directed, intent and dialogue-act
read-outs; sound and channel events (silence, telephone / narrowband, clipping, ...); speakers (diarization turns, each
segment's speaker, a role read-out per speaker); n-best alternatives; streaming with partials and finals.

**Does not (in this C build)**: languages other than English; names / numbers linking inside streams;
Arc Transcriber (full, Parakeet backbone) and its accuracy; compressed audio formats (send WAV or PCM).

**Checked against the Python reference** (from `c/README.md`): golden parity on `tests/golden/mini.json` (transcript,
inline tags, linked names, number values identical); names, numbers and word times identical with `arc_audio` on 609
clip runs (8,744 words, 756 names, 443 numbers); tokenizer 320/320; word times to 1 ms; tag-head probabilities within
6.5e-5. This service adds its own parity run (below): server output vs `arc_audio` on the same audio.

**Speed and memory** (from `c/README.md`, Apple M3 Pro, 4 threads): real-time factor 0.026 for the transcript only,
0.046 with every read-out; load 30-90 ms (memory-mapped weights); peak memory about 180 MB for a 30 s clip with
everything, about 1 GB for 8 minutes. Raspberry Pi 5 streaming (`README.md`): RTF 0.24-0.37 on 4 threads, 129-136 MB.

**Accuracy**: benchmarks against other systems are on the model card, https://huggingface.co/WhissleAI/arc-v1.1. They
were measured with arc_audio; libarc reproduces arc_audio's decisions on the clips listed above.

**Dates**: a date number with a month name ("March third", "the third of March", "June twenty first, twenty twenty
five") keeps the month in `heard` and carries `date` ({year, month, day}) and `iso` ("--03-03"), as arc_audio does.
libarc 1.3.0 did not (its `heard` was "third"); this service's conformance run found it, and libarc 1.3.1 closes it.

**Speakers** (libarc 1.4.0): `verbose_json` and `arc_json` carry arc_audio's `speakers` and `diarization`, and each
segment its `speaker` (the turn it overlaps most, as arc_transcriber's lines). They come from Arc Mini's own heads (no
extra model, no new weights): the slot state machine (turns "S1", "S2" ...) or, in the TrackBank domains, the voice
embedding head (turns "T1", "T2" ...). On Arc Mini's synthetic multi-voice clips the slot machine mostly keeps one "S1"
for several TTS voices, while the TrackBank (`meeting`, or `diarization=voice`) separates them; arc_audio does exactly
the same, which is what parity means here. Checked: turns, speaker ids, role labels identical to arc_audio and role p
within 1e-4 (4-decimal rounding) on every clip of the conformance run and 16 more multi-voice clip runs (`unknown`,
`meeting`, `nhl`, `cricket`); realtime `session.closed` equals the offline result.

**Licence**: code and weights CC BY-NC 4.0 with the Whissle terms ([LICENSE](../LICENSE)); Moonshine-streaming-tiny inside
`mini/` is MIT. Commercial use, including a paid marketplace listing, needs Whissle's commercial licence: the listing's
EULA must be Whissle's commercial terms, not CC BY-NC.

## Conformance

```bash
deploy/conformance/run.sh /path/to/arc-v1.1/mini --python /path/to/python-with-arc_audio-deps
deploy/conformance/run.sh /path/to/arc-v1.1/mini --python ... --combined ../arc-tagger-c /path/to/arc-tagger-model
```

It builds the binary (`nice -n 19`, `make -j2`), starts it, and checks every route, every response format, the error
contract, resampled and raw-PCM input, SageMaker `/invocations` (audio, JSON, batch, JSON lines), Snowflake rows
(base64 and stage files, path traversal refused), realtime (finals equal the offline transcript), concurrency, and
**parity with arc_audio** on the same audio: `c/tools/dump_names.py` (Python) vs the server's `arc_json`, compared with
`c/tools/compare_names.py` (text, every word with times to 1 ms and inline tags, every name and number, the speakers,
the diarization turns and each utterance's speaker). Clips: `examples/example_synth.wav` with its names list, three
clips synthesized with the OS text-to-speech and two two- / three-voice dialogues (written to files, never played; the
dialogues in `unknown`, `meeting` and `nhl`, one slot and two TrackBank set-ups), a tone and silence. Realtime also
checks the dialogues' speakers (`unknown`, `meeting`).

**Partial names** (libarc 1.4.1): a name the linker leaves not_in_list / abstain is checked against the list once more,
as arc_audio does (kbsnap AliasIndex): one entry's name or alias with punctuation, case and spacing ignored links with
`via: "alias"`, and a part only one entry's name has ("Daniel" for "Daniel Brooks") with `via: "partial"`; a part two
entries share stays unlinked. The 1.4.0 conformance run found it on a dialogue saying a bare "Daniel". With a running Docker daemon it repeats everything against the image, after
checking that the image starts and is ready with `--network none`; it never starts Docker itself.
