# Arc as a hosted service (`deploy/`)

`arc-serve` runs **Arc Mini** through libarc ([`c/`](../c/README.md)) behind a small HTTP + WebSocket server in C. No
Python, no ONNX Runtime, no network access at run time. One binary, one image, the same contract on every platform.

| route | what |
|---|---|
| `POST /v1/audio/transcriptions` | OpenAI-compatible, plus Arc's fields: word times, inline tags, linked names, number values, utterance tags, events, speakers |
| `WS /v1/realtime` | streaming: 16 kHz PCM16 in; partials, endpoints, finals, settled tags, events out |
| `GET /health`, `/v1/health/live`, `/v1/health/ready` | liveness and readiness |
| `GET /v1/models`, `/v1/models/{id}` | the loaded model (OpenAI list shape) |
| `GET /ping`, `POST /invocations`, `GET /execution-parameters` | Amazon SageMaker's container contract |
| `POST /v1/snowflake/transcribe` | Snowflake service-function rows |
| `POST /v1/tags`, `POST /v1/snowflake/tags` | combined image only: Arc Tagger |

The contract is in [`openapi.yaml`](openapi.yaml) (the Snowflake row routes are described in
[platforms/snowflake/](platforms/snowflake/README.md)). Platform guides and marketplace status:
[`platforms/`](platforms/README.md).

## Quick start

```bash
make -C deploy -j2                                   # deploy/build/arc-serve (builds libarc first)
ARC_MODEL_DIR=/path/to/arc-v1.1/mini deploy/build/arc-serve
curl -s localhost:8080/v1/audio/transcriptions -F file=@examples/example_synth.wav \
     -F names=@examples/example_names.json -F domain=nba -F response_format=verbose_json
```

OpenAI SDKs work unchanged (`base_url="http://host:8080/v1"`). Arc's fields come with `response_format=verbose_json`;
Arc's inputs (`names`, `domain`, ...) are ordinary form fields.

## Request options

Form fields, query parameters, JSON keys (with base64 `audio`), or SageMaker custom attributes
(`domain=nba;response_format=text`).

| option | default | |
|---|---|---|
| `file` | | WAV (any bit depth, channels, rate: resampled to 16 kHz mono) or raw PCM16 (`audio/pcm`, `sample_rate=`) |
| `response_format` | `json` | `json`, `text`, `verbose_json`, `srt`, `vtt`, `arc_json` (arc_audio's keys) |
| `names` | | names list: JSON array of strings or `{name, kind, group, aliases}`, or newline / comma separated |
| `prompt`, `terms`, `topic` | | decoder context: earlier text, vocabulary, topic |
| `domain` | `unknown` (`ARC_DOMAIN`) | `assistant`, `meeting`, `talk`, `news`, `nba`, `nfl`, `nhl`, `soccer`, `cricket`, `esports`, ... |
| `language` | | only `en`; anything else is refused with 400 |
| `events`, `words`, `trunk`, `link_names`, `numbers` | `true` | switch read-outs off to save time |
| `diarization` | `auto` | `auto` (by domain), `slots`, `fused`, `voice` |
| `one_utterance` | `false` | decode the clip as one utterance |
| `beams` | 4 | beam width (1 = greedy) |

Errors use OpenAI's envelope `{"error": {"message", "type", "code"}}`.

## Realtime

```
ws://host:8080/v1/realtime?domain=assistant&partials=400&events=1
-> binary PCM16 frames, or {"type":"input_audio_buffer.append","audio":"<base64>"}
-> {"type":"session.update","session":{"names":[...],"prompt":"..."}}  {"type":"input_audio_buffer.commit"}  {"type":"input_audio_buffer.finish"}
<- session.created, transcript.partial, transcript.endpoint, transcript.final, transcript.tags, event, frame, session.closed, error
```

Finals carry `speaker`; `session.closed` carries the stream's speakers and diarization. Names and numbers are not
linked in streams: use the file route for that.

## Configuration

| variable | default | |
|---|---|---|
| `PORT`, `HOST` | `8080`, `0.0.0.0` | SageMaker requires 8080 |
| `ARC_MODEL_DIR` | first of `/opt/ml/model`, `/repository`, `/opt/arc/model` (or their `mini/`) | the Arc Mini folder |
| `ARC_MODEL_ID` | `arc-mini` | the id responses report |
| `ARC_WORKERS` | 2 | engines (concurrent requests or sessions); weights are shared |
| `ARC_THREADS` | cores / workers | threads per engine |
| `ARC_MAX_AUDIO_S` | 900 | longest clip (memory grows with length: about 1 GB for 8 minutes) |
| `ARC_MAX_BODY_MB`, `ARC_QUEUE_TIMEOUT_S` | 100, 60 | |
| `ARC_FILE_ROOT` | | Snowflake: a mounted stage for file-path rows |
| `ARC_MAX_CONNECTIONS`, `ARC_ACCESS_LOG` | 64, 1 | |

## Images

```bash
# weights mounted by the platform (SageMaker /opt/ml/model, Hugging Face /repository)
docker buildx build --platform linux/amd64,linux/arm64 -f deploy/Dockerfile --target runtime -t arc-transcriber:0.1.0 .
# weights inside the image (Marketplace, Snowflake)
HF_TOKEN=hf_... deploy/fetch_weights.sh mini
docker buildx build --platform linux/amd64,linux/arm64 -f deploy/Dockerfile --target bundled -t arc-transcriber:0.1.0-mini .
```

- Runtime base: `gcr.io/distroless/cc-debian12:nonroot` (no shell, no package manager), uid 65532, read-only weights.
  `HEALTHCHECK` is the binary itself.
- One build serves amd64 and arm64: libarc picks AVX2 or NEON kernels at run time.
- **CPU only.** libarc has no CUDA, Metal or other GPU code, so there is no GPU image (see
  [platforms/nim/](platforms/nim/README.md)).
- Other weights in the Arc Mini folder format work with the same image (`--build-arg WEIGHTS=... --build-arg MODEL_ID=...`).

### Combined image: transcriber + Arc Tagger

`Dockerfile.combined` links [arc-tagger-c](https://github.com/WhissleAI/arc-tagger-c) into the same binary: audio in,
transcript plus `arc_tags` (PII, safety, intent, act, affect, roles, events) out, plus `POST /v1/tags`. Build context
is the folder holding both checkouts.

```bash
make -C deploy -j2 ARCTAG=/path/to/arc-tagger-c         # deploy/build/arc-serve-combined
cd .. && docker buildx build -f arc_transcriber/deploy/Dockerfile.combined --target runtime -t arc-transcriber-tagger:0.1.0 .
```

## Conformance

```bash
deploy/conformance/run.sh /path/to/arc-v1.1/mini --python /path/to/python-with-arc_audio-deps
deploy/conformance/run.sh /path/to/arc-v1.1/mini --python ... --combined ../arc-tagger-c /path/to/arc-tagger-model
```

It builds and starts the server and checks every route and response format, errors, resampling, SageMaker and
Snowflake shapes, realtime (finals equal the offline transcript), concurrency, and **parity with arc_audio** on the
same audio (text, word times, tags, names, numbers, speakers). With a running Docker daemon it repeats the run against
the image, after checking it starts with `--network none`.

## Folders

| folder | what |
|---|---|
| [`server/`](server/README.md) | the server's C source |
| [`conformance/`](conformance/README.md) | the conformance run |
| [`platforms/`](platforms/README.md) | Hugging Face, SageMaker / AWS Marketplace, Snowflake, NVIDIA NIM |
