# arc_transcriber

Inference code, a CLI and a C library for **Arc**, a streaming audio model that returns a tagged transcript and
read-outs every 100 ms.

## What you get

- A **transcript with inline tags** on names, numbers and entities, plus a written form ("2:30 p.m.", "$45").
- **Names linked to your own list**, and **numbers with values** (scores, clocks, money, dates, durations).
- **Speakers and roles**, per-utterance **intents and acts**, and **device-directed** speech.
- **Sound and channel events**: silence, telephone audio, clipping, degraded audio.
- The same output from Python (`arc_audio`), a dependency-free C library (`libarc`) and an HTTP / WebSocket server.

## Models

| model | `--backbone` | download | runs on | use it for |
|---|---|---|---|---|
| **Arc Mini** | `mini` (default; alias `edge`) | about 71 MB | CPU; GPU optional | apps, laptops, on-device, lowest latency |
| **Arc Transcriber** (full) | `full` (aliases `parakeet`, `server`) | about 77 MB + NVIDIA Parakeet-TDT 0.6B v2 encoder (about 2.5 GB) | GPU recommended; CPU for files | best transcript and name accuracy |
| **Arc Transcriber Space** | `nft2` | separate repo, see [docs/SPACE.md](docs/SPACE.md) | CPU (int8 ONNX) | voice apps about the Moon and space flight |

English, 16 kHz mono (other rates are resampled). Weights download on first use.

## Quickstart

```bash
pip install git+https://github.com/WhissleAI/arc_transcriber
hf auth login        # once: accept the terms at https://huggingface.co/WhissleAI/arc-v1.1 first (approval is automatic)
arc-transcribe examples/example_synth.wav --domain nba --names examples/example_names.txt
```

Output, one line per utterance:

```
[   2.40-   7.00] S1: [Marcus Delaney](name: Marcus Delaney) brings the ball up the floor fourth quarter, [three minutes](duration: 180 s) left.
[  15.40-  19.40] S1: That puts the Harbor City Hawks ahead, [102 to 98.](score: 102-98)
```

Python:

```python
from arc_audio import Arc

arc = Arc.from_pretrained("WhissleAI/arc-v1.1", backbone="mini", device="cpu", threads=4)
out = arc.predict("game.wav", domain="nba", names=["Marcus Delaney", "Harbor City Hawks"])
for n in out["names"]:
    print(n["heard"], n["entity"], n["state"])
```

Requirements: Python 3.10+, torch 2.1+, transformers 5.19+. Extras: `[live]` (microphone, live external ASR),
`[integrations]` (cloud recognisers), `[space]` (Arc Transcriber Space), `[test]`.

## Documentation

| doc | covers |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | CLI options, Python API, streaming, output schema, device-directed scores, written form, external ASR |
| [docs/SPACE.md](docs/SPACE.md) | Arc Transcriber Space: names lists, live events, echo, room read-outs |
| [docs/NAMES_LISTS.md](docs/NAMES_LISTS.md) | how to build names lists |
| [docs/PORTING.md](docs/PORTING.md) | running Arc on other runtimes (ONNX Runtime, MLX, Core ML), parity checks |
| [docs/README.md](docs/README.md) | all docs |

## Deployment

| where | folder | what |
|---|---|---|
| C / C++ apps | [c/](c/README.md) | libarc: one header, no Python, runs Arc Mini and Arc Transcriber Space |
| Hosted service | [deploy/](deploy/README.md) | `arc-serve`: OpenAI-compatible HTTP, WebSocket, distroless image |
| Cloud platforms | [deploy/platforms/](deploy/platforms/README.md) | SageMaker / AWS Marketplace, Snowflake, Hugging Face, NVIDIA NIM: status and checklists |
| Raspberry Pi | [raspberry-pi/](raspberry-pi/README.md) | Arc Mini on a Pi 4 or 5 |
| macOS / Android | [macos/](macos/README.md) | Arc with the OS's on-device recogniser |
| Cloud ASR | [arc_audio/integrations/](arc_audio/integrations/README.md) | Arc read-outs on Deepgram, AssemblyAI or Speechmatics words |

## Tests

```bash
pip install "arc-transcriber[test] @ git+https://github.com/WhissleAI/arc_transcriber"
pytest -q                          # Arc Mini on CPU; skipped when offline
ARC_TEST_PARAKEET=1 pytest -q      # also Arc Transcriber (full)
```

More test options are in [docs/USAGE.md](docs/USAGE.md#tests).

## Licence

Code: CC BY-NC 4.0 plus the Whissle terms in [LICENSE](LICENSE), non-commercial use only. Weights: Arc weights
CC BY-NC 4.0 under the same terms; the modified Moonshine Streaming tiny model is MIT; Arc Transcriber (full) uses
NVIDIA Parakeet-TDT 0.6B v2 (CC-BY-4.0), downloaded from NVIDIA's repository. See [NOTICE](NOTICE).

- **Commercial use** (in a product or service, paid or hosted services, business use, or commercial use of outputs)
  needs a separate written licence from Whissle Inc.: ksingla@whissle.ai, https://whissle.ai.
- Whissle Inc. owns Arc and keeps all rights not expressly granted; the public licences do not limit its own use of Arc
  or its right to license Arc on other terms. No patent or trademark rights are granted; "Whissle", "Whissle Arc",
  "Arc Mini" and "Arc Transcriber" are Whissle Inc. trademarks.
- Contributions: see [CONTRIBUTING.md](CONTRIBUTING.md) (includes a licence grant that lets Whissle Inc. relicense them).

## Links

- Model weights, model card and benchmarks: https://huggingface.co/WhissleAI/arc-v1.1
- Arc Transcriber Space: https://huggingface.co/WhissleAI/arc-transcriber-space
- Arc Tagger (text tagging, pairs with Arc): https://github.com/WhissleAI/arc-tagger-c
- SDKs: https://whissle.ai/sdks
- Bugs: [Issues](../../issues)
