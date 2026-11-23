# Examples

| file | what |
|---|---|
| [`quickstart.py`](quickstart.py) | Arc Mini (or full) offline and live on the bundled clip: transcript, names, numbers, events |
| [`quickstart_space.py`](quickstart_space.py) | Arc Transcriber Space offline and streaming (`pip install "arc-transcriber[space] @ ..."`) |
| [`cloud_asr.py`](cloud_asr.py) | Arc with a cloud recogniser's words (`[integrations]` extra and a provider key) |
| `example_synth.wav` | synthetic basketball commentary with fictional names (16 kHz mono) |
| `example_names.txt`, `example_names.json` | its names list, in both formats |
| `example_synth.json` | the clip's script: lines, voices and timings it was synthesised from |

```bash
python examples/quickstart.py
arc-transcribe examples/example_synth.wav --domain nba --names examples/example_names.txt
```
