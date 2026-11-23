# Conformance run

Builds `arc-serve`, starts it, and checks every route against the contract and against `arc_audio` on the same audio.
What it covers: [deploy/README.md](../README.md#conformance).

```bash
deploy/conformance/run.sh MINI_DIR [--python PY] [--skip-reference] [--combined ARCTAG_REPO TAGGER_MODEL_DIR] [--no-docker]
```

- `MINI_DIR`: the Arc Mini folder (`WhissleAI/arc-v1.1`, `mini/`).
- `PY`: a Python with arc_audio's dependencies, for the reference run.
- With a running Docker daemon it repeats the run against the image; it never starts Docker itself.

| file | what |
|---|---|
| `run.sh` | build, start, run, stop (native, then Docker) |
| `conformance.py` | the checks |
| `http_client.py` | a small HTTP / WebSocket client with no dependencies |
