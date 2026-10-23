---
name: Bug report
about: Something crashes, errors, or gives a clearly wrong record
labels: bug
---

**What happened**

**Command or code that reproduces it**

```bash
arc-transcribe clip.wav --domain nba --names names.txt --asr builtin --format json
```

**Expected**

**Environment**: OS, Python, `pip show torch transformers arc-transcriber`, CPU or CUDA (GPU model), `--asr` mode

**Audio** (optional): a short clip that shows the problem, if you can share it. Never paste API keys.
