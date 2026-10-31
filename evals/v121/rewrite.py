"""Re-render the written transcript of run_set.py rows with the installed arc_audio.written (for example this branch's fix),
from the saved tagged words, without running the model again. The old written text is kept as `written_before`.

    python -I rewrite.py rows.jsonl rows_fixed.jsonl
"""
import json
import sys

from arc_audio.written import written_words

src, dst = sys.argv[1], sys.argv[2]
with open(dst, "w") as f:
    for line in open(src):
        r = json.loads(line)
        if "id" in r:
            r["written_before"] = r["written"]
            r["written"] = " ".join(" ".join((written_words(ws)[0] if ws else "").strip() for ws in r["utts"]).split())
        f.write(json.dumps(r) + "\n")
