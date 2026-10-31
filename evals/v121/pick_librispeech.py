"""Pick N random utterances (seed 0) from an extracted LibriSpeech split, write a manifest, delete the other FLAC files.

    python -I pick_librispeech.py LIBRISPEECH_DIR SPLIT N MANIFEST_OUT
"""
import glob
import json
import os
import random
import sys

root, split, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
utts = []
for tf in sorted(glob.glob(os.path.join(root, split, "*", "*", "*.trans.txt"))):
    d = os.path.dirname(tf)
    for line in open(tf):
        uid, text = line.strip().split(" ", 1)
        utts.append(dict(id=uid, wav=os.path.relpath(os.path.join(d, uid + ".flac"), os.path.dirname(os.path.abspath(out))),
                         ref=text))
random.Random(0).shuffle(utts)
keep = sorted(utts[:n], key=lambda u: u["id"])
json.dump(keep, open(out, "w"), indent=0)
keepf = {os.path.basename(u["wav"]) for u in keep}
for f in glob.glob(os.path.join(root, split, "*", "*", "*.flac")):
    if os.path.basename(f) not in keepf:
        os.remove(f)
print(split, len(utts), "->", len(keep))
