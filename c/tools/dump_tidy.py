"""tidy.py reference cases (arc_transcriber 1.1.1) as JSON lines for tests/test_tidy.c.

Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import json, sys
from arc_audio import tidy as T
words = ["Brandt.", "“Hello,”", "...", "$5", "£20,", "€3!", "100%", "rock'n'roll", "O'Brien's", "(maybe)", "--", "twenty-six", "twenty-", "4.30",
         "fourth", "Fourteenth,", "nil", "one", "hundred.", "pts", "x", "", "  spaced  ", "naïve", "café—", "#1", "@home", "+44", "a.m.", "o'clock"]
phrases = ["102 to 98.", "has twenty six points tonight", "the score was nil to one", "about $ 5 million dollars more", "no 7 shirt",
           "minus three degrees outside", "fourth quarter", "three minutes left.", "Hello there", "", "it costs £20 per month and"]
out = open(sys.argv[1], "w")
for w in words:
    out.write(json.dumps(dict(k="word", i=w, clean=T.clean_word(w), num=bool(T._numeric(w)))) + "\n")
for p in phrases:
    out.write(json.dumps(dict(k="phrase", i=p, clean=T.clean_text(p), num=T.tidy_number(dict(heard=p))["heard"], keep=T.keep_number(dict(heard=p)))) + "\n")
recs = [
    dict(heard="to Theo Brandt.", entity="Theo Brandt", state="linked", via=None, p=0.9, start=1.0, end=2.0,
         words=[dict(w="to", start=1.0, end=1.2), dict(w="Theo", start=1.2, end=1.5), dict(w="Brandt.", start=1.5, end=2.0)]),
    dict(heard="Marcus Delaney brings", entity=None, state="not_in_list", via=None, p=0.4, start=3.0, end=4.2,
         words=[dict(w="Marcus", start=3.0, end=3.4), dict(w="Delaney", start=3.4, end=3.9), dict(w="brings", start=3.9, end=4.2)]),
    dict(heard="Brandt", entity="Theo Brandt", state="linked", via="snap", p=0.5, start=1.5, end=2.0, words=[dict(w="Brandt", start=1.5, end=2.0)]),
    dict(heard="the Hawks,", entity="Harbor City Hawks", state="linked", via="snap", p=0.6, start=5.0, end=5.6,
         words=[dict(w="the", start=5.0, end=5.2), dict(w="Hawks,", start=5.2, end=5.6)]),
]
tidied = [T.tidy_name(json.loads(json.dumps(r))) for r in recs]
for r, t in zip(recs, tidied):
    out.write(json.dumps(dict(k="name", i=r, heard=t["heard"], start=t["start"], end=t["end"])) + "\n")
d = T.dedupe_names(json.loads(json.dumps(tidied)))
out.write(json.dumps(dict(k="dedupe", i=tidied, kept=[(r["heard"], r.get("entity"), r["start"]) for r in d])) + "\n")
