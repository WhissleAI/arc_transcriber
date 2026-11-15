"""tidy.py reference cases as JSON lines for tests/test_tidy.c.

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
    # names with numbers: the spoken number is part of the name (digits, roman numerals, years, hyphenated numbers)
]
def _rec(heard, entity, t=6.0):
    ws = heard.split()
    return dict(heard=heard, entity=entity, state="linked", via=None, p=0.9, start=t, end=t + 0.3 * len(ws),
                words=[dict(w=w, start=round(t + 0.3 * i, 3), end=round(t + 0.3 * (i + 1), 3)) for i, w in enumerate(ws)])
recs += [_rec("Surveyor six", "Surveyor 6"), _rec("the Apollo twelve landing", "Apollo 12"), _rec("Artemis two", "Artemis II"),
         _rec("Luna twenty-four", "Luna 24"), _rec("Apollo nineteen sixty nine", "Apollo 1969"), _rec("Apollo eleven", "Apollo XI"),
         _rec("Station one oh five", "Station 1905"), _rec("Apollo one", "Apollo I"), _rec("Apollo twenty one", "Apollo 21-B"),
         _rec("Mons seven", "Mons Hadley")]
tidied = [T.tidy_name(json.loads(json.dumps(r))) for r in recs]
for r, t in zip(recs, tidied):
    out.write(json.dumps(dict(k="name", i=r, heard=t["heard"], start=t["start"], end=t["end"])) + "\n")
d = T.dedupe_names(json.loads(json.dumps(tidied)))
out.write(json.dumps(dict(k="dedupe", i=tidied, kept=[(r["heard"], r.get("entity"), r["start"]) for r in d])) + "\n")
