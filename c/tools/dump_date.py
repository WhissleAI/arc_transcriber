"""Spoken-date reference cases (arc_audio written.parse_date / date_iso, api.extend_date + tidy.tidy_number) as JSON lines
for tests/test_date.c.

    python tools/dump_date.py tests/data/date.jsonl

"parse" lines: words -> parse_date (date or null) and date_iso. "extend" lines: a transcript (words with times) and a
date record over some of them (as the decoder tags it: the number words) -> the record after extend_date and
tidy_number (heard, start, end, date, iso; no date / iso when none was read).

Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import json, sys
from arc_audio import api as A
from arc_audio import tidy as T
from arc_audio.written import parse_date, date_iso

# (transcript, the record's words as [first, last] word indexes): month + ordinal, month + cardinal, ordinal "of"
# month, with / without a year, digits, punctuation stops, and phrases that are not clean dates
EXTEND = [
    ("Remind me on March third.", (4, 4)),
    ("Remind me on March third", (4, 4)),
    ("the meeting is on March 3", (5, 5)),
    ("the meeting is on March 3rd.", (5, 5)),
    ("we met on the third of March", (4, 4)),
    ("we met on the third of March.", (4, 4)),
    ("on the twenty first of June we fly", (2, 3)),
    ("it opens June twenty first, twenty twenty five", (3, 4)),
    ("it opens June twenty first, twenty twenty five.", (3, 7)),
    ("born on July fourth, nineteen ninety", (3, 5)),
    ("born on July fourth nineteen ninety", (3, 3)),
    ("due March third, twenty twenty four", (2, 5)),
    ("due March third twenty twenty four", (2, 2)),
    ("due March 3, 2024", (2, 3)),
    ("see you December twenty fifth", (3, 4)),
    ("see you December twenty five", (3, 4)),
    ("see you December 25", (3, 3)),
    ("paid on January first", (3, 3)),
    ("paid on January one", (3, 3)),
    ("the first of January is a holiday", (1, 1)),
    ("the thirty first of October", (1, 2)),
    ("the thirty first of September", (1, 2)),
    ("February twenty ninth, twenty twenty four", (1, 5)),
    ("February twenty ninth, twenty twenty three", (1, 5)),
    ("February twenty ninth", (1, 2)),
    ("February thirtieth", (1, 1)),
    ("in twenty twenty four we grew", (1, 3)),
    ("in nineteen eighty four", (1, 3)),
    ("since 1999", (1, 1)),
    ("by May fifth", (2, 2)),
    ("by may fifth", (2, 2)),
    ("in Sept tenth", (2, 2)),
    ("on the tenth", (2, 2)),
    ("on the 10th", (2, 2)),
    ("March, third", (1, 1)),
    ("the third. Of March", (1, 1)),
    ("April eleventh of next year", (1, 1)),
    ("August two thousand and five", (1, 4)),
    ("October the fifth", (2, 2)),
    ("November fifteenth twenty", (1, 2)),
    ("“March third,” she said", (1, 1)),
    ("March fourth and fifth", (1, 1)),
    ("on March fifth at noon", (2, 2)),
    ("the fifth of May, two thousand", (1, 1)),
    ("the second of the month", (1, 1)),
]
PARSE = [
    "March third", "the third of March", "March third, twenty twenty four", "twenty twenty four", "March 3", "March 3rd",
    "March 3, 2024", "third of June", "June twenty first", "June twenty one", "June thirty first", "June thirty", "February 29",
    "February 29, 2023", "the third", "3rd", "nineteen ninety", "two thousand and five", "March", "March April third",
    "third fourth fifth", "March third and fourth", "July fourth, nineteen seventy six", "may fifth", "Sept tenth",
    "on the fifth in May", "the fifth for May", "June first, 1999", "twelve", "December twelve", "December one hundred",
    "December twenty five thousand", "January 1st, 3000", "January 1st, 999",
]


def words_of(text):
    """the transcript's words with times: 0.3 s each, 0.05 s apart"""
    W, t = [], 0.0
    for w in text.split():
        W.append(dict(w=w, text=w, start=round(t, 3), end=round(t + 0.3, 3)))
        t += 0.35
    return W


out = open(sys.argv[1], "w")
for p in PARSE:
    ws = p.split()
    d = parse_date(ws)
    out.write(json.dumps(dict(k="parse", words=ws, date=d, iso=date_iso(d))) + "\n")
for text, (a, b) in EXTEND:
    W = words_of(text)
    r = dict(kind="date", heard=" ".join(w["w"] for w in W[a:b + 1]), start=W[a]["start"], end=W[b]["end"])
    rec = dict(k="extend", words=[[w["w"], w["start"], w["end"]] for w in W], i=a, j=b, heard_in=r["heard"],
               start_in=r["start"], end_in=r["end"])
    A.extend_date(r, W, a, b)
    T.tidy_number(r)
    rec.update(heard=r["heard"], start=r["start"], end=r["end"], date=r.get("date"), iso=r.get("iso"), has_date="date" in r)
    out.write(json.dumps(rec) + "\n")
