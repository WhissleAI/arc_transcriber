"""Reference cases for arc_num.c (tests/test_num.c) from arc_audio numlib.py / numvalue.py / textlayer.py.

    python -I c/tools/dump_num.py [-o c/tests/data/num.jsonl]

One JSON object per line, "k" naming the function: compose, finalize, tv (text_value), unit (unit_of), to (tokens_of),
wt (word_tokens), word (numeric_word / connector / has_num).

Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import argparse, json, os, random
from arc_audio import numlib as NL, numvalue as NV, textlayer as TL
from arc_audio.engine import numeric_word, connector

KINDS = NV.KINDS + [None]
WORDS = """zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen
eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion point and a an
half quarter dozen oh o nil nought naught love to for dash minus negative couple first second third fourth fifth sixth
twelfth twentieth hundredth thousandth millionth seconds minutes hour hours days weeks months years decades fortnight
am pm a.m. p.m. o'clock oclock dollars dollar bucks $ cents euros € pounds quid £ yen rupees miles feet inches yards
meters km kilos kg ounces gallons liters degrees mph acres times people old points rebounds assists runs wickets overs
saves shots goals kills steals blocks sacks touchdowns catches threes turnovers fouls boundaries sixes fours hits
strikeouts ppg run page pages exhibit section chapter channel docket rule line flight room case item number no.
jersey per percent left the Brandt scored up quarter. half, three, four. seven? 3 12 102 98 2023 1,500 3.7 4.6 0.5
$80,000 £20 €15 #7 11th 1st 2nd 3rd 21st 90s 50% 5k 2m 102-98 4:15 9:30 twenty-six forty-two ninety-nine one-hundred
twenty-first Ninety TWENTY Four, Twenty-Six. (seven) "eight" five's ten’s it’s 1990s 3.14159 007 -5 - , . 'tis
seventy-five thirty-three.""".split()


def num(x):
    return x


def val(r):
    if r is None or r.get('v') is None:
        return None
    return [r['v'], r.get('v2')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tests', 'data', 'num.jsonl'))
    a = ap.parse_args()
    rnd = random.Random(1234)
    out = []
    add = lambda **d: out.append(json.dumps(d, ensure_ascii=False, separators=(',', ':')))
    # compose / finalize on numlib's own test cases and random token lists
    for w, k, _ in [(['twenty', 'five'], None, 0), (['a', 'hundred', 'twenty', 'four', 'point', 'six'], None, 0), (['two', 'thousand', 'twenty', 'five'], None, 0),
                    (['twenty', 'twenty', 'three'], 'date', 0), (['nine', 'thirty', 'six'], 'clock', 0), (['4:15'], 'clock', 0), (['$80,000'], None, 0),
                    (['3.7'], None, 0), (['hundred', 'and', 'one', 'for', 'six'], 'score', 0), (['two', 'oh', 'five'], None, 0), (['102-98'], 'score', 0),
                    (['one', 'twenty'], 'score', 0), (['third'], 'ordinal', 0), (['11th'], 'ordinal', 0), (['seven', 'and', 'a', 'half'], None, 0),
                    (['Ninety', 'four,', 'ninety', 'three'], 'score', 0), (['two', 'one'], 'score', 0), (['six', 'ten'], 'measure', 0), (['6671'], 'ref_number', 0),
                    (['three', 'minutes', 'left'], 'clock', 0), (['first', 'quarter.'], 'ordinal', 0), (['$4', 'a', 'gallon'], 'money', 0), (['dozen'], 'stat', 0),
                    (['one', 'million'], None, 0)]:
        tok, _ = NL.tokens_of(w)
        add(k='compose', tok=tok, kind=k, r=val(NL.compose(tok, k)))
    voc = NL.NWT[1:] + ['UNITW', 'SEP', 'OH', 'NEG', 'point', 'and', 'a', 'half', '<none>']
    small = NL.UNITS[:10] + ['twenty', 'thirty', 'ninety', 'hundred', 'thousand', 'OH', 'SEP', 'point', 'UNITW', 'NEG', 'and', 'a', 'half', 'quarter', 'dozen']
    for i in range(1000):
        pool = voc if i % 2 else small
        tok = [rnd.choice(pool) for _ in range(rnd.randint(1, 6))]
        kind = rnd.choice(KINDS)
        try:
            r = val(NL.compose(tok, kind))
        except Exception:
            r = 'err'
        add(k='compose', tok=tok, kind=kind, r=r)
    units = NV.UNITS + [None, None, None]
    for i in range(800):
        pool = voc if i % 3 == 0 else small
        tok = [rnd.choice(pool) for _ in range(rnd.randint(1, 5))]
        kind = rnd.choice(KINDS[:-1] + ['time_of_day', 'duration', 'clock', 'time_of_day', 'duration'])
        unit = rnd.choice(units if i % 2 else ['minute', 'hour', 'second', 'month', 'year', 'am', 'pm', 'oclock', None])
        try:
            r = val(NV.finalize(kind, tok, unit))
        except Exception:
            r = 'err'
        add(k='finalize', tok=tok, kind=kind, unit=unit, r=r)
    # words: numeric_word, connector, has_num, tokens_of, word_tokens, unit_of
    for w in WORDS + ['', ' ', 'one,', 'and', 'Point', 'thirty-', '-thirty', 'a-half', 'four-oh', "o'", '1,2,3', '12.', '.5', '$', '#',
                      '10s', '10th.', '1.5k', '2.0m', 'eleventh', 'secondly', 'seconds,']:
        nw = numeric_word(w)
        add(k='word', w=w, num=nw, conn=connector(w) if nw else None, has=TL.has_num(w))
    phrases = ['twenty six points', 'three minutes left', '102 to 98', 'fourth quarter', 'two hundred and five', 'nine thirty pm',
               'seven thirty a.m.', "five o'clock", 'ten fifteen', 'a hundred dollars', '$4 a gallon', '15 euros', 'twenty quid', 'one, two, three',
               'one, and two', 'two seconds', 'second two', 'one second', 'three and a half hours', 'a quarter', 'half a dozen', 'a dozen eggs',
               'six foot ten', 'five miles per hour', '60 mph', 'page 12', 'room 304', 'number 23', 'jersey 7', 'twenty twenty three',
               'nineteen ninety nine', 'oh seven', 'two oh five', 'four point five', 'point five', 'minus five', 'negative two degrees',
               '50 percent', 'fifty per cent', '3-2', '4:15', '11:59 pm', '12 am', '12 pm', 'twelve thirty', 'nine', 'first', 'the 3rd',
               'third and long', 'four for six', 'ten to two', 'one hundred one', 'a thousand', 'two million', 'three billion',
               '1,500 people', 'eighty years old', 'two weeks', 'a fortnight', 'six months', '3 decades', 'two days', '4 hrs', '10 secs',
               'twenty-six points', 'Ninety four, ninety three', 'ten,', 'seven.', 'one two three four', 'oh oh seven', 'five, six',
               'a', 'the', 'four fours', 'twelve rebounds', 'nine assists', '2 sixes', 'ninety nine runs', 'first quarter.', '7:30pm',
               '7 p.m.', '8 a m', 'p.m. seven', '9am', 'six PM', 'quarter to nine', 'half past four', '1990s', 'the 90s', '2nd half']
    for p in phrases:
        ws = p.split()
        t, o = NL.tokens_of(ws)
        add(k='to', w=ws, toks=t, ordf=o)
        pr, o2 = NV.word_tokens(ws)
        add(k='wt', w=ws, toks=[x for _, x in pr], wi=[i for i, _ in pr], ordf=o2)
        for kind in rnd.sample(NV.KINDS, 3):
            prev = rnd.choice(['', 'page', 'Room', 'no.', 'the', '"Number', 'chapter'])
            add(k='unit', kind=kind, w=ws, prev=prev, u=NV.unit_of(kind, ws, prev))
    # text_value: phrases with every kind, next / previous words, given units
    nexts = [[], ['points'], ['minutes'], ['dollars'], ['percent'], ['pm'], ["o'clock"], ['years'], ['left'], ['miles', 'per'], ['euros']]
    for p in phrases:
        ws = p.split()
        for kind in rnd.sample(KINDS + [''], 4):
            nx = rnd.choice(nexts)
            prev = rnd.choice(['', 'page', 'the', 'number', 'Room', 'about'])
            unit_in = rnd.choice([None] * 5 + ['minute', 'hour', 'pm', 'USD', 'points'])
            v, u = TL.text_value(kind, ws, nx, prev, unit=unit_in)
            add(k='tv', kind=kind, w=ws, next=nx, prev=prev, unit_in=unit_in, r=None if v is None else [v['v'], v['v2']], u=u)
    for i in range(350):
        ws = [rnd.choice(WORDS) for _ in range(rnd.randint(1, 4))]
        kind = rnd.choice(KINDS)
        nx = rnd.choice(nexts)
        prev = rnd.choice(WORDS + [''])
        v, u = TL.text_value(kind, ws, nx, prev)
        add(k='tv', kind=kind, w=ws, next=nx, prev=prev, unit_in=None, r=None if v is None else [v['v'], v['v2']], u=u)
    with open(a.o, 'w') as f:
        f.write('\n'.join(out) + '\n')
    print('wrote', len(out), 'cases to', a.o, os.path.getsize(a.o), 'bytes')


if __name__ == '__main__':
    main()
