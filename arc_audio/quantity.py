"""Typed quantities in a transcript: numbers, percents, distances, durations, speeds, temperatures ... with their numeric
value, read from the words as said ("forty five percent", "45 more per cent", "two kilometers", "shields to 40").

    quantities(['set', 'power', 'to', 'seventy', 'five', 'percent'])
    -> [dict(type='percent', value=75, unit='%', relative=0, word_begin=3, word_end=6, text='seventy five percent', written='75%')]

relative: +1 for "45 MORE percent" / "extra", -1 for "less" / "fewer", else 0. A number inside a name tag (ENTITY_*:
"Apollo sixteen") is not a quantity. A lone "one" / "a" is not a number without a unit ("this one"). The same rules are in
libarc (c/src/arc_post.c, arc_post_quantities); c/tests/test_post.c checks it against this module.
Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import re

ONES = {'zero': 0, 'oh': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
        'nine': 9}
TEENS = {'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
         'seventeen': 17, 'eighteen': 18, 'nineteen': 19}
TENS = {'twenty': 20, 'thirty': 30, 'forty': 40, 'fourty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80,
        'ninety': 90}
SCALES = {'thousand': 1000, 'million': 1000000, 'billion': 1000000000}
MORE = {'more': 1, 'extra': 1, 'additional': 1, 'less': -1, 'fewer': -1}

# (words, type, unit), longest first when matching
UNITS = [
    ('kilometers per hour', 'speed', 'km/h'), ('kilometres per hour', 'speed', 'km/h'), ('km per hour', 'speed', 'km/h'),
    ('miles per hour', 'speed', 'mph'), ('meters per second', 'speed', 'm/s'), ('metres per second', 'speed', 'm/s'),
    ('km/h', 'speed', 'km/h'), ('kph', 'speed', 'km/h'), ('mph', 'speed', 'mph'),
    ('percentage points', 'percent', '%'), ('per cent', 'percent', '%'), ('percent', 'percent', '%'),
    ('%', 'percent', '%'), ('pct', 'percent', '%'),
    ('kilometers', 'distance', 'km'), ('kilometer', 'distance', 'km'), ('kilometres', 'distance', 'km'),
    ('kilometre', 'distance', 'km'), ('km', 'distance', 'km'), ('klicks', 'distance', 'km'), ('klick', 'distance', 'km'),
    ('meters', 'distance', 'm'), ('meter', 'distance', 'm'), ('metres', 'distance', 'm'), ('metre', 'distance', 'm'),
    ('centimeters', 'distance', 'cm'), ('centimeter', 'distance', 'cm'), ('centimetres', 'distance', 'cm'),
    ('cm', 'distance', 'cm'), ('millimeters', 'distance', 'mm'), ('millimetres', 'distance', 'mm'),
    ('mm', 'distance', 'mm'), ('miles', 'distance', 'mi'), ('mile', 'distance', 'mi'), ('feet', 'distance', 'ft'),
    ('foot', 'distance', 'ft'), ('ft', 'distance', 'ft'), ('inches', 'distance', 'in'), ('inch', 'distance', 'in'),
    ('yards', 'distance', 'yd'), ('yard', 'distance', 'yd'), ('light years', 'distance', 'ly'),
    ('light year', 'distance', 'ly'), ('astronomical units', 'distance', 'au'),
    ('seconds', 'duration', 's'), ('second', 'duration', 's'), ('secs', 'duration', 's'), ('sec', 'duration', 's'),
    ('minutes', 'duration', 'min'), ('minute', 'duration', 'min'), ('mins', 'duration', 'min'),
    ('hours', 'duration', 'h'), ('hour', 'duration', 'h'), ('hrs', 'duration', 'h'), ('days', 'duration', 'd'),
    ('day', 'duration', 'd'), ('weeks', 'duration', 'wk'), ('week', 'duration', 'wk'), ('months', 'duration', 'mo'),
    ('month', 'duration', 'mo'), ('years', 'duration', 'yr'), ('year', 'duration', 'yr'),
    ('degrees celsius', 'temperature', 'C'), ('degrees centigrade', 'temperature', 'C'),
    ('degrees fahrenheit', 'temperature', 'F'), ('degrees kelvin', 'temperature', 'K'), ('celsius', 'temperature', 'C'),
    ('fahrenheit', 'temperature', 'F'), ('kelvin', 'temperature', 'K'), ('degrees', 'degrees', 'deg'),
    ('degree', 'degrees', 'deg'),
    ('kilograms', 'mass', 'kg'), ('kilogram', 'mass', 'kg'), ('kilos', 'mass', 'kg'), ('kg', 'mass', 'kg'),
    ('grams', 'mass', 'g'), ('gram', 'mass', 'g'), ('pounds', 'mass', 'lb'), ('pound', 'mass', 'lb'),
    ('lbs', 'mass', 'lb'), ('tons', 'mass', 't'), ('tonnes', 'mass', 't'), ('ton', 'mass', 't'),
    ('kilowatt hours', 'energy', 'kWh'), ('kilowatts', 'power', 'kW'), ('kilowatt', 'power', 'kW'),
    ('watts', 'power', 'W'), ('watt', 'power', 'W'), ('volts', 'voltage', 'V'), ('volt', 'voltage', 'V'),
    ('amps', 'current', 'A'), ('amp', 'current', 'A'), ('times', 'multiplier', 'x'),
]
_UNITS = sorted(((tuple(w.split()), t, u) for w, t, u in UNITS), key=lambda x: -len(x[0]))
_DIG = re.compile(r'^\d+(?:,\d{3})*(?:\.\d+)?$')


def tokens(words):
    """words as said -> lower-case tokens, each with the index of the word it came from ("40%" -> "40", "%")"""
    out = []
    for i, w in enumerate(words):
        t = str(w).lower().replace('’', "'")
        t = re.sub(r"[^\w%.,'/-]+", '', t).strip(".,'-")
        if not t:
            continue
        m = re.match(r'^(\d+(?:,\d{3})*(?:\.\d+)?)(%|km|cm|mm|kg|m|s)?$', t)
        if m:
            out.append((m.group(1), i))
            if m.group(2):
                out.append((m.group(2), i))
            continue
        for p in t.split('-'):
            if p:
                out.append((p, i))
    return out


def _num(tk, i):
    """the longest number at token i -> (value, next token, is_int, n_number_words) or None"""
    n = len(tk)
    t = tk[i][0]
    if _DIG.match(t):
        v = float(t.replace(',', ''))
        j = i + 1
        if j < n and tk[j][0] in SCALES:
            v *= SCALES[tk[j][0]]
            j += 1
        return v, j, float(v).is_integer(), 1
    total, cur, j, last, used = 0, 0, i, None, 0
    while j < n:
        w = tk[j][0]
        if w == 'a' and last is None and j + 1 < n and tk[j + 1][0] in ('hundred', 'thousand', 'million', 'billion'):
            cur, last = 1, 'a'
        elif w in ONES and w != 'oh':
            if last in ('one', 'teen'):
                break
            if last == 'ten' and cur % 10:
                break
            cur += ONES[w]
            last = 'one'
        elif w in TEENS:
            if last in ('one', 'teen', 'ten'):
                break
            cur += TEENS[w]
            last = 'teen'
        elif w in TENS:
            if last in ('one', 'teen', 'ten'):
                break
            cur += TENS[w]
            last = 'ten'
        elif w == 'hundred':
            if last == 'hundred' or (cur % 100 == 0 and cur > 0 and last != 'a'):
                break
            cur = (cur or 1) * 100
            last = 'hundred'
        elif w in SCALES:
            if last is None:
                break
            total += (cur or 1) * SCALES[w]
            cur, last = 0, 'scale'
        elif w == 'and' and last in ('hundred', 'scale') and j + 1 < n and (tk[j + 1][0] in ONES or tk[j + 1][0] in TEENS or tk[j + 1][0] in TENS):
            j += 1
            continue
        else:
            break
        used += 1
        j += 1
    if last is None or last == 'a':
        return None
    v = float(total + cur)
    is_int = True
    if j + 1 < n and tk[j][0] == 'point' and tk[j + 1][0] in ONES:     # "two point five"
        k, frac = j + 1, ''
        while k < n and tk[k][0] in ONES:
            frac += str(ONES[tk[k][0]])
            k += 1
        v = float(f'{int(v)}.{frac}')
        is_int, j = False, k
    elif j + 2 < n and tk[j][0] == 'and' and tk[j + 1][0] == 'a' and tk[j + 2][0] == 'half':
        v += 0.5
        is_int, j = False, j + 3
    return v, j, is_int, used


def _unit(tk, j):
    for ws, t, u in _UNITS:
        if tuple(x[0] for x in tk[j:j + len(ws)]) == ws:
            return t, u, j + len(ws)
    return None


def _fmt(v, is_int):
    return str(int(v)) if is_int else ('%g' % v)


def quantities(words, tags=None):
    """words (strings, as said) [+ the word tags, e.g. 'ENTITY_MISSION', or None] -> [dict(type, value, unit, relative,
    word_begin, word_end, text, written)]"""
    tk = tokens(words)
    out, i, n = [], 0, len(tk)
    skip = lambda wi: bool(tags) and wi < len(tags) and str(tags[wi] or '').startswith('ENTITY_')
    while i < n:
        neg = 0
        if tk[i][0] in ('minus', 'negative') and i + 1 < n:
            r = _num(tk, i + 1)
            neg = 1 if r else 0
        else:
            r = _num(tk, i)
        if not r or skip(tk[i][1]):
            i += 1
            continue
        v, j, is_int, used = r
        if neg:
            v = -v
        rel = 0
        if j < n and tk[j][0] in MORE and j + 1 < n and _unit(tk, j + 1):
            rel = MORE[tk[j][0]]
            j += 1
        un = _unit(tk, j) if j < n else None
        if un:
            typ, unit, j = un
        else:
            typ, unit = 'number', None
            if rel == 0 and j < n and tk[j][0] in MORE:
                rel = MORE[tk[j][0]]           # "give it ten more": relative, unit left to the app
                j += 1
        if typ == 'number' and used == 1 and tk[i][0] in ('one', 'a') and not neg and not rel:
            i += 1
            continue
        wb, we = tk[i][1], tk[j - 1][1] + 1
        written = _fmt(v, is_int) + ('' if unit is None else ('%' if unit == '%' else ('x' if unit == 'x' else ' ' + unit)))
        out.append(dict(type=typ, value=int(v) if is_int else round(v, 6), unit=unit, relative=rel, word_begin=wb,
                        word_end=we, text=' '.join(str(w) for w in words[wb:we]), written=written))
        i = j
    return out


def attach(utt):
    """utterance dict with 'words' (text, start, end, tag) -> utt['quantities'] with times (start of its first word, end
    of its last)"""
    ws = utt.get('words') or []
    qs = quantities([w['text'] for w in ws], [w.get('tag') for w in ws])
    for q in qs:
        q['start'] = ws[q['word_begin']]['start']
        q['end'] = ws[q['word_end'] - 1]['end']
    utt['quantities'] = qs
    return qs
