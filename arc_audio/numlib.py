import re
UNITS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
TENS = ['twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
SCALES = ['hundred', 'thousand', 'million', 'billion']
MISC = ['point', 'and', 'a', 'half', 'quarter', 'dozen', 'SEP', 'OH', 'NEG']
NWT = ['<none>'] + UNITS + TENS + SCALES + MISC + ['UNITW']
NWT_ID = {t: i for i, t in enumerate(NWT)}
VAL = {w: i for i, w in enumerate(UNITS)}
VAL.update({w: 20 + 10 * i for i, w in enumerate(TENS)})
SC = {'hundred': 100, 'thousand': 1000, 'million': 10 ** 6, 'billion': 10 ** 9}
ORD = {'first': 'one', 'second': 'two', 'third': 'three', 'fourth': 'four', 'fifth': 'five', 'sixth': 'six', 'seventh': 'seven', 'eighth': 'eight', 'ninth': 'nine', 'tenth': 'ten', 'eleventh': 'eleven', 'twelfth': 'twelve', 'thirteenth': 'thirteen', 'fourteenth': 'fourteen', 'fifteenth': 'fifteen', 'sixteenth': 'sixteen', 'seventeenth': 'seventeen', 'eighteenth': 'eighteen', 'nineteenth': 'nineteen', 'twentieth': 'twenty', 'thirtieth': 'thirty', 'fortieth': 'forty', 'fiftieth': 'fifty', 'sixtieth': 'sixty', 'seventieth': 'seventy', 'eightieth': 'eighty', 'ninetieth': 'ninety', 'hundredth': 'hundred', 'thousandth': 'thousand', 'millionth': 'million'}
ALIAS = {'oh': 'OH', 'o': 'OH', 'nil': 'zero', 'nought': 'zero', 'naught': 'zero', 'love': 'zero', 'to': 'SEP', 'for': 'SEP', 'dash': 'SEP', '-': 'SEP', 'minus': 'NEG', 'negative': 'NEG', 'an': 'a', 'couple': None}

def int_words(n):
    n = int(n)
    if n < 20:
        return [UNITS[n]]
    if n < 100:
        return [TENS[n // 10 - 2]] + ([UNITS[n % 10]] if n % 10 else [])
    for s, v in (('billion', 10 ** 9), ('million', 10 ** 6), ('thousand', 1000), ('hundred', 100)):
        if n >= v:
            return int_words(n // v) + [s] + (int_words(n % v) if n % v else [])
    return []

def tokens_of(words):
    out, ordf = ([], False)
    for wi, w in enumerate(words):
        if out and out[-1] != 'SEP' and (wi > 0) and words[wi - 1].rstrip().endswith(',') and (w.strip().lower() not in ('and', 'point')):
            out.append('SEP')
        w0 = w.lower().strip('.,?!;:"\'()').replace('’', "'")
        w0 = re.sub("'s$", '', w0)
        if not w0:
            continue
        m = re.fullmatch('[$£€#]?(\\d[\\d,]*)(\\.\\d+)?(st|nd|rd|th|s|%|k|m)?', w0)
        if m:
            out += int_words(m.group(1).replace(',', ''))
            if m.group(2):
                out += ['point'] + [UNITS[int(c)] for c in m.group(2)[1:]]
            if m.group(3) in ('st', 'nd', 'rd', 'th'):
                ordf = True
            if m.group(3) == 'k':
                out += ['thousand']
            if m.group(3) == 'm':
                out += ['million']
            continue
        m = re.fullmatch('(\\d+)[:\\-](\\d+)', w0)
        if m:
            out += int_words(m.group(1)) + ['SEP'] + int_words(m.group(2))
            continue
        parts = w0.split('-')
        for p in parts:
            if p in ORD:
                out.append(ORD[p])
                ordf = True
            elif p in VAL or p in SC or p in ('point', 'and', 'a', 'half', 'quarter', 'dozen'):
                out.append(p)
            elif p in ALIAS:
                if ALIAS[p]:
                    out.append(ALIAS[p])
            else:
                out.append('UNITW')
    for i, t in enumerate(out):
        nxt = out[i + 1] if i + 1 < len(out) else None
        if t == 'a' and nxt not in set(VAL) | set(SC) | {'half', 'quarter', 'dozen'}:
            out[i] = 'UNITW'
        if ordf and t in ('half', 'quarter') and (i > 0) and (out[i - 1] in VAL):
            out[i] = 'UNITW'
    if not any((t in VAL or t in SC or t in ('half', 'dozen') for t in out)):
        return (None, ordf)
    return (out, ordf)

def _compose_int(tok):
    total, cur, seen = (0, 0, False)
    for i, t in enumerate(tok):
        if t == 'and':
            continue
        if t == 'a':
            if i + 1 < len(tok) and tok[i + 1] in ('half', 'quarter'):
                continue
            cur += 1
            seen = True
            continue
        if t == 'quarter':
            return total + cur + 0.25 if seen else 0.25
        if t in VAL:
            v = VAL[t]
            if cur % 100 and v < 100 and (cur % 10 == 0) and (v < 10) and (cur % 100 >= 20):
                cur += v
            elif cur % 100 and v >= 10:
                return None
            elif cur % 10 and v < 10 and (cur >= 1):
                return None
            else:
                cur += v
            seen = True
        elif t in SC:
            s = SC[t]
            if s == 100:
                cur = (cur or 1) * 100
            else:
                total += (cur or 1) * s
                cur = 0
            seen = True
        elif t == 'dozen':
            cur = (cur or 1) * 12
            seen = True
        elif t == 'half':
            return total + cur + 0.5 if seen else 0.5
        else:
            return None
    return total + cur if seen else None

def _groups(tok):
    g, cur = ([], [])
    for t in tok:
        if t == 'OH':
            if cur:
                g.append(cur)
                cur = []
            cur = ['OH']
            continue
        if cur and cur != ['OH'] and (t in TENS or (t in VAL and VAL[t] >= 10) or (cur[-1] in UNITS[1:10] and t in VAL and (cur[-1] not in TENS))):
            g.append(cur)
            cur = []
        cur.append(t)
    if cur:
        g.append(cur)
    out = []
    for x in g:
        if x[0] == 'OH':
            v = _compose_int(x[1:]) if len(x) > 1 else 0
        else:
            v = _compose_int(x)
        if v is None:
            return None
        out.append(int(v))
    return out

def compose(tok, kind=None):
    if not tok:
        return None
    t2 = []
    for t in tok:
        if t == 'UNITW':
            if t2 and t2[-1] not in ('SEP',):
                t2.append('|')
            continue
        t2.append(t)
    while t2 and t2[-1] == '|':
        t2.pop()
    if '|' in t2:
        i = t2.index('|')
        rest = [t for t in t2[i + 1:] if t != '|']
        t2 = t2[:i] if not rest or kind not in ('score', 'ratio_odds', 'clock') else t2[:i] + ['SEP'] + rest
    tok = [t for t in t2 if t != '|']
    if not tok:
        return None
    neg = tok[0] == 'NEG'
    tok = [t for t in tok if t != 'NEG']
    if 'SEP' in tok:
        i = tok.index('SEP')
        a = compose(tok[:i], None)
        b = compose(tok[i + 1:], None)
        if not a or not b:
            return None
        if kind in ('clock', 'time_of_day'):
            return dict(v=a['v'] * 60 + b['v'], v2=None)
        return dict(v=a['v'], v2=b['v'])
    if 'point' in tok:
        i = tok.index('point')
        a = _compose_int(tok[:i]) if i else 0
        d = tok[i + 1:]
        if a is None or not d or any((t not in UNITS[:10] and t != 'OH' for t in d)):
            return None
        frac = ''.join((str(VAL.get(t, 0)) for t in d))
        v = float(f'{int(a)}.{frac}')
        return dict(v=-v if neg else v, v2=None)
    v = _compose_int(tok)
    if kind in ('id_pii', 'ref_number', 'jersey_number') and v is not None:
        return dict(v=v, v2=None)
    if v is None or kind in ('date', 'clock', 'time_of_day', 'field_position', 'score', 'ratio_odds', 'measure'):
        g = _groups(tok)
        if g and len(g) == 2 and (kind in ('score', 'ratio_odds')) and (max(g) < 20) and (v is None):
            return dict(v=g[0], v2=g[1])
        if g and len(g) == 2 and (kind == 'measure') and (g[0] < 8) and (g[1] < 12):
            return dict(v=g[0], v2=g[1])
        if v is not None and kind in ('score', 'ratio_odds', 'measure', 'field_position'):
            return dict(v=v, v2=None)
        if g and len(g) == 2 and (kind in ('clock', 'time_of_day')):
            return dict(v=g[0] * 60 + g[1], v2=None)
        if g and len(g) == 2 and (kind == 'date') and (10 <= g[0] <= 20) and (g[1] < 100):
            return dict(v=g[0] * 100 + g[1], v2=None)
        if g and len(g) >= 2 and (kind in ('id_pii', 'ref_number', 'jersey_number')):
            return dict(v=int(''.join((str(x) for x in g))), v2=None)
        if v is None and g and (len(g) == 2):
            return dict(v=g[0] * 100 + g[1], v2=None)
        if v is None:
            return None
    return dict(v=-v if neg else v, v2=None)

def _test():
    T = [(['twenty', 'five'], None, 25), (['a', 'hundred', 'twenty', 'four', 'point', 'six'], None, 124.6), (['two', 'thousand', 'twenty', 'five'], None, 2025), (['twenty', 'twenty', 'three'], 'date', 2023), (['nine', 'thirty', 'six'], 'clock', 576), (['4:15'], 'clock', 255), (['$80,000'], None, 80000), (['3.7'], None, 3.7), (['hundred', 'and', 'one', 'for', 'six'], 'score', (101, 6)), (['two', 'oh', 'five'], None, 205), (['102-98'], 'score', (102, 98)), (['one', 'twenty'], 'score', 120), (['third'], 'ordinal', 3), (['11th'], 'ordinal', 11), (['seven', 'and', 'a', 'half'], None, 7.5), (['Ninety', 'four,', 'ninety', 'three'], 'score', (94, 93)), (['two', 'one'], 'score', (2, 1)), (['six', 'ten'], 'measure', (6, 10)), (['6671'], 'ref_number', 6671), (['three', 'minutes', 'left'], 'clock', 3), (['first', 'quarter.'], 'ordinal', 1), (['$4', 'a', 'gallon'], 'money', 4), (['dozen'], 'stat', 12), (['one', 'million'], None, 10 ** 6)]
    bad = 0
    for w, k, want in T:
        tok, _ = tokens_of(w)
        r = compose(tok, k)
        got = None if r is None else (r['v'], r['v2']) if r['v2'] is not None else r['v']
        ok = got == want
        bad += not ok
        print('ok ' if ok else 'BAD', w, k, tok, got, want)
    print('NWT size', len(NWT), 'failures', bad)
    return bad
