import re
_EDGE = re.compile("^[^\\w$£€%#@+]+|[^\\w%']+$")
VIA_RANK = {'exact': 0, None: 1, 'snap': 2, 'fallback': 3}
UNIT_WORDS = {'percent', 'per', 'cent', 'points', 'point', 'pts', 'dollars', 'dollar', 'cents', 'euros', 'euro', 'pounds', 'pound', 'yen', 'bucks', 'k', 'million', 'billion', 'thousand', 'hundred', 'seconds', 'second', 'minutes', 'minute', 'hours', 'hour', 'days', 'day', 'weeks', 'week', 'months', 'month', 'years', 'year', 'yards', 'yard', 'feet', 'foot', 'inches', 'inch', 'miles', 'mile', 'meters', 'metres', 'meter', 'kilometers', 'km', 'kg', 'pounds', 'mph', 'am', 'pm', 'a.m', 'p.m', "o'clock", 'oclock', 'st', 'nd', 'rd', 'th', 'to', 'and', 'of', 'out', 'by', 'a', 'an', 'x', 'times', 'percentage', 'degrees', 'degree', 'goals', 'goal', 'runs', 'run', 'wickets', 'overs', 'rebounds', 'assists', 'shots', 'kills', 'games', 'sets', 'laps'}

def clean_word(t):
    return _EDGE.sub('', str(t or '').strip())

def clean_text(t):
    return ' '.join((w for w in (clean_word(x) for x in str(t or '').split()) if w))

ORDINALS = {'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth', 'twentieth', 'thirtieth', 'fortieth', 'fiftieth', 'sixtieth', 'seventieth', 'eightieth', 'ninetieth', 'hundredth', 'thousandth', 'millionth', 'nil', 'none', 'nothing', 'love', 'twice', 'once'}

def _numeric(w):
    from .words import NUMWORD
    c = clean_word(w).lower()
    if not c:
        return False
    return bool(NUMWORD.match(c)) or any((ch.isdigit() for ch in c)) or c in ORDINALS or bool(re.fullmatch('(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-\\w+', c))

def _forms(entity):
    return {clean_word(x).lower() for x in str(entity or '').split()}

def _capital(w):
    c = clean_word(w)
    return bool(c) and c[0].isupper()

def tidy_name(r):
    W = r.get('words') or []
    if len(W) > 1:
        ent = _forms(r.get('entity')) if r.get('entity') else set()
        caps = [_capital(w['w']) for w in W]

        def drop(k):
            c = clean_word(W[k]['w'])
            if not c:
                return True
            if ent:
                return c.lower() not in ent and any((clean_word(w['w']).lower() in ent for j, w in enumerate(W) if j != k))
            return not caps[k] and any((caps[j] for j in range(len(W)) if j != k))
        while len(W) > 1 and drop(len(W) - 1):
            W = W[:-1]
            caps = caps[:-1]
        while len(W) > 1 and drop(0):
            W = W[1:]
            caps = caps[1:]
        if len(W) != len(r['words']):
            r['words'] = W
            r['start'], r['end'] = (W[0]['start'], W[-1]['end'])
            r['heard'] = ' '.join((w['w'] for w in W))
    r['heard'] = clean_text(r.get('heard', ''))
    return r

def tidy_number(r):
    ws = str(r.get('heard') or '').split()
    num = [_numeric(w) for w in ws]
    if any(num):
        i, j = (num.index(True), len(num) - 1 - num[::-1].index(True))
        while j + 1 < len(ws) and clean_word(ws[j + 1]).lower() in UNIT_WORDS:
            j += 1
        while i > 0 and clean_word(ws[i - 1]).lower() in ('$', '£', '€', 'minus', 'negative', 'number', 'no'):
            i -= 1
        ws = ws[i:j + 1]
    r['heard'] = clean_text(' '.join(ws))
    return r

def keep_number(r):
    return r.get('value') is not None or any((_numeric(w) for w in str(r.get('heard') or '').split()))

def tidy_record(r, kind):
    if kind == 'name':
        return tidy_name(r)
    if kind == 'number':
        return tidy_number(r)
    return r

def same_mention(a, b):
    ov = min(a['end'], b['end']) - max(a['start'], b['start'])
    if ov <= 0:
        return False
    short = max(1e-06, min(a['end'] - a['start'], b['end'] - b['start']))
    if ov / short < 0.5:
        return False
    if a.get('entity') or b.get('entity'):
        return a.get('entity') == b.get('entity')
    return clean_text(a.get('heard', '')).lower() == clean_text(b.get('heard', '')).lower()

def rank(r):
    return (0 if r.get('state') == 'linked' else 1, VIA_RANK.get(r.get('via'), 1), -(r.get('p') or r.get('span_p') or 0.0))

def dedupe_names(names):
    out = []
    for r in sorted(names, key=rank):
        if any((same_mention(r, o) for o in out)):
            continue
        out.append(r)
    return sorted(out, key=lambda r: (r['start'], r['end']))

def group_words(words, spans, source='external'):
    W = sorted([w for w in words if str(w.get('text', w.get('w', ''))).strip()], key=lambda w: (w['start'], w['end']))
    if not W:
        return []
    S = sorted(spans)
    groups = [[] for _ in S]
    rest = []
    for w in W:
        m = (float(w['start']) + float(w['end'])) / 2
        best, bd = (None, None)
        for k, (s, e) in enumerate(S):
            d = 0.0 if s <= m <= e else min(abs(m - s), abs(m - e))
            if bd is None or d < bd:
                best, bd = (k, d)
        if best is None or bd > 1.0:
            rest.append(w)
        else:
            groups[best].append(w)
    out = []
    for g in groups:
        if g:
            out.append(g)
    cur = []
    for w in rest:
        if cur and float(w['start']) - float(cur[-1]['end']) > 0.8:
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    out.sort(key=lambda g: g[0]['start'])
    res = []
    for g in out:
        ws = [dict({k: v for k, v in w.items() if k in ('tag', 'tag_begin')}, text=str(w.get('text', w.get('w', ''))), start=round(float(w['start']), 3), end=round(float(w['end']), 3), is_final=True) for w in g]
        res.append(dict(start=round(ws[0]['start'], 2), end=round(ws[-1]['end'], 2), text=' '.join((w['text'] for w in ws)), words=ws, source=source))
    return res

def utterance_spans(wav, cfg=None, fr=1600):
    import numpy as np
    from .transcribe import UttSegmenter
    seg = UttSegmenter(cfg or {})
    closed = []
    db = lambda a: -120.0 if len(a) == 0 else float(20 * np.log10(np.sqrt(np.mean(a * a)) + 1e-06))
    n = (len(wav) + fr - 1) // fr
    for f in range(n):
        closed += seg.feed(f, db(wav[f * fr:(f + 1) * fr]))
    closed += seg.flush()
    return [(f0 / 10, (f1 + 1) / 10) for f0, f1, _ in closed]
