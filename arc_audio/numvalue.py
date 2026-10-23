import re
from . import numlib as NL
KINDS = ['score', 'clock', 'time_of_day', 'date', 'duration', 'money', 'percent', 'ordinal', 'count', 'stat', 'measure', 'jersey_number', 'field_position', 'ratio_odds', 'ref_number', 'id_pii']
UNITS = ['second', 'minute', 'hour', 'day', 'week', 'month', 'year', 'decade', 'am', 'pm', 'oclock', 'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'INR', 'XXX', 'mile', 'foot', 'inch', 'ft_in', 'yard', 'meter', 'kilometer', 'centimeter', 'kilogram', 'pound_wt', 'ounce', 'gallon', 'liter', 'degree', 'mph', 'kph', 'acre', 'points', 'rebounds', 'assists', 'yards', 'runs', 'wickets', 'overs', 'saves', 'shots', 'goals', 'kills', 'steals', 'blocks', 'sacks', 'touchdowns', 'catches', 'threes', 'turnovers', 'fouls', 'boundaries', 'sixes', 'fours', 'hits', 'strikeouts', 'ppg', 'times', 'people', 'age', 'page', 'exhibit', 'section', 'chapter', 'channel', 'docket', 'rule', 'line', 'flight', 'paragraph', 'article', 'volume', 'question', 'route', 'room', 'case', 'item', 'number', 'other']
UNIT_ID = {u: i for i, u in enumerate(UNITS)}
UWORD = {'second': 'second', 'seconds': 'second', 'sec': 'second', 'secs': 'second', 'minute': 'minute', 'minutes': 'minute', 'mins': 'minute', 'min': 'minute', 'hour': 'hour', 'hours': 'hour', 'hrs': 'hour', 'day': 'day', 'days': 'day', 'week': 'week', 'weeks': 'week', 'month': 'month', 'months': 'month', 'year': 'year', 'years': 'year', 'decade': 'decade', 'decades': 'decade', 'fortnight': 'week', 'am': 'am', 'a.m.': 'am', 'a.m': 'am', 'pm': 'pm', 'p.m.': 'pm', 'p.m': 'pm', "o'clock": 'oclock', 'oclock': 'oclock', 'dollars': 'USD', 'dollar': 'USD', 'bucks': 'USD', '$': 'USD', 'cents': 'USD', 'euros': 'EUR', 'euro': 'EUR', '€': 'EUR', 'pounds': 'GBP', 'quid': 'GBP', '£': 'GBP', 'yen': 'JPY', 'rupees': 'INR', 'miles': 'mile', 'mile': 'mile', 'feet': 'foot', 'foot': 'foot', 'inches': 'inch', 'inch': 'inch', 'yard': 'yard', 'meters': 'meter', 'metres': 'meter', 'meter': 'meter', 'kilometers': 'kilometer', 'kilometres': 'kilometer', 'km': 'kilometer', 'centimeters': 'centimeter', 'kilos': 'kilogram', 'kg': 'kilogram', 'kilograms': 'kilogram', 'ounces': 'ounce', 'gallons': 'gallon', 'gallon': 'gallon', 'liters': 'liter', 'litres': 'liter', 'degrees': 'degree', 'degree': 'degree', 'mph': 'mph', 'acres': 'acre', 'times': 'times', 'people': 'people', 'old': 'age'}
for _s in ('points', 'rebounds', 'assists', 'yards', 'runs', 'wickets', 'overs', 'saves', 'shots', 'goals', 'kills', 'steals', 'blocks', 'sacks', 'touchdowns', 'catches', 'threes', 'turnovers', 'fouls', 'boundaries', 'sixes', 'fours', 'hits', 'strikeouts', 'ppg'):
    UWORD.setdefault(_s, _s)
UWORD.setdefault('run', 'runs')
UWORD.setdefault('point', 'points')
REFW = {'page': 'page', 'pages': 'page', 'exhibit': 'exhibit', 'section': 'section', 'chapter': 'chapter', 'channel': 'channel', 'docket': 'docket', 'rule': 'rule', 'line': 'line', 'flight': 'flight', 'paragraph': 'paragraph', 'article': 'article', 'volume': 'volume', 'question': 'question', 'route': 'route', 'interstate': 'route', 'highway': 'route', 'room': 'room', 'case': 'case', 'item': 'item', 'number': 'number', 'no.': 'number'}
APPROX = {'about', 'around', 'approximately', 'roughly', 'nearly', 'almost', 'some', 'over', 'under', 'like', 'maybe', 'close', 'upwards', 'approx', 'circa', 'estimated'}
DUR_SEC = {'second': 1, 'minute': 60, 'hour': 3600, 'day': 86400, 'week': 7 * 86400, 'month': 30.44 * 86400, 'year': 365.25 * 86400, 'decade': 3652.5 * 86400}

def lw(w):
    return re.sub('^[\\"\'(\\[]+|[\\"\'),.?!;:\\]]+$', '', str(w).lower()).replace('’', "'")

def word_tokens(words):
    out, ordf = ([], False)
    for wi, w in enumerate(words):
        n0 = len(out)
        if out and out[-1][1] != 'SEP' and (wi > 0) and words[wi - 1].rstrip().endswith(',') and (w.strip().lower() not in ('and', 'point')):
            out.append((wi, 'SEP'))
        w0 = w.lower().strip('.,?!;:"\'()').replace('’', "'")
        w0 = re.sub("'s$", '', w0)
        if w0:
            m = re.fullmatch('[$£€#]?(\\d[\\d,]*)(\\.\\d+)?(st|nd|rd|th|s|%|k|m)?', w0)
            m2 = re.fullmatch('(\\d+)[:\\-](\\d+)', w0)
            if m:
                out += [(wi, t) for t in NL.int_words(m.group(1).replace(',', ''))]
                if m.group(2):
                    out += [(wi, 'point')] + [(wi, NL.UNITS[int(c)]) for c in m.group(2)[1:]]
                if m.group(3) in ('st', 'nd', 'rd', 'th'):
                    ordf = True
                if m.group(3) == 'k':
                    out.append((wi, 'thousand'))
                if m.group(3) == 'm':
                    out.append((wi, 'million'))
            elif m2:
                out += [(wi, t) for t in NL.int_words(m2.group(1))] + [(wi, 'SEP')] + [(wi, t) for t in NL.int_words(m2.group(2))]
            else:
                for p in w0.split('-'):
                    if p in NL.ORD:
                        out.append((wi, NL.ORD[p]))
                        ordf = True
                    elif p in NL.VAL or p in NL.SC or p in ('point', 'and', 'a', 'half', 'quarter', 'dozen'):
                        out.append((wi, p))
                    elif p in NL.ALIAS:
                        if NL.ALIAS[p]:
                            out.append((wi, NL.ALIAS[p]))
                    else:
                        out.append((wi, 'UNITW'))
        if len(out) == n0:
            out.append((wi, '<none>'))
    numtok = set(NL.VAL) | set(NL.SC) | {'half', 'quarter', 'dozen', 'OH'}
    for i in range(len(out)):
        w_, t = out[i]
        if t == 'two' and lw(words[w_]) in ('second', 'seconds') and (i > 0) and (out[i - 1][1] in numtok):
            out[i] = (w_, 'UNITW')
    for i in range(len(out)):
        if out[i][1] == 'SEP' and (not any((t in numtok for _, t in out[i + 1:]))):
            out[i] = (out[i][0], 'UNITW')
    toks = [t for _, t in out]
    real = [i for i, t in enumerate(toks) if t != '<none>']
    for k, i in enumerate(real):
        t = toks[i]
        nxt = toks[real[k + 1]] if k + 1 < len(real) else None
        prv = toks[real[k - 1]] if k else None
        if t == 'a' and nxt not in set(NL.VAL) | set(NL.SC) | {'half', 'quarter', 'dozen'}:
            out[i] = (out[i][0], 'UNITW')
        if ordf and t in ('half', 'quarter') and (k > 0) and (prv in NL.VAL):
            out[i] = (out[i][0], 'UNITW')
    return (out, ordf)

def plain(tokpairs):
    return [t for _, t in tokpairs if t != '<none>']

def is_pair(tok, kind):
    t = [x for x in tok if x != 'UNITW']
    if 'SEP' in t:
        return True
    return NL._compose_int([x for x in t if x not in ('NEG',)]) is None

def unit_of(kind, words, prev_word):
    lws = [lw(w) for w in words]
    raw = ' '.join(words).lower().replace('_', '')
    if kind == 'ref_number':
        return REFW.get(lw(prev_word)) or 'number'
    if kind == 'money':
        if '€' in raw or 'euro' in raw:
            return 'EUR'
        if '£' in raw or 'pound' in raw or 'quid' in raw:
            return 'GBP'
        for x in lws:
            if UWORD.get(x) in ('USD', 'JPY', 'INR', 'CAD', 'AUD'):
                return UWORD[x]
        return 'USD' if '$' in raw or 'dollar' in raw or 'buck' in raw else 'XXX'
    if kind == 'time_of_day':
        if re.search('\\bp\\.?\\s?m\\b|\\d\\s?pm\\b', raw):
            return 'pm'
        if re.search('\\ba\\.?\\s?m\\b|\\d\\s?am\\b', raw):
            return 'am'
        return 'oclock' if "o'clock" in raw or 'oclock' in raw else None
    if kind == 'measure' and any((x in ('mile', 'miles') for x in lws)) and any((x in ('hour', 'hr') for x in lws)):
        return 'mph'
    for x in lws:
        u = UWORD.get(x)
        if u:
            if kind in ('duration', 'clock') and u in DUR_SEC:
                return u
            if kind == 'measure' and u in UNIT_ID and (UNITS.index(u) >= UNITS.index('mile')) and (UNITS.index(u) <= UNITS.index('acre')):
                return u
            if kind in ('stat', 'count') and u in UNIT_ID and (UNITS.index(u) >= UNITS.index('points')) and (UNITS.index(u) <= UNITS.index('age')):
                return u
    return None

def finalize(kind, tok, unit, words=None):
    t = [x for x in tok if x != '<none>']
    if not t:
        return None
    r = NL.compose(t, kind)
    if r is None:
        return None
    v, v2 = (r['v'], r['v2'])
    if kind == 'time_of_day':
        tn = [x for x in t if x not in ('UNITW', 'NEG')]
        vi = NL._compose_int(tn) if 'SEP' not in tn else None
        if vi is not None and ('hundred' in tn or 'thousand' in tn):
            if not float(vi).is_integer() or int(vi) % 100 >= 60 or int(vi) // 100 > 24:
                return None
            hh, mm = divmod(int(vi), 100)
        elif vi is not None and vi < 24 and float(vi).is_integer():
            hh, mm = (int(vi), 0)
        elif v is not None and v >= 0 and float(v).is_integer() and (vi is None):
            hh, mm = divmod(int(v), 60)
        else:
            return None
        if unit == 'pm' and hh < 12:
            hh += 12
        if unit == 'am' and hh == 12:
            hh = 0
        if not (0 <= hh <= 24 and 0 <= mm < 60):
            return None
        return dict(v=hh % 24 * 3600 + mm * 60, v2=None)
    if kind == 'duration':
        if unit not in DUR_SEC or v is None or v2 is not None:
            return None
        return dict(v=round(v * DUR_SEC[unit], 3), v2=None)
    if kind == 'clock':
        if v2 is not None:
            return None
        if is_pair(t, kind):
            return dict(v=v, v2=None)
        if unit in ('minute', 'hour'):
            return dict(v=v * DUR_SEC[unit], v2=None)
        return dict(v=v, v2=None)
    if kind == 'id_pii':
        return None
    return dict(v=v, v2=v2)
