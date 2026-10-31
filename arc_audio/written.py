"""Written form of a transcript ("inverse text normalization") from Arc's own number spans.

Arc's decoder writes numbers the way they are spoken ("two thirty", "March third", "forty five dollars") and marks each one with an
inline number tag (NUMBER_TIME, NUMBER_DATE, NUMBER_MONEY, ...). This module turns those spans into their written form ("2:30",
"March 3", "$45") so one field is the text to show, or to score against digit references.

Rules:
- Only number spans are rewritten: the spans Arc tagged, plus untagged runs of number words (read with the same text rules as Arc's
  text layer, `textlayer.kind_rule`).
- Never invent: a span whose value is not certain from its words (for example "two thirty" with no tag, no "at" and no "p.m."; "twelve
  fifty" with no currency) keeps its words.
- Single-digit counts, durations and ordinals stay words ("two kids", "first"), as in common style guides; times, dates, money,
  percents, measures, stats, scores and IDs always use digits. `spell_small=False` writes those as digits too.

    from arc_audio.written import to_written
    to_written("the meeting moved to two thirty p.m. on March third")   # 'the meeting moved to 2:30 p.m. on March 3'
"""
import re
from . import numlib as NL
from . import numvalue as NV
from .tagdec import NUMBER_KIND

SPORTS = {'nba', 'nhl', 'nfl', 'soccer', 'cricket', 'esports'}
MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
MONTH_ID = {m: i + 1 for i, m in enumerate(MONTHS)}
MONTH_ID.update({'sept': 9})
CUR_WORD = {'dollar': 'USD', 'dollars': 'USD', 'buck': 'USD', 'bucks': 'USD', 'euro': 'EUR', 'euros': 'EUR', 'pound': 'GBP',
            'pounds': 'GBP', 'quid': 'GBP', 'yen': 'JPY', 'rupee': 'INR', 'rupees': 'INR'}
CUR_SYM = {'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥', 'INR': '₹'}
CENT_WORD = {'cent', 'cents', 'pence', 'penny'}
AMPM = {'am', 'a.m', 'pm', 'p.m'}
OCLOCK = {"o'clock", 'oclock'}
SCALE_BIG = ('million', 'billion')
NUMW = set(NL.VAL) | set(NL.SC) | set(NL.ORD)
TIME_PREP = {'at', 'by', 'until', 'till', 'til', 'around', 'from', 'before', 'after', 'since'}
SPELL_SMALL = {None, 'count', 'ordinal', 'duration'}
DIGITS_OK = {'id_pii', 'ref_number', 'jersey_number', 'phone'}
_LEAD = re.compile(r'^["\'(\[]+')
_DIGIT = re.compile(r'\d')


def is_month(w):
    """a month name as Arc writes it (capitalized, so 'may' the verb is not a month)"""
    w = lead(w) and str(w)[len(lead(w)):] or str(w)
    return core(w) in MONTH_ID and w[:1].isupper()


def core(w):
    return NV.lw(w)


def trail(w):
    """trailing punctuation of a word, keeping the dot of 'p.m.' / 'a.m.' with the word"""
    w = str(w)
    m = re.search(r'[,;:?!.]+$', w)
    if not m:
        return ''
    t = m.group()
    if re.search(r'\b[ap]\.m\.[,;:?!.]*$', w, re.I):
        t = t[1:]
    return t


def lead(w):
    m = _LEAD.match(str(w))
    return m.group() if m else ''


def is_num_word(c):
    if not c:
        return False
    if re.fullmatch(r'[$£€¥₹#]?\d[\d,]*(\.\d+)?(st|nd|rd|th|s|%|k|m)?', c) or re.fullmatch(r'\d+[:\-]\d+', c):
        return True
    return all((p in NUMW for p in c.split('-')))


def has_digit(ws):
    return any((_DIGIT.search(str(w)) for w in ws))


def num_runs(texts, lo=0, hi=None, extra=()):
    """maximal runs [i, j) of number words: 'and' / 'point' / 'oh' glue inside a run, 'a'/'an' before a scale word, 'and a half';
    a run ends after a word with trailing punctuation"""
    hi = len(texts) if hi is None else hi
    C = [core(t) for t in texts]
    out, i = ([], lo)
    while i < hi:
        c = C[i]
        nxt = C[i + 1] if i + 1 < hi else ''
        if not (is_num_word(c) or c in extra or (c in ('a', 'an') and nxt in NL.SC) or (c in ('minus', 'negative') and is_num_word(nxt))):
            i += 1
            continue
        j = i + 1
        while j < hi and (not trail(texts[j - 1])):
            c = C[j]
            n1 = C[j + 1] if j + 1 < hi else ''
            n2 = C[j + 2] if j + 2 < hi else ''
            if is_num_word(c) or c in extra:
                j += 1
            elif c in ('point', 'oh', 'o') and is_num_word(n1) and (not trail(texts[j])):
                j += 2
            elif c == 'and' and n1 in ('a', 'an') and n2 in ('half', 'quarter') and (not trail(texts[j])) and (not trail(texts[j + 1])):
                j += 3
            elif c == 'and' and is_num_word(n1) and (not trail(texts[j])) and C[j - 1] in ('hundred', 'thousand', 'million', 'billion'):
                j += 2
            else:
                break
        out.append((i, j))
        i = j
    return out


def toks_of(words):
    pairs, ordf = NV.word_tokens([str(w) for w in words])
    return ([t for t in NV.plain(pairs) if t != 'UNITW'], ordf)


def fmt_int(v, kind=None):
    """thousands separators from 1,000; none for IDs / references, or for a year-like number (1900-2099, not a round hundred)
    that is not money / a measure / a stat"""
    v = int(v)
    if abs(v) < 1000 or kind in DIGITS_OK or (kind not in ('money', 'measure', 'stat', 'percent') and 1900 <= v <= 2099 and v % 100):
        return str(v)
    return f'{v:,}'


def fmt_num(v, kind=None):
    if float(v).is_integer():
        return fmt_int(v, kind)
    return f'{v:.6f}'.rstrip('0').rstrip('.')


def well_formed(tok):
    """False for scale words that cannot be one number: 'one hundred and four hundred twenty five' (a second 'hundred' after 104),
    'two thousand three thousand' (a scale repeated or rising). _compose_int multiplies through these (10425, 5000)."""
    cur, last = (0, float('inf'))
    for t in tok:
        if t == 'point':
            break
        if t in NL.VAL:
            cur += NL.VAL[t]
        elif t == 'hundred':
            if cur >= 100:
                return False
            cur = (cur or 1) * 100
        elif t in NL.SC:
            if NL.SC[t] >= last or cur >= 1000:
                return False
            last, cur = (NL.SC[t], 0)
    return True


def value_str(tok, kind=None):
    """written number for a token list (numlib tokens), or None when the reading is not certain"""
    tok = [t for t in tok if t != 'UNITW']
    if not well_formed(tok):
        return None
    neg = bool(tok) and tok[0] == 'NEG'
    tok = [t for t in tok if t != 'NEG']
    if not tok or not any((t in NL.VAL or t in NL.SC for t in tok)) or 'SEP' in tok:
        return None
    if tok[0] in NL.SC:
        return None                                     # 'million' alone: its number was not heard
    if tok[-1] in SCALE_BIG and (not any((t in SCALE_BIG for t in tok[:-1]))) and len(tok) > 1:
        m = value_str(tok[:-1], kind)
        if m is not None and re.fullmatch(r'\d+(\.\d+)?', m.replace(',', '')) and float(m.replace(',', '')) < 1000:
            return ('-' if neg else '') + m + ' ' + tok[-1]
    if 'point' in tok:
        i = tok.index('point')
        a = NL._compose_int(tok[:i]) if i else 0
        d = tok[i + 1:]
        if a is None or not float(a).is_integer() or (not d) or any((t not in NL.UNITS[:10] and t != 'OH' for t in d)):
            return None
        s = f"{fmt_int(a, kind)}.{''.join((str(NL.VAL.get(t, 0)) for t in d))}"
    else:
        v = NL._compose_int(tok)
        if v is None:
            return None
        s = fmt_num(v, kind)
    return ('-' if neg else '') + s


def digit_string(tok, phone=False):
    """digits as read one group at a time ('two oh five' -> '205', 'four twelve' -> '412', 'double five' handled by caller);
    phone=True reads 'eight hundred' as 800 even when digits follow"""
    tok = [t for t in tok if t not in ('UNITW', 'and')]
    if not tok or any((t not in NL.VAL and t not in NL.SC and t != 'OH' for t in tok)):
        return None
    groups, cur = ([], [])
    for t in tok:
        if t == 'OH':
            if cur:
                groups.append(cur)
            groups.append(['OH'])
            cur = []
            continue
        if cur and (t in NL.TENS or (t in NL.VAL and NL.VAL[t] >= 10) or (cur[-1] in NL.UNITS[0:10] and t in NL.VAL and cur[-1] not in NL.TENS) or (phone and cur[-1] in NL.SC and t in NL.VAL)):
            groups.append(cur)
            cur = []
        cur.append(t)
    if cur:
        groups.append(cur)
    out = ''
    for g in groups:
        if g == ['OH']:
            out += '0'
            continue
        v = NL._compose_int(g)
        if v is None or not float(v).is_integer():
            return None
        out += str(int(v))
    return out


def ord_suffix(n):
    n = int(n)
    if 10 <= n % 100 <= 20:
        return 'th'
    return {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')


def _wrap(ws, s):
    return lead(ws[0]) + s + trail(ws[-1]) if s is not None else None


def render_run(ws, kind=None, spell_small=True):
    """one run of number words -> written string (with the run's leading / trailing punctuation), or None to keep the words"""
    if has_digit(ws) and all((has_digit([w]) or core(w) in ('and', 'point') for w in ws)):
        return None
    tok, ordf = toks_of(ws)
    last_ord = ordf and core(ws[-1]).split('-')[-1] in NL.ORD
    if ordf and (not last_ord):
        return None
    s = value_str(tok, kind)
    if s is None and kind in DIGITS_OK:
        s = digit_string(tok)
    if s is None and 'and' in [core(w) for w in ws]:
        k = [core(w) for w in ws].index('and')
        a, b = (render_run(ws[:k], kind, False), render_run(ws[k + 1:], kind, False))
        if a and b:
            return lead(ws[0]) + a.strip('",;:?!.') + ' ' + ws[k] + ' ' + b
        return None
    if s is None:
        return None
    if last_ord:
        if not re.fullmatch(r'\d+', s.replace(',', '')):
            return None
        if spell_small and int(s.replace(',', '')) < 10 and kind in SPELL_SMALL:
            return None
        s = s + ord_suffix(int(s.replace(',', '')))
    elif spell_small and kind in SPELL_SMALL and len(ws) == 1 and re.fullmatch(r'\d', s):
        return None
    return _wrap(ws, s)


def render_generic(ws, kind=None, spell_small=True):
    out, k, changed = ([], 0, False)
    for i, j in num_runs(ws, extra=('oh',) if kind in DIGITS_OK else ()):
        out += ws[k:i]
        r = render_run(ws[i:j], kind, spell_small)
        if r is None:
            out += ws[i:j]
        else:
            out.append(r)
            changed = True
        k = j
    out += ws[k:]
    return ' '.join(out) if changed else None


def hour_minute(tok, min_hundred_hour=1):
    """(hour, minute) from a spoken clock time: 'two thirty' (2, 30), 'nine oh five' (9, 5), 'seven' (7, 0), 'fourteen hundred' (14, 0).
    min_hundred_hour: the smallest hour accepted before 'hundred' ('six hundred forty five' is 06:45 only when Arc tagged a time)"""
    tok = [t for t in tok if t not in ('UNITW', 'and')]
    if not tok or 'point' in tok or 'NEG' in tok or 'SEP' in tok:
        return None
    if any((t in tok for t in ('thousand',) + SCALE_BIG)) or not well_formed(tok):
        return None                                     # 'one thousand four hundred twenty five' is a number, never a clock time
    v = NL._compose_int(tok) if 'OH' not in tok else None
    if v is not None and float(v).is_integer():
        v = int(v)
        if 'hundred' in tok and v >= 100:
            # 24-hour 'fourteen hundred', 'nineteen hundred thirty'. Without a time tag the hour before 'hundred' must be 10-24:
            # 'at one hundred twenty five' / 'around four hundred thirty' are numbers (they read as 01:25 / 04:30 otherwise)
            hh = NL._compose_int(tok[:tok.index('hundred')])
            if hh is None or not min_hundred_hour <= hh <= 24:
                return None
            h, m = divmod(v, 100)
        elif v <= 24 and len(tok) == 1 or (v <= 12 and len(tok) <= 2):
            h, m = (v, 0)
        else:
            return None
    else:
        g = NL._groups(tok)
        if not g or len(g) != 2:
            return None
        h, m = g
        if not ('OH' in tok or m >= 10):
            return None
    if not (0 <= h <= 24 and 0 <= m <= 59):
        return None
    return (h, m)


def render_time(ws, nxt=(), require_context=False, prev=''):
    """time-of-day span -> '2:30', '2:30 p.m.', '2 p.m.', "5 o'clock"; None when not a clean time"""
    C = [core(w) for w in ws]
    if has_digit(ws) and any((':' in w for w in ws)):
        return None
    for pat, mm, dh in ((('half', 'past'), 30, 0), (('quarter', 'past'), 15, 0), (('quarter', 'after'), 15, 0), (('quarter', 'to'), 45, -1)):
        if len(C) >= 3 and tuple(C[:2]) == pat:
            hm = hour_minute(toks_of(ws[2:3])[0])
            if hm is None or hm[1] or not 1 <= hm[0] <= 12:
                return None
            h = (hm[0] + dh - 1) % 12 + 1
            rest = ' '.join(ws[3:])
            return lead(ws[0]) + f'{h}:{mm:02d}' + (trail(ws[2]) if not rest else '') + (' ' + rest if rest else '')
    runs = num_runs(ws, extra=())
    if len(runs) != 1:
        return None
    i, j = runs[0]
    if ws[:i] and any((c not in ('at', 'by', 'around') for c in C[:i])):
        return None
    suffix = ws[j:]
    sc = [c for c in C[j:]]
    if any((c not in AMPM and c not in OCLOCK and c not in ('in', 'the', 'morning', 'afternoon', 'evening', 'night', 'tonight', 'hours') for c in sc)):
        return None
    tok, ordf = toks_of(ws[i:j])
    if ordf:
        return None
    hm = hour_minute(tok, 10 if require_context else 1)
    if hm is None:
        return None
    h, m = hm
    ampm = any((c in AMPM for c in sc)) or (bool(nxt) and core(nxt[0]) in AMPM)
    ocl = any((c in OCLOCK for c in sc)) or (bool(nxt) and core(nxt[0]) in OCLOCK)
    if require_context and (not (ampm or ocl or (core(prev) in TIME_PREP and m))):
        return None
    if h >= 20 and not (ampm or ocl or 'hundred' in C[i:j]):
        return None                                     # 'twenty eighteen' is a year far more often than 20:18, even with a time tag
    if ampm and not 1 <= h <= 12:
        return None
    if 'hundred' in C[i:j]:
        s = f'{h:02d}:{m:02d}'
    elif m or not (ampm or ocl or j - i == 1):
        s = f'{h}:{m:02d}'
    else:
        s = str(h)
    head = ' '.join(ws[:i] + [lead(ws[i]) + s + (trail(ws[j - 1]) if not suffix else '')])
    return head + (' ' + ' '.join(suffix) if suffix else '')


def parse_date(ws):
    """dict(year, month, day) from a spoken date ('March third', 'the third of March', 'March third, twenty twenty four',
    'twenty twenty four', 'March 3'); None when the words are not a clean date"""
    C = [core(w) for w in ws]
    month = [k for k, c in enumerate(C) if c in MONTH_ID]
    if len(month) > 1:
        return None
    d = dict(year=None, month=MONTH_ID[C[month[0]]] if month else None, day=None)
    runs = num_runs(ws)
    if not runs or len(runs) > 2:
        return None
    for i, j in runs:
        tok, ordf = toks_of(ws[i:j])
        if ordf and d['day'] is None:
            v = NL._compose_int(tok)
            if v is None or not 1 <= v <= 31:
                return None
            d['day'] = int(v)
            continue
        v = NL._compose_int(tok)
        if v is not None and float(v).is_integer() and 1 <= v <= 31 and d['day'] is None and month and (j - i) <= 2 and not ('thousand' in tok or 'hundred' in tok):
            d['day'] = int(v)
            continue
        r = NL.compose(tok, 'date')
        y = None if r is None or r.get('v2') is not None else r['v']
        if y is None or not float(y).is_integer() or not 1000 <= y <= 2999 or d['year'] is not None:
            return None
        d['year'] = int(y)
    other = [c for k, c in enumerate(C) if k not in month and (not any((i <= k < j for i, j in runs)))]
    if any((c not in ('the', 'of', 'on', 'in', '') for c in other)):
        return None
    if d['day'] is None and d['year'] is None:
        return None
    if d['day'] is not None and d['month'] is not None:
        import calendar
        if d['day'] > calendar.monthrange(d['year'] or 2000, d['month'])[1]:
            return None
    return d


def date_iso(d):
    """ISO 8601 text: '2024-03-03', '--03-03' (month and day), '2024-03', '2024'; None for a day alone"""
    if d is None:
        return None
    y, m, dd = (d.get('year'), d.get('month'), d.get('day'))
    if y and m and dd:
        return f'{y:04d}-{m:02d}-{dd:02d}'
    if m and dd:
        return f'--{m:02d}-{dd:02d}'
    if y and m:
        return f'{y:04d}-{m:02d}'
    if y and not dd:
        return f'{y:04d}'
    return None


def render_date(ws):
    d = parse_date(ws)
    if d is None or has_digit(ws) and not any((core(w) in NUMW for w in ws)):
        return None
    C = [core(w) for w in ws]
    runs = num_runs(ws)
    mk = [k for k, c in enumerate(C) if c in MONTH_ID]
    tr = trail(ws[-1])
    if mk and d['day'] is not None:
        Mw = ws[mk[0]].rstrip(',.;:?!')
        day_run = next((r for r in runs if r[0] < r[1] and toks_of(ws[r[0]:r[1]])[1] or (r[1] - r[0] <= 2 and NL._compose_int(toks_of(ws[r[0]:r[1]])[0]) == d['day'])))
        if day_run[0] > mk[0]:
            s = f"{Mw} {d['day']}" + (f", {d['year']}" if d['year'] else '')
            pre = ws[:mk[0]]
            return ' '.join(pre + [lead(ws[mk[0]]) + s + tr]) if not (pre and lead(ws[0])) else None
        pre = ws[:day_run[0]]
        mid = ws[day_run[1]:mk[0]]
        s = f"{d['day']}{ord_suffix(d['day'])}"
        tail = ' '.join(mid + [Mw]) + (f", {d['year']}" if d['year'] else '')
        return ' '.join(pre + [s, tail]) + tr
    if d['day'] is not None:
        out = render_generic(ws, 'date', spell_small=False)
        return out
    if d['year'] is not None:
        i, j = runs[0]
        s = str(d['year'])
        return ' '.join(ws[:i] + [lead(ws[i]) + s + (trail(ws[j - 1]) if j == len(ws) else '')] + ws[j:])
    return None


def money_parts(ws):
    C = [core(w) for w in ws]
    cur = [k for k, c in enumerate(C) if c in CUR_WORD]
    cents = [k for k, c in enumerate(C) if c in CENT_WORD]
    return (C, cur, cents)


def render_money(ws):
    """'forty five dollars' -> '$45'; 'twelve euros fifty' -> '€12.50'; 'forty five dollars and fifty cents' -> '$45.50';
    'two million dollars' -> '$2 million'; 'fifty cents' -> '50 cents'"""
    if has_digit(ws) and all((has_digit([w]) or core(w) in CUR_WORD or core(w) in CENT_WORD for w in ws)):
        return None
    C, cur, cents = money_parts(ws)
    if not cur:
        if cents and len(cents) == 1 and cents[0] == len(ws) - 1:
            return render_generic(ws, 'money', spell_small=False)
        if not cents and not any((ch in ''.join(ws) for ch in '$€£¥₹')):
            return render_generic(ws, 'count')       # tagged money with no currency said: the number, without a unit
        return None
    if len(cur) > 1:
        return None
    k = cur[0]
    unit = CUR_WORD[C[k]]
    sym = CUR_SYM.get(unit)
    runs = num_runs(ws)
    before = [r for r in runs if r[1] <= k]
    after = [r for r in runs if r[0] > k]
    if len(before) != 1 or before[0][1] != k or len(after) > 1 or sym is None:
        return None
    i, j = before[0]
    if i > 0 and any((c not in ('a', 'about', 'around', 'over', 'under', 'just', 'only', 'nearly', 'almost') for c in C[:i])):
        return None
    tok, ordf = toks_of(ws[i:j])
    if ordf:
        return None
    a = value_str(tok, 'money')
    if a is None:
        return None
    frac = ''
    end = k + 1
    if after:
        p, q = after[0]
        mid = C[k + 1:p]
        tail = C[q:]
        if any((c != 'and' for c in mid)) or any((c not in CENT_WORD for c in tail)) or (mid and not tail):
            return None
        cv = value_str(toks_of(ws[p:q])[0])
        if cv is None or not re.fullmatch(r'\d{1,2}', cv) or ' ' in a or '.' in a:
            return None
        frac = f'.{int(cv):02d}'
        end = q + len(tail)
    elif any((c not in CENT_WORD for c in C[k + 1:])) or C[k + 1:]:
        return None
    s = '-' + sym + a[1:] if a.startswith('-') else sym + a + frac
    return ' '.join(ws[:i] + [lead(ws[i]) + s + trail(ws[end - 1])])


def render_percent(ws):
    C = [core(w) for w in ws]
    if 'percent' in C:
        p = C.index('percent')
        n = 1
    elif 'per' in C and C.index('per') + 1 < len(C) and C[C.index('per') + 1] == 'cent':
        p = C.index('per')
        n = 2
    else:
        return render_generic(ws, 'percent', spell_small=False)
    runs = num_runs(ws[:p])
    if len(runs) != 1 or runs[0][1] != p or C[p + n:]:
        return None
    i, j = runs[0]
    tok, ordf = toks_of(ws[i:j])
    s = None if ordf else value_str(tok, 'percent')
    if s is None:
        return None
    return ' '.join(ws[:i] + [lead(ws[i]) + s + '%' + trail(ws[p + n - 1])])


def render_score(ws, domain):
    C = [core(w) for w in ws]
    runs = num_runs(ws)
    if len(runs) == 2:
        (a0, a1), (b0, b1) = runs
        if a0 != 0 or b1 != len(ws) or not (C[a1:b0] in (['to'], ['-'], []) and (b0 - a1) <= 1):
            return None
        a, b = (value_str(toks_of(ws[a0:a1])[0], 'score'), value_str(toks_of(ws[b0:b1])[0], 'score'))
        if a is None or b is None or ordf_any(ws):
            return None
        return lead(ws[0]) + f'{a}–{b}' + trail(ws[-1])
    if len(runs) == 1 and runs[0] == (0, len(ws)):
        tok, ordf = toks_of(ws)
        if ordf:
            return None
        r = NL.compose(tok, 'score')
        if r is not None and r.get('v2') is not None and NL._compose_int(tok) is None:
            return lead(ws[0]) + f"{fmt_num(r['v'])}–{fmt_num(r['v2'])}" + trail(ws[-1])
    return None


def ordf_any(ws):
    return toks_of(ws)[1]


def render_ids(ws, kind):
    """phone numbers / IDs / references: digits as read ('five five five one two three four' -> '555-1234')"""
    C = [core(w) for w in ws]
    out, k, changed = ([], 0, False)
    runs = num_runs(ws, extra=('oh', 'o', 'double', 'triple', 'zero'))
    for i, j in runs:
        out += ws[k:i]
        rw = ws[i:j]
        digits = ''
        ok = True
        rep = 1
        group = []

        def flush():
            nonlocal digits, ok
            if group:
                s = digit_string(toks_of(group)[0], phone=kind == 'phone')
                if s is None:
                    ok = False
                else:
                    digits += s
                group.clear()
        for w in rw:
            c = core(w)
            if c in ('double', 'triple'):
                flush()
                rep = 2 if c == 'double' else 3
                continue
            if rep > 1:
                flush()
                s = digit_string(toks_of([w])[0])
                if s is None or len(s) != 1:
                    ok = False
                else:
                    digits += s * rep
                rep = 1
                continue
            group.append(w)
        flush()
        if not ok or not digits or toks_of(rw)[1] or (has_digit(rw) and not any((core(w) in NUMW for w in rw))):
            out += rw
        else:
            if kind == 'phone':
                if len(digits) == 10:
                    digits = f'{digits[:3]}-{digits[3:6]}-{digits[6:]}'
                elif len(digits) == 7:
                    digits = f'{digits[:3]}-{digits[3:]}'
                elif len(digits) == 11 and digits[0] == '1':
                    digits = f'1-{digits[1:4]}-{digits[4:7]}-{digits[7:]}'
            out.append(lead(rw[0]) + digits + trail(rw[-1]))
            changed = True
        k = j
    out += ws[k:]
    return ' '.join(out) if changed else None


def render_span(ws, kind, domain='unknown', prev='', nxt=(), spell_small=True):
    """written text for one number span (list of word texts) of a kind (numvalue KINDS, None for an untyped NUMBER, 'phone');
    None to keep the words"""
    ws = [str(w) for w in ws if str(w).strip()]
    if not ws:
        return None
    try:
        if kind == 'time_of_day':
            return render_time(ws, nxt) or render_generic(ws, kind, spell_small=False)
        if kind == 'date':
            return render_date(ws)
        if kind == 'money':
            return render_money(ws)
        if kind == 'percent':
            return render_percent(ws)
        if kind == 'score':
            if domain in SPORTS:
                return render_score(ws, domain)
            return render_generic(ws, kind, spell_small=False) if all((r[1] - r[0] == 1 or NL._compose_int(toks_of(ws[r[0]:r[1]])[0]) is not None for r in num_runs(ws))) else None
        if kind in ('id_pii', 'phone'):
            return render_ids(ws, kind)
        if kind == 'clock':
            C = [core(w) for w in ws]
            if not any((NV.UWORD.get(c) in NV.DUR_SEC for c in C)):
                runs = num_runs(ws)
                if len(runs) == 1 and runs[0][0] == 0:
                    i, j = runs[0]
                    tok, ordf = toks_of(ws[i:j])
                    g = NL._groups(tok) if NL._compose_int(tok) is None else None
                    if g and len(g) == 2 and 0 <= g[1] <= 59 and not ordf:
                        return ' '.join([lead(ws[0]) + f'{g[0]}:{g[1]:02d}' + trail(ws[j - 1])] + ws[j:])
            return render_generic(ws, kind, spell_small=False)
        if kind in (None, 'count'):
            t = render_time(ws, nxt, require_context=True, prev=prev)
            if t is not None:
                return t
        return render_generic(ws, kind, spell_small=spell_small and kind in SPELL_SMALL)
    except Exception:
        return None


EXTEND_NEXT = {'money': lambda c: c in CUR_WORD or c in CENT_WORD, 'percent': lambda c: c == 'percent',
               'time_of_day': lambda c: c in AMPM or c in OCLOCK}


def spans_of(words):
    """Arc's number spans in a word list: [(i, j, kind)] from the words' inline tags (tag / tag_begin)"""
    out = []
    i = 0
    n = len(words)
    while i < n:
        t = words[i].get('tag')
        if not t or not (t in NUMBER_KIND or t == 'CONTACT_PHONE'):
            i += 1
            continue
        j = i + 1
        while j < n and words[j].get('tag') == t and (not words[j].get('tag_begin')):
            j += 1
        out.append((i, j, 'phone' if t == 'CONTACT_PHONE' else NUMBER_KIND[t]))
        i = j
    return out


def _extend(texts, i, j, kind, used):
    C = [core(t) for t in texts]
    if kind in EXTEND_NEXT and j < len(texts) and (not used[j]) and EXTEND_NEXT[kind](C[j]) and (not trail(texts[j - 1])):
        j += 1
    if kind == 'money' and C[j - 1] in CUR_WORD and j < len(texts) and (not trail(texts[j - 1])):
        a = j + 1 if C[j] == 'and' else j
        r = num_runs(texts, a, len(texts))
        if r and r[0][0] == a:
            q = r[0][1]
            has_cents = q < len(texts) and C[q] in CENT_WORD and (not trail(texts[q - 1]))
            if has_cents or (a == j and (q == len(texts) or trail(texts[q - 1]))):
                q += 1 if has_cents else 0
                if not any(used[j:q]):
                    j = q
    if kind == 'percent' and j + 1 < len(texts) and C[j] == 'per' and C[j + 1] == 'cent' and (not used[j]) and (not used[j + 1]):
        j += 2
    if kind == 'date':
        if i > 0 and is_month(texts[i - 1]) and (not used[i - 1]) and (not trail(texts[i - 1])):
            i -= 1
        if j + 1 < len(texts) and C[j] == 'of' and is_month(texts[j + 1]) and (not used[j]) and (not used[j + 1]) and (not trail(texts[j - 1])):
            j += 2
        if j < len(texts) and trail(texts[j - 1]) == ',' and any((c in MONTH_ID for c in C[i:j])):
            r = num_runs(texts, j, len(texts))
            if r and r[0][0] == j and (not any(used[j:r[0][1]])):
                y = NL.compose(toks_of(texts[j:r[0][1]])[0], 'date')
                if y is not None and y.get('v2') is None and 1000 <= (y['v'] or 0) <= 2999 and NL._compose_int(toks_of(texts[j:r[0][1]])[0]) is None or (y and 1000 <= (y['v'] or 0) <= 2999 and 'thousand' in toks_of(texts[j:r[0][1]])[0]):
                    j = r[0][1]
    return (i, j)


def written_words(words, domain='unknown', spell_small=True, untagged=True):
    """written form of one utterance's words (dicts with 'text' and optional start / end / tag / tag_begin).
    Returns (text, words): a rewritten span becomes one word {text, start, end, written_from, kind}."""
    W = [dict(w) for w in words if str(w.get('text', '')).strip()]
    texts = [str(w['text']) for w in W]
    used = [False] * len(W)
    spans = []
    tagged = spans_of(W)
    C = [core(t) for t in texts]
    t = 0
    while t < len(tagged):
        i, j, kind = tagged[t]
        t += 1
        while i > 0 and (not used[i - 1]) and is_num_word(C[i - 1]) and (not trail(texts[i - 1])):
            i -= 1                                      # number words just outside the tag: Arc's span edge, not a new number
        while j < len(W) and (not trail(texts[j - 1])):
            if is_num_word(C[j]):
                j += 1
            elif C[j] in ('and', 'point') and j + 1 < len(W) and is_num_word(C[j + 1]) and (C[j] == 'point' or C[j - 1] in NL.SC):
                j += 2
            else:
                break
        i, j = _extend(texts, i, j, kind, used)
        while t < len(tagged) and tagged[t][0] < j:     # the extension reached the next tagged span
            if tagged[t][2] == kind or tagged[t][2] is None:
                j = max(j, tagged[t][1])
                t += 1
            else:
                j = tagged[t][0]
                break
        if kind is None:
            from .textlayer import kind_rule
            k2 = kind_rule(texts, i, j, domain)
            kind = k2 if k2 in ('money', 'percent', 'time_of_day', 'ordinal') else None
        for k in range(i, j):
            used[k] = True
        spans.append((i, j, kind))
    if untagged:
        from .textlayer import kind_rule
        k = 0
        while k < len(W):
            if used[k]:
                k += 1
                continue
            e = k
            while e < len(W) and (not used[e]):
                e += 1
            for i, j in num_runs(texts, k, e):
                if any(used[i:j]):
                    continue
                kind = kind_rule(texts, i, j, domain)
                if i > 0 and is_month(texts[i - 1]) or (j + 1 < len(texts) and core(texts[j]) == 'of' and is_month(texts[j + 1])):
                    kind = 'date'
                i2, j2 = _extend(texts, i, j, kind, used)
                if any((used[x] for x in range(i2, j2))):
                    i2, j2 = (i, j)
                for x in range(i2, j2):
                    used[x] = True
                spans.append((i2, j2, kind))
            k = e
    spans.sort()
    out, k = ([], 0)
    for i, j, kind in spans:
        out += W[k:i]
        s = render_span(texts[i:j], kind, domain, prev=texts[i - 1] if i else '', nxt=texts[j:j + 1], spell_small=spell_small)
        if s is None or s == ' '.join(texts[i:j]):
            out += W[i:j]
        else:
            w = dict(text=s, written_from=' '.join(texts[i:j]), kind=kind)
            if 'start' in W[i]:
                w.update(start=W[i]['start'], end=W[j - 1]['end'])
            out.append(w)
        k = j
    out += W[k:]
    return (' '.join((w['text'] for w in out)), out)


def to_written(text, domain='unknown', spell_small=True):
    """written form of plain text (no tags: number runs and kinds from the text rules)"""
    return written_words([dict(text=t) for t in str(text).split()], domain, spell_small)[0]


def _sim(a, b):
    import difflib
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def spell_names(words, names):
    """linked names in the list's spelling, word for word: the words Arc heard are replaced by the matching words of the entity
    ('Brant' linked to 'Theo Brandt' -> 'Brandt'). Unlike transcript_rewritten, words that were not said are not added."""
    W = [dict(w) for w in words]
    for n in names or []:
        if n.get('state') != 'linked' or not n.get('entity') or not n.get('words'):
            continue
        idx = [k for k, w in enumerate(W) if any((abs(w.get('start', -1) - x['start']) < 0.001 and w['text'] == x['w'] for x in n['words']))]
        if not idx or idx != list(range(idx[0], idx[-1] + 1)) or any((is_num_word(core(W[k]['text'])) for k in idx)):
            continue
        ent = str(n['entity']).split()
        if len(idx) > len(ent):
            continue
        if len(idx) == len(ent):
            pick = list(range(len(ent)))
        else:
            pick, lo = ([], 0)
            for k in idx:
                c = core(W[k]['text']).replace("'s", '')
                best = max(range(lo, len(ent) - (len(idx) - len(pick) - 1)), key=lambda e: _sim(c, ent[e]))
                pick.append(best)
                lo = best + 1
        for k, e in zip(idx, pick):
            t = W[k]['text']
            poss = "'s" if re.search(r"['’]s[,.;:?!]*$", t) and (not ent[e].endswith("'s")) else ''
            new = lead(t) + ent[e] + poss + trail(t)
            if new != t:
                W[k]['written_from'] = t
                W[k]['text'] = new
    return W


def written_transcript(transcript, names=None, domain='unknown', spell_small=True):
    """transcript_written: per utterance, linked names in the list's spelling (spell_names) and numbers in written form
    (written_words): the text to show, or to score against written references."""
    out = []
    for u in transcript or []:
        ws = u.get('words')
        if ws and names:
            ws = spell_names(ws, names)
        if ws:
            text, words = written_words(ws, domain, spell_small)
        else:
            text, words = (to_written(u.get('text', ''), domain, spell_small), [])
        r = {k: u[k] for k in ('start', 'end', 'source', 'speaker', 'device_directed') if k in u}
        r.update(text=text, words=words)
        out.append(r)
    return out
