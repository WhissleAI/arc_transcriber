import re
import numpy as np
from . import numvalue as NV
from . import numlib as NL
from .namesim import norm as _norm
SPORTS = {'nba', 'nhl', 'nfl', 'soccer', 'cricket', 'esports'}
KIND_TAG = {'person': 'entity.mention.person', 'team': 'entity.mention.team', 'organization': 'entity.mention.organization', 'place': 'entity.mention.place', 'work_product': 'entity.mention.work_product', 'event_name': 'entity.mention.event_name', 'legal_ref': 'entity.mention.legal_ref', 'game_entity': 'entity.mention.game_entity', 'game_map': 'entity.mention.game_map'}

def norm(t):
    return _norm(t)

def is_cap(t):
    t = str(t).strip('"\'(')
    return bool(t) and t[0].isupper()

class ExactNames:

    def __init__(self, kb):
        self.kb = list(kb or [])
        full = {}
        for r, c in enumerate(self.kb):
            for f in [c.get('name', '')] + list(c.get('aliases') or []):
                k = norm(f)
                if k:
                    full.setdefault(k, r)
        sur = {}
        for r, c in enumerate(self.kb):
            t = norm(c.get('name', '')).split()
            if len(t) >= 2 and len(t[-1]) >= 3:
                sur.setdefault(t[-1], set()).add(r)
        self.full = full
        self.uniq = {k: next(iter(v)) for k, v in sur.items() if len(v) == 1 and k not in full}
        self.maxn = max([len(k.split()) for k in full] + [1])

    def __bool__(self):
        return bool(self.full)

    def rows_in(self, text):
        s = ' ' + norm(text) + ' '
        return sorted({r for k, r in self.full.items() if f' {k} ' in s})

    def spans(self, texts, used=None):
        toks = [norm(t) for t in texts]
        used = np.zeros(len(texts), bool) if used is None else np.asarray(used, bool).copy()
        out = []
        for n in range(min(4, max(self.maxn, 1)), 0, -1):
            for i in range(len(texts) - n + 1):
                if used[i:i + n].any():
                    continue
                s = ' '.join((t for t in toks[i:i + n] if t))
                if not s or not toks[i] or (not toks[i + n - 1]):
                    continue
                r = self.full.get(s)
                if r is None and n == 1:
                    r = self.uniq.get(s)
                if r is None:
                    continue
                if n == 1 and (not is_cap(texts[i])):
                    continue
                used[i:i + n] = True
                out.append((i, i + n, r))
        return sorted(out)

    def tag_of(self, row):
        return KIND_TAG.get(self.kb[row].get('kind', 'other'), 'entity.mention.person')
NUMTOK = set(NL.VAL) | set(NL.SC) | set(NL.ORD.values()) | {'OH'}
GLUE = ('point', 'and', 'to', '-', 'oh', 'of')

def numtoks(w):
    pairs, _ = NV.word_tokens([w])
    return [t for _, t in pairs]

def has_num(w):
    return any((t in NUMTOK for t in numtoks(w)))

def text_value(kind, words, next_words=(), prev_word='', unit=None):
    words = [str(w) for w in words]
    if not words:
        return (None, None)
    tok = NV.plain(NV.word_tokens(words)[0])
    if not any((t in NUMTOK or t in ('half', 'quarter', 'dozen') for t in tok)):
        return (None, None)
    if unit is None and kind:
        unit = NV.unit_of(kind, words + [str(w) for w in list(next_words)[:1]], prev_word or '')
    try:
        v = NV.finalize(kind, tok, unit, words) if kind else NL.compose([t for t in tok if t != 'UNITW'], None)
    except Exception:
        v = None
    if v is None or v.get('v') is None:
        return (None, unit)
    return (dict(v=v['v'], v2=v.get('v2')), unit)

def number_runs(texts):
    out = []
    i = 0
    while i < len(texts):
        if not has_num(texts[i]):
            i += 1
            continue
        j = i + 1
        while j < len(texts):
            if has_num(texts[j]) and (not texts[j - 1].rstrip().endswith(('.', '?', '!'))) and (not NV.word_tokens([texts[j - 1]])[1]):
                j += 1
                continue
            if j + 1 < len(texts) and NV.lw(texts[j]) in GLUE and has_num(texts[j + 1]) and (not texts[j - 1].rstrip().endswith(('.', '?', '!', ','))):
                j += 2
                continue
            break
        out.append((i, j))
        i = j
    return out

def kind_rule(texts, i, j, domain='unknown'):
    words = texts[i:j]
    nxt = [NV.lw(w) for w in texts[j:j + 2]]
    prv = NV.lw(texts[i - 1]) if i else ''
    raw = ' '.join(words).lower()
    lws = [NV.lw(w) for w in words]
    if '$' in raw or '€' in raw or '£' in raw or any((NV.UWORD.get(x) in ('USD', 'EUR', 'GBP', 'JPY', 'INR') for x in nxt[:1] + lws)):
        return 'money'
    if '%' in raw or (nxt[:1] and nxt[0] in ('percent', 'per')):
        return 'percent'
    if re.search('\\b[ap]\\.?m\\b', ' '.join([raw] + nxt[:1])) or (nxt[:1] and nxt[0] in ("o'clock", 'oclock')) or (re.search('\\d:\\d\\d', raw) and domain not in SPORTS):
        return 'time_of_day'
    u = NV.UWORD.get(nxt[0]) if nxt else None
    if u in NV.DUR_SEC:
        return 'duration'
    if u in ('mile', 'foot', 'inch', 'yard', 'meter', 'kilometer', 'centimeter', 'kilogram', 'pound_wt', 'ounce', 'gallon', 'liter', 'degree', 'mph', 'acre'):
        return 'measure'
    if u in ('points', 'rebounds', 'assists', 'yards', 'runs', 'wickets', 'overs', 'saves', 'shots', 'goals', 'kills', 'steals', 'blocks', 'sacks', 'touchdowns', 'catches', 'threes', 'turnovers', 'fouls', 'boundaries', 'sixes', 'fours', 'hits', 'strikeouts', 'ppg'):
        return 'stat'
    _, ordf = NV.word_tokens(words)
    if ordf:
        return 'ordinal'
    if prv in NV.REFW:
        return 'ref_number'
    if domain in SPORTS and (re.search('\\d+-\\d+', raw) or 'to' in lws):
        return 'score'
    if domain in SPORTS and prv in ('number', 'no.', 'jersey'):
        return 'jersey_number'
    return 'count'

def text_numbers(texts, domain='unknown'):
    out = []
    for i, j in number_runs(texts):
        kind = kind_rule(texts, i, j, domain)
        v, unit = text_value(kind, texts[i:j], texts[j:j + 1], texts[i - 1] if i else '')
        if v is not None:
            out.append(dict(i=i, j=j, kind=kind, value=v['v'], value2=v['v2'], unit=unit))
    return out

def _ov(a, b):
    return min(a['end'], b['end']) > max(a['start'], b['start'])

def fallback_names(primary, fallback):
    out = []
    for x in fallback:
        if x.get('state') != 'linked' or not x.get('entity'):
            continue
        if any((_ov(x, y) for y in primary)) or any((_ov(x, y) for y in out)):
            continue
        out.append(dict(x, src='audio', via='fallback'))
    return out
