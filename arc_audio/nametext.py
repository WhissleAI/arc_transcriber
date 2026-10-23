import re
CONNECT = {'of', 'de', 'la', 'le', 'da', 'del', 'della', 'van', 'von', 'der', 'den', 'di', 'du', 'st', 'st.', 'and', '&'}
COMMON = set("a about above after again against all almost along already also although always am among an and another any anyone anything\nare around as ask asked at away back bad be became because become been before began behind being below best better between big\nboth bring brought but by call called came can cannot can't case change come comes coming could couldn't course day days did didn't\ndifferent do does doesn't doing done don't down during each early either else end enough even ever every everyone everything\nfact far feel felt few find first five for found four from full gave get gets getting give given go goes going gone good got great\nhad half hand happen happened hard has hasn't have haven't having he head hear heard help her here hers herself high him himself his\nhit hold home how however i if i'm in inside instead into is isn't it its it's itself just keep kind knew know known last late later\nleast left less let let's life like line little long look looked looking lot lots made make makes making man many may maybe me mean\nmight mind minute minutes more most move much must my myself name need needs never new next nice night no none nor not nothing now\nof off often oh ok okay old on once one only open or other others our out over own part people perhaps place play played player\nplayers playing point points put quite rather really right run said same saw say says second see seem seemed seen set several shall\nshe she's shot should show side since six small so some someone something sometimes soon sort start started still stop such sure\ntake taken talk talking team ten than thank thanks that that's the their them themselves then there there's these they they're\nthing things think thinking third this those though thought three through time times to today together told too took top toward\ntry trying turn two under until up upon us use used very want wanted was wasn't watch way we week well went were weren't what\nwhat's when where whether which while who whole whose why will win with within without won't word work world would wouldn't yeah\nyear years yes yet you you're your yours game games ball goal goals season night tonight left right shot shots score scored win\nlead quarter period half inning over overs run runs wicket yards yard down downs pass foul free kick corner timeout timeouts\ncoach coaches fans crowd floor court ice field pitch net basket layup dunk three-pointer rebound assist block steal save".split())
TAIL_PUNCT = '.,?!;:"\')]'

def norm(t):
    return re.sub("[^\\w'&.-]", '', t).strip(".-'").replace("'s", '') if t else ''

def low(t):
    return re.sub("[^a-z0-9']", '', t.lower()).replace("'s", '')

class ListIndex:

    def __init__(self, kb):
        self.toks = set()
        for c in kb or []:
            for s in [c.get('name', '')] + list(c.get('aliases') or []):
                for t in str(s).split():
                    t = low(t)
                    if len(t) >= 2:
                        self.toks.add(t)
        self.toks = sorted(self.toks)
        from .rescorer import _metaphone
        self._meta = _metaphone
        self.mp = {_metaphone(t) for t in self.toks if len(t) >= 4}
        self.by_len = {}
        for t in self.toks:
            self.by_len.setdefault(len(t), []).append(t)

    def match(self, t, sim=0.85):
        t = low(t)
        if len(t) < 2 or not self.toks:
            return False
        if t in self.toks:
            return True
        from .rescorer import ratio
        n = len(t)
        for L in range(max(2, int(n * sim / (2 - sim))), int(n * (2 - sim) / sim) + 1):
            for c in self.by_len.get(L, ()):
                if ratio(t, c) >= sim:
                    return True
        return bool(self._meta and len(t) >= 4 and (self._meta(t) in self.mp))

def snap(words, rows_p, f_lo, thr, frac=0.6):
    keep = []
    for k, w in enumerate(words):
        fs = range(w['f0'], w['f1'] + 1)
        on = sum((1 for f in fs if 0 <= f - f_lo < len(rows_p) and rows_p[f - f_lo] >= thr))
        if on >= frac * len(fs):
            keep.append(k)
    return keep

def gate(words, idx, past, lst, sim=0.85, stop=(), is_num=None):
    if not idx:
        return []
    cased = any((any((ch.isupper() for ch in w['w'])) for w in list(past[-400:]) + [words[k] for k in idx]))
    cap_seen = set()
    for j, w in enumerate(past):
        if j > 0 and (not re.search('[.?!]$', past[j - 1]['w'])) and w['w'][:1].isupper():
            cap_seen.add(low(w['w']))

    def initial(k):
        if k == 0:
            prev = past[-1] if past else None
        else:
            prev = words[k - 1]
        return prev is None or bool(re.search('[.?!]$', prev['w'])) or words[k]['start'] - prev['end'] > 1.0

    def name_like(k):
        w = words[k]['w']
        t = low(w)
        if not t or t in stop or (is_num and is_num(w)) or t.isdigit():
            return False
        if lst.match(w, sim):
            return True
        if not cased:
            return t not in COMMON and len(t) >= 3
        if t in COMMON and (not w[:1].isupper()):
            return False
        if not w[:1].isupper():
            return False
        if initial(k):
            return t in cap_seen and t not in COMMON
        return t != 'i'
    ok = {k: name_like(k) for k in idx}
    runs, cur, pend = ([], [], None)
    for k in idx:
        if ok[k]:
            if cur and k == cur[-1] + 1:
                cur.append(k)
            elif cur and pend is not None and (pend == cur[-1] + 1) and (k == pend + 1):
                cur += [pend, k]
            else:
                if cur:
                    runs.append(cur)
                cur = [k]
            pend = None
        elif cur and pend is None and (k == cur[-1] + 1) and (low(words[k]['w']) in CONNECT):
            pend = k
        else:
            if cur:
                runs.append(cur)
            cur, pend = ([], None)
    if cur:
        runs.append(cur)
    if not runs:
        return []
    return max(runs, key=lambda r: (sum((1 for k in r if ok.get(k))), -r[0]))
