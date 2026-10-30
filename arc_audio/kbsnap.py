import difflib, re
from .namesim import phon
LINK, ADD, ADD_STRICT = (0.85, 0.92, 0.97)
KIND_TAG = {'person': 'entity.mention.person', 'team': 'entity.mention.team', 'organization': 'entity.mention.organization', 'org': 'entity.mention.organization', 'place': 'entity.mention.place'}

# words that start or carry a spoken command, and function words: never matched to a list entry on their own, and trimmed from the
# edges of a heard span before it is matched ("Read Frankenstein" is matched as "Frankenstein")
COMMAND_WORDS = {'read', 'reading', 'play', 'playing', 'open', 'show', 'start', 'stop', 'pause', 'resume', 'continue', 'call', 'text',
                 'send', 'set', 'turn', 'find', 'search', 'look', 'listen', 'watch', 'launch', 'close', 'skip', 'next', 'previous',
                 'mute', 'unmute', 'message', 'tell', 'ask', 'remind', 'add', 'remove', 'delete', 'go', 'take', 'put', 'give', 'get',
                 'buy', 'order', 'book', 'books', 'put', 'save', 'share', 'queue', 'shuffle', 'repeat', 'rewind', 'forward', 'louder',
                 'quieter', 'volume', 'up', 'down', 'please', 'hey', 'ok', 'okay', 'can', 'could', 'would', 'you', 'me', 'my', 'i',
                 'want', 'need', 'like', 'let', "let's", 'lets', 'some', 'something', 'by', 'from', 'about', 'called', 'named'}
STOPWORDS = {'a', 'an', 'the', 'of', 'to', 'in', 'on', 'at', 'for', 'and', 'or', 'with', 'is', 'it', 'be', 'this', 'that', 'these',
             'those', 'his', 'her', 'their', 'our', 'your', 'its', 'every', 'all', 'any', 'one', 'no', 'not', 'so', 'as', 'into',
             'over', 'after', 'before', 'again', 'now', 'then', 'there', 'here', 'what', 'who', 'which', 'when', 'where', 'how',
             'why', 'uh', 'um', 'er', 'ah', 'oh', 'yes', 'yeah', 'just', 'do', 'does', 'did', 'have', 'has', 'had', 'will', 'shall',
             'more', 'most', 'some', 'very', 'too', 'also'}
IGNORE = COMMAND_WORDS | STOPWORDS

def ignorable(w):
    return nrm(w) in IGNORE

def trim(words, key=lambda w: w):
    """drop command words and stopwords from both ends of a heard span"""
    i, j = (0, len(words))
    while i < j and ignorable(key(words[i])):
        i += 1
    while j > i and ignorable(key(words[j - 1])):
        j -= 1
    return (i, j)

def nrm(s):
    return ' '.join(re.sub("[^a-z0-9' ]", ' ', (s or '').lower()).split())

def metaphone(t):
    return ' '.join((phon(w) for w in t.split()))

class KB:

    def __init__(self, kb):
        self.F = []
        for c in kb or []:
            n = ' '.join(str(c.get('name', '')).split())
            if not n:
                continue
            forms = {n}
            parts = [p for p in n.split() if p.lower().strip('.') not in ('jr', 'sr', 'ii', 'iii', 'iv')] or [n]
            if len(parts) > 1 and (not ignorable(parts[-1])) and len(nrm(parts[-1])) >= 3:
                forms.add(parts[-1])
            forms.update((' '.join(str(a).split()) for a in c.get('aliases') or [] if str(a).strip()))
            for f in forms:
                fn = nrm(f)
                if fn:
                    self.F.append((n, c.get('kind', 'other'), fn, metaphone(fn)))

    def __bool__(self):
        return bool(self.F)

    def score(self, text):
        r = self.ranked(text, 1)
        return r[0] if r else (0.0, None, None, False)

    def ranked(self, text, k=5):
        """the list entries closest to a heard text, best first: [(score, name, kind, phonetic_match)], one per entry. Command
        words and stopwords at the edges of the text may be left out (the better of the two readings counts); a text of only such
        words matches nothing."""
        ws = nrm(text).split()
        i, j = trim(ws)
        if i >= j:
            return []
        texts = [(t, metaphone(t)) for t in dict.fromkeys([' '.join(ws), ' '.join(ws[i:j])])]
        best = {}
        for name, kind, fn, fm in self.F:
            for t, mt in texts:
                sc = difflib.SequenceMatcher(None, t, fn).ratio()
                ph = bool(mt) and mt == fm
                s2 = max(sc, 0.9 if ph else 0.0)
                if s2 > best.get(name, (0.0,))[0]:
                    best[name] = (s2, name, kind, ph)
        return sorted(best.values(), key=lambda x: -x[0])[:k]

def link(rec, kb):
    if not kb or rec.get('entity') or rec.get('state') == 'linked':
        return False
    R = kb.ranked(rec.get('heard', ''), 5)
    if R and 'snap_candidates' not in rec:
        rec['snap_candidates'] = [dict(entity=n, score=round(sc, 3)) for sc, n, _, _ in R]
    sc, name, kind, ph = R[0] if R else (0.0, None, None, False)
    if sc >= LINK and name:
        rec['entity'] = name
        rec['state'] = 'linked'
        rec['via'] = 'snap'
        rec['snap_score'] = round(sc, 3)
        return True
    return False

def add_spans(words, names, kb, new_id):
    if not kb or not words:
        return []
    taken = []
    for r in names:
        taken.append((r['start'], r['end']))
    W = [w for w in words if w.get('text') or w.get('w')]
    txt = [w.get('text', w.get('w', '')) for w in W]
    used = {i for i, w in enumerate(W) if str(w.get('tag') or '').startswith('ENTITY_')}
    out = []
    for n in (3, 2, 1):
        for i in range(0, len(W) - n + 1):
            rng = set(range(i, i + n))
            if rng & used:
                continue
            s, e = (W[i]['start'], W[i + n - 1]['end'])
            if any((a < e and s < b for a, b in taken)):
                continue
            seg = txt[i:i + n]
            t = ' '.join(seg)
            if ignorable(seg[0]) or ignorable(seg[-1]):
                continue
            if not any((x[:1].isupper() for x in seg)) or len(nrm(t)) < 4:
                continue
            R = kb.ranked(t, 5)
            sc, name, kind, ph = R[0] if R else (0.0, None, None, False)
            if sc >= ADD and (ph or sc >= ADD_STRICT) and name:
                out.append(dict(start=round(s, 3), end=round(e, 3), tag=KIND_TAG.get(kind, 'entity.mention.person'), heard=t, model=None, src='snap', entity=name, p=None, state='linked', via='snap', snap_score=round(sc, 3), snap_candidates=[dict(entity=n_, score=round(x, 3)) for x, n_, _, _ in R], words=[dict(w=x, start=w['start'], end=w['end']) for x, w in zip(seg, W[i:i + n])], show=True, id=new_id()))
                used |= rng
                taken.append((s, e))
    return out


def akey(t):
    """alias key of a text: lower case, possessive 's dropped, punctuation inside words removed ("De'Marco" == "DeMarco")"""
    out = []
    for w in str(t or '').split():
        w = re.sub("['\u2019]s$", '', w.lower().strip('.,?!;:"()[]'))
        w = re.sub('[^a-z0-9]', '', w)
        if w:
            out.append(w)
    return ' '.join(out)

class AliasIndex:
    """the list check before a name is called not_in_list: the heard text (edges trimmed of command / function words) against
    each entry's name and aliases with punctuation, case and spacing ignored ("DeMarco" = "De'Marco"), and against the
    partial names that only one entry has ("Comets" = "Riverside Comets", "Hale" = "De'Marco Hale")."""

    def __init__(self, kb):
        from .nametext import COMMON
        self.full, self.part = ({}, {})
        stop = IGNORE | COMMON
        for r, c in enumerate(kb or []):
            forms = [c.get('name', '')] + list(c.get('aliases') or [])
            for f in forms:
                k = akey(f)
                if k:
                    self.full.setdefault(k, set()).add(r)
                    self.full.setdefault(k.replace(' ', ''), set()).add(r)
            t = akey(c.get('name', '')).split()
            for n in range(1, len(t)):
                for i in range(len(t) - n + 1):
                    sub = t[i:i + n]
                    if all((x in stop or len(x) < 3 for x in sub)):
                        continue
                    k = ' '.join(sub)
                    self.part.setdefault(k, set()).add(r)
                    self.part.setdefault(k.replace(' ', ''), set()).add(r)

    def __bool__(self):
        return bool(self.full)

    def lookup(self, heard, allowed=None):
        """(row, via) for one list entry, via 'alias' (whole name or alias) or 'partial' (a part of one entry's name), else None"""
        ws = akey(heard).split()
        i, j = trim(ws)
        ws = ws[i:j]
        if not ws:
            return None
        for k in (' '.join(ws), ''.join(ws)):
            rows = self.full.get(k)
            if rows and len(rows) == 1:
                return (next(iter(rows)), 'alias')
            if rows:
                ok = [r for r in rows if allowed is None or r in allowed]
                if len(ok) == 1:
                    return (ok[0], 'alias')
                return None
        for k in (' '.join(ws), ''.join(ws)):
            rows = self.part.get(k)
            if rows and len(rows) == 1 and (self.full.get(k) is None):
                r = next(iter(rows))
                if allowed is None or r in allowed:
                    return (r, 'partial')
                return None
        return None
