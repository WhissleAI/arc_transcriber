import difflib, re
from .namesim import phon
LINK, ADD, ADD_STRICT = (0.85, 0.92, 0.97)
KIND_TAG = {'person': 'entity.mention.person', 'team': 'entity.mention.team', 'organization': 'entity.mention.organization', 'org': 'entity.mention.organization', 'place': 'entity.mention.place'}

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
            if len(parts) > 1:
                forms.add(parts[-1])
            forms.update((' '.join(str(a).split()) for a in c.get('aliases') or [] if str(a).strip()))
            for f in forms:
                fn = nrm(f)
                if fn:
                    self.F.append((n, c.get('kind', 'other'), fn, metaphone(fn)))

    def __bool__(self):
        return bool(self.F)

    def score(self, text):
        t = nrm(text)
        if not t:
            return (0.0, None, None, False)
        mt = metaphone(t)
        best = (0.0, None, None, False)
        for name, kind, fn, fm in self.F:
            sc = difflib.SequenceMatcher(None, t, fn).ratio()
            ph = bool(mt) and mt == fm
            s2 = max(sc, 0.9 if ph else 0.0)
            if s2 > best[0]:
                best = (s2, name, kind, ph)
        return best

def link(rec, kb):
    if not kb or rec.get('entity') or rec.get('state') == 'linked':
        return False
    sc, name, kind, ph = kb.score(rec.get('heard', ''))
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
            if not any((x[:1].isupper() for x in seg)) or len(nrm(t)) < 4:
                continue
            sc, name, kind, ph = kb.score(t)
            if sc >= ADD and (ph or sc >= ADD_STRICT) and name:
                out.append(dict(start=round(s, 3), end=round(e, 3), tag=KIND_TAG.get(kind, 'entity.mention.person'), heard=t, model=None, src='snap', entity=name, p=None, state='linked', via='snap', snap_score=round(sc, 3), words=[dict(w=x, start=w['start'], end=w['end']) for x, w in zip(seg, W[i:i + n])], show=True, id=new_id()))
                used |= rng
                taken.append((s, e))
    return out
