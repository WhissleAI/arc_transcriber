import collections, re, unicodedata
import numpy as np
try:
    from rapidfuzz import fuzz as _fuzz

    def ratio(a, b):
        return _fuzz.ratio(a, b) / 100.0
except Exception:

    def ratio(a, b):
        if not a and (not b):
            return 1.0
        if not a or not b:
            return 0.0
        prev = [0] * (len(b) + 1)
        for ca in a:
            cur = [0]
            for j, cb in enumerate(b):
                cur.append(prev[j] + 1 if ca == cb else max(prev[j + 1], cur[j]))
            prev = cur
        return 2.0 * prev[-1] / (len(a) + len(b))
try:
    import jellyfish as _jf
    _metaphone = _jf.metaphone
except Exception:
    _jf = None

    # Fallback port of jellyfish.metaphone (https://github.com/jamesturk/jellyfish).
    # MIT License, Copyright (c) 2015 James Turk; the licence text is in NOTICE.
    def _metaphone(s):
        result = []
        s = unicodedata.normalize('NFKD', s.lower())
        if s.startswith(('kn', 'gn', 'pn', 'wr', 'ae')):
            s = s[1:]
        i = 0
        while i < len(s):
            c = s[i]
            nx = s[i + 1] if i < len(s) - 1 else '*****'
            nn = s[i + 2] if i < len(s) - 2 else '*****'
            if c == nx and c != 'c':
                i += 1
                continue
            if c in 'aeiou':
                if i == 0 or s[i - 1] == ' ':
                    result.append(c)
            elif c == 'b':
                if not (i != 0 and s[i - 1] == 'm') or nx:
                    result.append('b')
            elif c == 'c':
                if nx == 'i' and nn == 'a' or nx == 'h':
                    result.append('x')
                    i += 1
                elif nx in 'iey':
                    result.append('s')
                    i += 1
                else:
                    result.append('k')
            elif c == 'd':
                if nx == 'g' and nn in 'iey':
                    result.append('j')
                    i += 2
                else:
                    result.append('t')
            elif c in 'fjlmnr':
                result.append(c)
            elif c == 'g':
                if nx in 'iey':
                    result.append('j')
                elif nx == 'h' and nn and (nn not in 'aeiou'):
                    i += 1
                elif nx == 'n' and (not nn):
                    i += 1
                else:
                    result.append('k')
            elif c == 'h':
                if i == 0 or nx in 'aeiou' or s[i - 1] not in 'aeiou':
                    result.append('h')
            elif c == 'k':
                if i == 0 or s[i - 1] != 'c':
                    result.append('k')
            elif c == 'p':
                if nx == 'h':
                    result.append('f')
                    i += 1
                else:
                    result.append('p')
            elif c == 'q':
                result.append('k')
            elif c == 's':
                if nx == 'h':
                    result.append('x')
                    i += 1
                elif nx == 'i' and nn in 'oa':
                    result.append('x')
                    i += 2
                else:
                    result.append('s')
            elif c == 't':
                if nx == 'i' and nn in 'oa':
                    result.append('x')
                elif nx == 'h':
                    result.append('0')
                    i += 1
                elif nx != 'c' or nn != 'h':
                    result.append('t')
            elif c == 'v':
                result.append('f')
            elif c == 'w':
                if i == 0 and nx == 'h':
                    i += 1
                    result.append('w')
                elif nx in 'aeiou':
                    result.append('w')
            elif c == 'x':
                if i == 0:
                    result.append('x' if nx == 'h' or (nx == 'i' and nn in 'oa') else 's')
                else:
                    result.append('k')
                    result.append('s')
            elif c == 'y':
                if nx in 'aeiou':
                    result.append('y')
            elif c == 'z':
                result.append('s')
            elif c == ' ':
                if result and result[-1] != ' ':
                    result.append(' ')
            i += 1
        return ''.join(result).upper()
_MP = {}

def mpw(t):
    if t not in _MP:
        _MP[t] = _metaphone(t)
    return _MP[t]

def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    s = re.sub("'s\\b", '', s)
    return ' '.join(re.sub('[^a-z0-9 ]+', ' ', s).split())

def name_forms(c):
    nm = norm(c.get('name', ''))
    fs = {nm} | {norm(a) for a in c.get('aliases') or []}
    return [f for f in fs if f and (not (' ' not in f and f == nm.replace(' ', '') and (' ' in nm)))]

def sim(span, forms):
    if not span or not forms:
        return 0.0
    st = span.split()
    best = 0.0
    for n in forms:
        toks = n.split()
        s1 = ratio(span, n)
        s3 = max((ratio(a, b) for a in st for b in toks if len(a) >= 3 and len(b) >= 3), default=0)
        s4 = max((1.0 for a in st for b in toks if len(a) >= 3 and len(b) >= 3 and (mpw(a) == mpw(b))), default=0)
        s5 = ratio(' '.join((mpw(t) for t in st)), ' '.join((mpw(t) for t in toks)))
        best = max(best, s1, 0.9 * s3, 0.85 * s4, 0.9 * s5)
    return best

class NameRescorer:

    def __init__(self, kb, params):
        self.kb = kb
        self.p = dict(params)
        self.forms = [name_forms(c) for c in kb]
        self.groups = [c.get('group') for c in kb]
        teams = {c.get('group') for c in kb if c.get('kind') == 'team' and c.get('group')}
        self.team_words = {}
        for c in kb:
            if c.get('kind') == 'team' and c.get('group') in teams:
                self.team_words.setdefault(c['group'], set()).update((t for f in name_forms(c) for t in f.split() if len(t) >= 4))
        self.recent = collections.deque()

    def score(self, t_s, z, allowed, span_text, past_text='', feed_text=None):
        p = self.p
        while self.recent and self.recent[0][0] < t_s - 120:
            self.recent.popleft()
        rc = collections.Counter((c for _, c in self.recent))
        zz = np.asarray(z, np.float64)
        ok = zz > -1000.0
        lp = zz - np.logaddexp.reduce(zz[ok])
        allowed = np.asarray(allowed, np.int64)
        span = norm(span_text)
        sims = np.array([sim(span, self.forms[r]) for r in allowed], np.float64)
        sh = np.clip((sims - 0.6) / 0.4, 0, 1) ** 2
        s = lp[2 + allowed].copy() if len(allowed) else np.zeros(0)
        if p.get('A'):
            s = s + p['A'] * sh
        if p.get('B') and feed_text:
            ft = {tok for tok in norm(feed_text).split() if len(tok) >= 4}
            s = s + p['B'] * np.array([1.0 if any((tk in ft for f in self.forms[r] for tk in f.split() if len(tk) >= 4)) else 0.0 for r in allowed])
        if p.get('T') and self.team_words:
            past = norm(past_text)
            src = {g: sum((past.count(f) for f in fs)) for g, fs in self.team_words.items()}
            tot = sum(src.values())
            if tot:
                s = s + p['T'] * np.array([src.get(self.groups[r], 0) / tot if self.groups[r] else 0.0 for r in allowed])
        if p.get('R') and rc:
            s = s + p['R'] * np.array([min(1.0, rc.get(int(r), 0) / 2) for r in allowed])
        ms = float(sh.max()) if len(sh) else 0.0
        sn = float(lp[0]) + p.get('bias', 0.0) + p.get('A2', 0.0) * (1 - ms)
        allv = np.r_[sn, s]
        pr = np.exp(allv - allv.max())
        pr /= pr.sum()
        j = int(np.argmax(allv))
        conf = float(pr[j])
        pred = None if j == 0 else int(allowed[j - 1])
        link = pred is not None and conf >= p.get('thr', 0.5)
        if link:
            self.recent.append((t_s, pred))
        order = np.argsort(-pr[1:], kind='stable')
        ranked = [(int(allowed[k]), float(pr[1 + k])) for k in order]
        return dict(pred=pred, p=round(conf, 4), link=bool(link), sim=round(float(sims[j - 1]), 3) if pred is not None else None, ranked=ranked, p_new=round(float(pr[0]), 4))
