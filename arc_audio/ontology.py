import hashlib, json
import numpy as np
FINE_DOMAINS = ['nba', 'nhl', 'nfl', 'cricket', 'soccer', 'esports', 'news', 'talk', 'stream_chat', 'meeting', 'court_appellate', 'court_trial', 'assistant', 'assistant_noisy', 'short_clips', 'unknown']
QUERIES = ('frame', 'span_in', 'span_end', 'utt', 'speaker', 'ev_at', 'ev_imm', 'ev_rec', 'item')
POOLED = ('utt', 'speaker', 'item')
SPORTS = ('nba', 'nhl', 'nfl', 'cricket', 'soccer')

def readouts(t):
    u = t['unit']
    if u in ('F', 'G'):
        return ('frame',)
    if u == 'W':
        return ('span_in', 'span_end')
    if u in ('U', 'T'):
        return ('utt',)
    if u == 'S':
        return ('speaker',)
    if u == 'D':
        return ('item',)
    if u == 'E':
        return ('ev_rec',) if t['id'].endswith('.recent') else ('ev_imm',) if t['id'].endswith('.imminent') else ('ev_at',)
    raise ValueError(f"unknown unit {u} for {t['id']}")

def domain_index(domain):
    d = (domain or 'unknown').lower()
    if d not in FINE_DOMAINS:
        raise ValueError(f'unknown domain {domain!r}; use one of {FINE_DOMAINS}')
    return FINE_DOMAINS.index(d)

class Table:

    def __init__(self, path):
        raw = open(path, 'rb').read()
        self.sha = hashlib.sha256(raw).hexdigest()
        j = json.loads(raw)
        self.version = j['version']
        self.j = j
        tags = sorted(j['tags'], key=lambda t: t['index'])
        self.n = max((t['index'] for t in tags)) + 1
        self.tag = [None] * self.n
        for t in tags:
            self.tag[t['index']] = t
        self.ids = [t['id'] if t else f'_unused{i}' for i, t in enumerate(self.tag)]
        self.idx = {t: i for i, t in enumerate(self.ids)}
        rg = {g['id']: g for g in j['groups']}
        self.groups, self.members = ([], {})
        for t in tags:
            if t.get('exclusivity') not in (None, 'independent'):
                if t['group'] not in self.members:
                    self.groups.append(t['group'])
                    self.members[t['group']] = []
                self.members[t['group']].append(t['index'])
        self.group_of = {i: g for g, m in self.members.items() for i in m}
        self.gate = {g: rg.get(g, {}).get('gate') for g in self.groups}
        self.group_info = rg
        self.trained = [i for i, t in enumerate(self.tag) if t and t.get('status') not in ('reserved', 'derived')]
        dg = j['domains']
        V = np.zeros((len(FINE_DOMAINS), self.n), bool)
        for i in self.trained:
            keys = self.tag[i]['domains']
            for d, fd in enumerate(FINE_DOMAINS):
                V[d, i] = any((k == '*' or k == fd or fd in dg.get(k, []) for k in keys))
        V[FINE_DOMAINS.index('unknown')] |= V.any(0)
        self.valid = V
        self.ro = {i: readouts(self.tag[i]) for i in self.trained}
        self.cols = {q: [i for i in self.trained if q in self.ro[i]] for q in QUERIES}
        self.facet = {i: self.tag[i]['facet'] for i in self.trained}
        self.status = {i: self.tag[i]['status'] for i in self.trained}

    def describe(self, tag_id):
        t = self.tag[self.idx[tag_id]]
        return dict(id=t['id'], facet=t['facet'], unit=t['unit'], group=t.get('group'), description=t.get('description'), paraphrases=t.get('paraphrases') or [], status=t.get('status'))
