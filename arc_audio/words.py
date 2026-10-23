import math, re, threading
NUMWORD = re.compile('^(\\d[\\d,.:\\-]*|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|half|quarter|dozen|first|second|third|fourth|fifth|point|oh)$', re.I)

def frames_of(start, end):
    f0 = int(math.floor(float(start) * 10 + 1e-06))
    return (f0, max(f0, int(math.floor(float(end) * 10 + 1e-06))))

def own_end(words, j):
    a, b = (words[j]['f0'], words[j]['f1'])
    if j + 1 < len(words) and a < words[j + 1]['f0'] <= b:
        b = words[j + 1]['f0'] - 1
    return max(a, b)

def norm_event(e, source='external'):
    txt = str(e.get('text', e.get('word', e.get('punctuated_word', '')))).strip()
    st, en = (float(e.get('start', e.get('t0', 0.0))), float(e.get('end', e.get('t1', 0.0))))
    en = max(en, st + 0.04)
    f0, f1 = frames_of(st, en)
    d = dict(w=txt, start=round(st, 3), end=round(en, 3), f0=f0, f1=f1, final=bool(e.get('is_final', e.get('final', True))), src=e.get('source', source))
    if e.get('arrive_f') is not None:
        d['arrive_f'] = int(e['arrive_f'])
        if e.get('oe_f') is not None:
            d['oe_f'] = int(e['oe_f'])
    if e.get('tag'):
        d['tag'] = str(e['tag'])
        d['tag_begin'] = bool(e.get('tag_begin', False))
    return d

class WordStore:

    def __init__(self):
        self.finals, self.interim, self.covered = ([], [], 0.0)
        self.spans_final = []
        self.final_upto = 0.0
        self.tagged = False
        self.partial_upto = 0.0
        self.version = 0
        self.lock = threading.RLock()

    def add(self, events, source='external'):
        with self.lock:
            return self._add(events, source)

    def _add(self, events, source):
        self.version += 1
        evs = [norm_event(e, source) for e in events if str(e.get('text', e.get('word', e.get('punctuated_word', '')))).strip()]
        fin = [e for e in evs if e['final']]
        inter = [e for e in evs if not e['final']]
        if inter:
            self.interim = sorted(inter, key=lambda w: w['start'])
            self.partial_upto = max(self.partial_upto, max((w['end'] for w in inter)))
        if fin:
            lo, hi = (min((w['start'] for w in fin)), max((w['end'] for w in fin)))
            self.interim = [w for w in self.interim if w['end'] <= lo or w['start'] >= hi]
            self.finals = sorted(self.finals + fin, key=lambda w: (w['start'], w['end']))
            self.covered = max(self.covered, hi)
            if any((w.get('tag') for w in fin)):
                from .tagdec import spans_of_words
                self.spans_final.append(spans_of_words(sorted(fin, key=lambda w: (w['start'], w['end']))))
                self.tagged = True
            self.final_upto = max(self.final_upto, hi)
        return len(fin)

    def mark_covered(self, t):
        with self.lock:
            if float(t) > self.covered:
                self.covered = float(t)
                self.version += 1

    def set_partial(self, events, upto, source='partial'):
        with self.lock:
            self.interim = sorted([dict(norm_event(e, source), final=False) for e in events if str(e.get('text', '')).strip()], key=lambda w: w['start'])
            self.partial_upto = max(self.partial_upto, float(upto))
            self.version += 1
            if any((w.get('tag') for w in self.interim)):
                self.tagged = True

    def partial_spans(self):
        with self.lock:
            from .tagdec import spans_of_words
            return spans_of_words(self.interim) if self.interim else []

    def clear_interim(self, before):
        with self.lock:
            n = len(self.interim)
            self.interim = [w for w in self.interim if w['start'] >= before]
            self.final_upto = max(self.final_upto, float(before) - 0.5)
            if len(self.interim) != n:
                self.version += 1

    def view(self):
        with self.lock:
            v = WordStore.__new__(WordStore)
            fin = self.finals
            extra = [w for w in self.interim if not any((f['end'] > w['start'] and f['start'] < w['end'] for f in fin[-50:]))]
            v.finals = sorted(fin + extra, key=lambda w: (w['start'], w['end'])) if extra else list(fin)
            v.interim = []
            v.covered = max(self.covered, self.partial_upto)
            v.partial_upto = v.covered
            v.version = self.version
            v.lock = threading.RLock()
            v.n_interim = len(extra)
            v.spans_final = []
            v.final_upto = self.final_upto
            v.tagged = self.tagged
            return v

    def overlapping(self, f0, f1):
        return [w for w in self.finals if w['f1'] >= f0 and w['f0'] <= f1]

    def text_between(self, t0, t1):
        return ' '.join((w['w'] for w in self.finals if w['end'] > t0 and w['start'] < t1))

    def heard_tail(self, f, n_words=3, lag=2, hold=30):
        av = [w for w in self.finals if w['f1'] + lag <= f]
        if not av:
            return ''
        last = av[-1]
        if f - (last['f1'] + lag) > hold:
            return ''
        t0 = last['f1'] + lag - hold
        return ' '.join(' '.join((w['w'] for w in av[-n_words:] if w['f1'] + lag >= t0)).lower().split())

class WordEventAdapter:

    def __init__(self, stream, parse=None, offset_s=0.0):
        self.stream, self.parse, self.offset = (stream, parse or self.default_parse, float(offset_s))

    @staticmethod
    def default_parse(msg):
        if isinstance(msg, (bytes, str)):
            import json
            msg = json.loads(msg)
        fin = bool(msg.get('is_final', msg.get('final', False)))
        ws = msg.get('words')
        if ws is None:
            alts = (msg.get('channel') or {}).get('alternatives') or [{}]
            ws = alts[0].get('words') or []
        return [dict(text=w.get('punctuated_word') or w.get('word') or w.get('text', ''), start=w['start'], end=w['end'], is_final=fin) for w in ws if 'start' in w and 'end' in w]

    def on_message(self, msg):
        evs = self.parse(msg)
        for e in evs:
            e['start'] = float(e['start']) + self.offset
            e['end'] = float(e['end']) + self.offset
        return self.stream.add_words(evs) if evs else None
