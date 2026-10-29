import difflib, re
BASE_ID = 32600
TAGS = ['ENTITY_PERSON', 'ENTITY_TEAM', 'ENTITY_ORG', 'ENTITY_PLACE', 'ENTITY_WORK', 'ENTITY_EVENT', 'ENTITY_LEGAL', 'ENTITY_GAME', 'ENTITY_MAP', 'NUMBER', 'NUMBER_CLOCK', 'NUMBER_COUNT', 'NUMBER_DATE', 'NUMBER_DURATION', 'NUMBER_FIELDPOS', 'NUMBER_ID', 'NUMBER_JERSEY', 'NUMBER_MEASURE', 'NUMBER_MONEY', 'NUMBER_ODDS', 'NUMBER_ORDINAL', 'NUMBER_PERCENT', 'NUMBER_REF', 'NUMBER_SCORE', 'NUMBER_STAT', 'NUMBER_TIME', 'END']
TAGS = TAGS + ['ENTITY_LAW', 'ENTITY_PRODUCT', 'ENTITY_TERM', '_RESERVED_A', '_RESERVED_B', 'CONTACT_PHONE', 'CONTACT_EMAIL', 'CONTACT_URL', 'CONTACT_ADDRESS']
TID = {t: BASE_ID + i for i, t in enumerate(TAGS) if not t.startswith('_RESERVED')}
ID2T = {v: k for k, v in TID.items()}
END_ID = TID['END']
RESERVED_IDS = [BASE_ID + i for i, t in enumerate(TAGS) if t.startswith('_RESERVED')]
NEVER_EMIT = sorted(set(RESERVED_IDS) | set(range(BASE_ID + len(TAGS), 32768)))
CONTACT_TAG = {'CONTACT_PHONE': 'entity.contact.phone', 'CONTACT_EMAIL': 'entity.contact.email', 'CONTACT_URL': 'entity.contact.url', 'CONTACT_ADDRESS': 'entity.contact.address'}
NAME_TAG = {'ENTITY_PERSON': 'entity.mention.person', 'ENTITY_TEAM': 'entity.mention.team', 'ENTITY_ORG': 'entity.mention.organization', 'ENTITY_PLACE': 'entity.mention.place', 'ENTITY_WORK': 'entity.mention.work_product', 'ENTITY_EVENT': 'entity.mention.event_name', 'ENTITY_LEGAL': 'entity.mention.legal_ref', 'ENTITY_GAME': 'entity.mention.game_entity', 'ENTITY_MAP': 'entity.mention.game_map', 'ENTITY_LAW': 'entity.mention.law', 'ENTITY_PRODUCT': 'entity.mention.product', 'ENTITY_TERM': 'entity.mention.term'}
NUMBER_KIND = {'NUMBER': None, 'NUMBER_CLOCK': 'clock', 'NUMBER_COUNT': 'count', 'NUMBER_DATE': 'date', 'NUMBER_DURATION': 'duration', 'NUMBER_FIELDPOS': 'field_position', 'NUMBER_ID': 'id_pii', 'NUMBER_JERSEY': 'jersey_number', 'NUMBER_MEASURE': 'measure', 'NUMBER_MONEY': 'money', 'NUMBER_ODDS': 'ratio_odds', 'NUMBER_ORDINAL': 'ordinal', 'NUMBER_PERCENT': 'percent', 'NUMBER_REF': 'ref_number', 'NUMBER_SCORE': 'score', 'NUMBER_STAT': 'stat', 'NUMBER_TIME': 'time_of_day'}

def is_tagged_tokenizer(tok):
    try:
        return tok.convert_ids_to_tokens(END_ID) == '<<END>>'
    except Exception:
        return False

def layer(tag):
    return 'name' if tag in NAME_TAG else 'number' if tag in NUMBER_KIND else 'contact' if tag in CONTACT_TAG else None

def readout(tok, ids):
    words, spans = ([], [])
    cur = []
    open_tag = None
    open_w = None

    def flush():
        if cur:
            words.append(tok.decode(cur, skip_special_tokens=True).strip())
            cur.clear()
    for t in ids:
        t = int(t)
        if t in ID2T:
            flush()
            if open_tag is not None and len(words) > open_w:
                spans.append((open_tag, open_w, len(words)))
            open_tag = None
            if t != END_ID:
                open_tag, open_w = (ID2T[t], len(words))
            continue
        if t < 3 or t >= 32000:
            continue
        piece = tok.convert_ids_to_tokens(t)
        if piece.startswith('▁') and cur:
            flush()
        cur.append(t)
    flush()
    if open_tag is not None and len(words) > open_w:
        spans.append((open_tag, open_w, len(words)))
    words2, mp = ([], {})
    for j, w in enumerate(words):
        if w:
            mp[j] = len(words2)
            words2.append(w)
    sp = []
    for t, a, b in spans:
        js = [mp[j] for j in range(a, b) if j in mp]
        if js:
            sp.append((t, min(js), max(js) + 1))
    return (words2, sp)

def _key(w):
    return re.sub("[^\\w']", '', str(w).lower())

def map_spans(src_words, spans, dst_words):
    if not spans:
        return []
    if [_key(w) for w in src_words] == [_key(w) for w in dst_words]:
        return list(spans)
    sm = difflib.SequenceMatcher(None, [_key(w) for w in src_words], [_key(w) for w in dst_words], autojunk=False)
    m = {}
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            m[a + k] = b + k
    out = []
    for t, a, b in spans:
        js = [m[j] for j in range(a, b) if j in m]
        if js:
            out.append((t, min(js), max(js) + 1))
    return out

def annotate(events, spans):
    ev = [dict(e) for e in events]
    for t, a, b in spans:
        for j in range(max(0, a), min(b, len(ev))):
            ev[j]['tag'] = t
            ev[j]['tag_begin'] = j == a
    return ev

def annotate_times(events, spans):
    ev = sorted((dict(e) for e in events), key=lambda e: float(e.get('start', 0.0)))
    for s in spans or []:
        if isinstance(s, dict):
            st, en, t = (float(s['start']), float(s['end']), s['tag'])
        else:
            st, en, t = (float(s[0]), float(s[1]), s[2])
        t = {v: k for k, v in NAME_TAG.items()}.get(t, t)
        first = True
        for e in ev:
            mid = (float(e.get('start', 0.0)) + float(e.get('end', 0.0))) / 2
            if st - 0.001 <= mid <= en + 0.001:
                e['tag'] = t
                e['tag_begin'] = first
                first = False
    return ev

def spans_of_words(words):
    out, cur = ([], None)
    for w in words:
        t = w.get('tag')
        if t and cur is not None and (t == cur['tag']) and (not w.get('tag_begin')):
            cur['end'] = w['end']
            cur['keys'].append((w['start'], w['w']))
            continue
        if cur is not None:
            out.append(cur)
            cur = None
        if t and layer(t):
            cur = dict(tag=t, start=w['start'], end=w['end'], keys=[(w['start'], w['w'])])
    if cur is not None:
        out.append(cur)
    return out
