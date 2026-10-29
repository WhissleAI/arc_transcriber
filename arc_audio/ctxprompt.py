import re
CTX_PREV, CTX_KB, CTX_DOM, CTX_ROLE, CTX_SEP = (32640, 32641, 32642, 32643, 32644)
KB_TYPES = ['person', 'team', 'org', 'place', 'law', 'product', 'term', 'id']
TYPE_ID = {t: 32645 + i for i, t in enumerate(KB_TYPES)}
PREV_WORDS, PREV_WIN_S, GUARD_S = (40, 60.0, 0.3)

def _num_words(tok):
    try:
        from num2words import num2words
        if re.fullmatch('\\d+(\\.\\d+)?', tok):
            return num2words(float(tok) if '.' in tok else int(tok)).replace(',', '')
    except Exception:
        pass
    return tok

def norm_prev(text):
    t = (text or '').replace('%', ' percent ').replace('&', ' and ')
    t = re.sub('(?<=\\d),(?=\\d{3})', '', t)
    t = re.sub('\\d+(?:\\.\\d+)?', lambda m: ' ' + _num_words(m.group(0)) + ' ', t)
    t = t.lower().replace('’', "'")
    t = re.sub("[^a-z' ]+", ' ', t)
    t = re.sub("(?<![a-z])'|'(?![a-z])", ' ', t)
    return ' '.join(t.split())

def prev_text(hist, t_now):
    if t_now is None:
        return ''
    W = []
    for s, e, txt in hist:
        if s >= t_now - PREV_WIN_S and e <= t_now - GUARD_S:
            W += norm_prev(txt).split()
    return ' '.join(W[-PREV_WORDS:])

def build_prompt(tok, ctx, mode='full', max_tokens=96, prev_max=48, kb_max=40, use_dom=True, use_role=False, use_types=True):
    if not ctx or mode == 'none':
        return []
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    out = []
    if use_dom and ctx.get('dom'):
        out += [CTX_DOM] + enc(ctx['dom'].replace('_', ' '))[:4]
    if use_role and ctx.get('role'):
        out += [CTX_ROLE] + enc(ctx['role'].replace('_', ' '))[:4]
    if mode == 'full':
        kb = list(ctx.get('kb') or [])
        ty = list(ctx.get('kb_type') or [])
        ty = ty + [''] * (len(kb) - len(ty))
        items = list(zip(kb, ty))
        sel, cost = ([], 0)
        lim = kb_max - (1 if use_types and any((y in TYPE_ID for _, y in items)) else 0)
        for t, y in items:
            ids = enc(t)
            if cost + len(ids) + 1 > lim:
                continue
            sel.append((ids, y))
            cost += len(ids) + 1
        if sel:
            typed = use_types and all((y in TYPE_ID for _, y in sel)) and (cost + 1 <= kb_max)
            k_ids = []
            for ids, y in sel:
                k_ids += [TYPE_ID[y]] + ids if typed else ids + [CTX_SEP]
            out += [CTX_KB] + (k_ids if typed else k_ids[:-1])
    p = ctx.get('prev') or ''
    if p:
        ids = enc(p)
        budget = min(prev_max, max_tokens - len(out) - 1)
        if budget > 4:
            out += [CTX_PREV] + ids[-budget:]
    return out[:max_tokens]
