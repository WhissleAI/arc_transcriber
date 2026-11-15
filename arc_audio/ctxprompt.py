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

# ---------------------------------------------------------------------------------------------- layered context (v2)
# [dom][role][session][kb][prev] (global -> local, prev closest to the decoder start), budgets in tokens (marker included):
# max (whole prompt), ses_max, kb_max, prev_max. Session memory = dict(spk=[[slot, role, turns]], nums=[[value, kind, minutes]],
# ents=[[surface, type, minutes, mentions]]) in salience order; entries that do not fit are dropped from the end.
CTX_SES = 32653
BUDGET_V2 = dict(max=112, prev_max=48, kb_max=56, ses_max=300)


def session_entries(ses):
    out = []
    if not ses:
        return out
    for x in ses.get('spk') or []:
        slot = str(x[0]) if x else ''
        role = str(x[1]).replace('_', ' ') if len(x) > 1 and x[1] else ''
        if slot:
            out.append((CTX_SEP, f'{slot} {role}'.strip()))
    for x in ses.get('nums') or []:
        if not x or x[0] in (None, ''):
            continue
        kind = str(x[1]).replace('_', ' ') if len(x) > 1 and x[1] else 'number'
        age = f' {int(x[2])}m' if len(x) > 2 and x[2] is not None else ''
        out.append((CTX_SEP, f'{kind} {x[0]}{age}'))
    for x in ses.get('ents') or []:
        if not x or not x[0]:
            continue
        ty = str(x[1]) if len(x) > 1 and x[1] else ''
        age = f' {int(x[2])}m' if len(x) > 2 and x[2] is not None else ''
        if ty in TYPE_ID:
            out.append((TYPE_ID[ty], f'{x[0]}{age}'))
        else:
            out.append((CTX_SEP, f'{ty} {x[0]}{age}'.strip()))
    return out


def render_session(tok, entries, budget):
    if not entries or budget < 3:
        return []
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    out = [CTX_SES]
    for mk, t in entries:
        ids = enc(t)
        if len(out) + 1 + len(ids) > budget:
            break
        out += [mk] + ids
    return out if len(out) > 1 else []


def build_prompt_v2(tok, ctx, mode='full', budget=None, use_dom=True, use_role=False, use_types=True):
    """layered prompt of the long-context Arc decoders; ctx keys dom / role / session / kb / kb_type / prev"""
    if not ctx or mode == 'none':
        return []
    B = dict(BUDGET_V2, **(budget or {}))
    mx = int(B['max'])
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    out = []
    if use_dom and ctx.get('dom'):
        out += [CTX_DOM] + enc(ctx['dom'].replace('_', ' '))[:4]
    if use_role and ctx.get('role'):
        out += [CTX_ROLE] + enc(ctx['role'].replace('_', ' '))[:4]
    ses = ctx.get('session')
    if ses:
        out += render_session(tok, session_entries(ses) if isinstance(ses, dict) else list(ses), min(int(B.get('ses_max', 0)), mx - len(out)))
    if mode == 'full' and ctx.get('kb'):
        kbud = min(int(B['kb_max']), mx - len(out))
        if kbud > 2:
            out += build_prompt(tok, dict(kb=ctx.get('kb'), kb_type=ctx.get('kb_type')), 'full', kbud, 0, kbud, use_dom=False, use_role=False, use_types=use_types)
    p = ctx.get('prev') or ''
    if p:
        ids = enc(p)
        budget_p = min(int(B['prev_max']), mx - len(out) - 1)
        if budget_p > 4:
            out += [CTX_PREV] + ids[-budget_p:]
    return out[:mx]
