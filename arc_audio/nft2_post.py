"""Arc Transcriber Space (nft2): read-outs on top of an utterance's words, the same in arc_audio and libarc.

  * guard(words, y, t0)     the no-speech gate and the tail / repetition guard (drops words the audio does not carry);
  * disfluency(words)       per-word 'disfl' = filler | repeat | fragment, and the text without them (clean_text);
  * almost(...)             sound-alike matches of unlinked heard words against the names list (state 'almost', the best
                            candidate and a link probability);
  * eot_prob(...)           a heuristic end-of-turn probability from the pause and the last words (streams).

Every function takes plain lists / dicts (the utterance's words: text, start, end, conf, tag) so that libarc's C
(c/src/arc_post.c) can follow it line by line. Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import math
import re

import numpy as np

from .phonetic import key as phon_key, similarity as phon_sim

GUARD = dict(silence_db=-60.0, flat_db=6.0, flat_conf=0.6, rep_conf=0.35, tail_s=0.8, tail_conf=0.4)


def _wk(w):
    return re.sub("[^a-z0-9']", '', str(w).lower())


def frame_dbs(y, fr=1600):
    """100 ms frame levels (dBFS, as api.frame_db) of y"""
    y = np.asarray(y, np.float32)
    n = (len(y) + fr - 1) // fr
    out = []
    for f in range(n):
        a = y[f * fr:(f + 1) * fr]
        out.append(-120.0 if len(a) == 0 else float(20 * np.log10(np.sqrt(np.mean(a.astype(np.float64) ** 2)) + 1e-06)))
    return out


def voiced_end(dbs, t0, margin_db=6.0):
    """end time (s) of the last 100 ms frame above the clip's floor (its 10th percentile level) + margin_db; None when no
    frame is"""
    if not dbs:
        return None
    s = sorted(dbs)
    floor = s[int(0.1 * (len(s) - 1))]
    last = None
    for f, d in enumerate(dbs):
        if d > floor + margin_db:
            last = f
    return None if last is None else t0 + (last + 1) * 0.1


def guard(words, y, t0=0.0, cfg=None):
    """words (dicts with text, start, end, conf) of one utterance decoded from audio y (starting at t0 s) -> (kept words,
    reason per dropped word). Words without 'conf' are never dropped by the confidence rules.
      no-speech gate: every 100 ms frame below silence_db, or a flat clip (loudest frame < flat_db over the 10th
        percentile) whose words have a mean confidence < flat_conf -> no words;
      repetition: a word repeating the previous one (same key) with confidence < rep_conf ('you you you', 'Vavilov
        Vavilov' written by the names bias) -> dropped;
      tail: a word starting more than tail_s after the last voiced frame with confidence < tail_conf -> dropped (the RNNT
        emits a word up to ~0.5 s after it is said, so the time rule alone would cut real last words)."""
    c = dict(GUARD, **(cfg or {}))
    if not words:
        return [], []
    dbs = frame_dbs(y)
    confs = [w['conf'] for w in words if w.get('conf') is not None]
    if dbs:
        s = sorted(dbs)
        top, floor = s[-1], s[int(0.1 * (len(s) - 1))]
        if top < c['silence_db']:
            return [], ['no_speech'] * len(words)
        if confs and top - floor < c['flat_db'] and sum(confs) / len(confs) < c['flat_conf']:
            return [], ['no_speech'] * len(words)
    ve = voiced_end(dbs, t0)
    keep, why = [], []
    prev = None
    for w in words:
        cf = w.get('conf')
        k = _wk(w['text'])
        if cf is not None and prev is not None and k and k == prev and cf < c['rep_conf']:
            why.append('repeat')
            continue
        if cf is not None and ve is not None and w['start'] > ve + c['tail_s'] and cf < c['tail_conf']:
            why.append('tail')
            continue
        keep.append(w)
        prev = k
    return keep, why


# ------------------------------------------------------------------------------------------------------------ disfluency
FILLERS = frozenset('um umm uh uhh uhm er erm ah ahh eh hmm hmmm hm mm mmm'.split())
_YK_PREV = frozenset("do did don't didn't does doesn't would if as what how let".split())
_YK_NEXT = frozenset('what how where who why when that about it him her them me the a an this'.split())
_IM_PREV = frozenset("what do did don't didn't does".split())
_IM_NEXT = frozenset('it that what this him her them you'.split())


def disfluency(words):
    """words (dicts with text) -> the same words, each with 'disfl': None | 'filler' | 'repeat' | 'fragment'
      filler: um / uh / er / hmm ...; "you know what i mean"; "you know" and "i mean" used as fillers (not "do you know",
        "you know what", "i mean it");
      repeat: the earlier copy of an immediately repeated word or word pair ("the the", "okay okay", "go to go to");
      fragment: a cut-off word followed by the word it starts ("resea research", "ro- rover")."""
    k = [_wk(w['text']) for w in words]
    n = len(k)
    tag = [None] * n
    for i in range(n):
        if k[i] in FILLERS:
            tag[i] = 'filler'
    i = 0
    while i < n:
        if k[i:i + 5] == ['you', 'know', 'what', 'i', 'mean']:
            for j in range(i, i + 5):
                tag[j] = 'filler'
            i += 5
            continue
        if k[i:i + 2] == ['you', 'know'] and (i == 0 or k[i - 1] not in _YK_PREV) and (i + 2 >= n or k[i + 2] not in _YK_NEXT) and n > 2:
            tag[i] = tag[i + 1] = 'filler'
            i += 2
            continue
        if k[i:i + 2] == ['i', 'mean'] and i + 2 < n and (i == 0 or k[i - 1] not in _IM_PREV) and k[i + 2] not in _IM_NEXT:
            tag[i] = tag[i + 1] = 'filler'
            i += 2
            continue
        i += 1
    for i in range(n - 1):
        if tag[i] is None and k[i] and k[i] == k[i + 1]:
            tag[i] = 'repeat'
        elif i + 3 < n and tag[i] is None and k[i] and k[i:i + 2] == k[i + 2:i + 4]:
            tag[i] = tag[i + 1] = 'repeat'
        elif tag[i] is None and len(k[i]) >= 2 and len(k[i + 1]) > len(k[i]) and k[i + 1].startswith(k[i]) and \
                (str(words[i]['text']).endswith('-') or len(k[i]) >= 3):
            tag[i] = 'fragment'
        elif tag[i] is None and str(words[i]['text']).endswith('-') and len(k[i]) >= 1:
            tag[i] = 'fragment'
    out = []
    for w, t in zip(words, tag):
        out.append(dict(w, disfl=t))
    return out


def clean_text(words):
    return ' '.join(w['text'] for w in words if not w.get('disfl'))


# ------------------------------------------------------------------------------------------------------------ almost
STOP = frozenset(("a an the to of in on at for and or but is are was were be been am i you he she it we they me my your "
                  "our us them this that these those what where when who why how which there here go goes going show "
                  "take fly see find look zoom tell about please can could would will do does did not no yes yeah okay "
                  "ok so then than just like um uh oh with from by up down out over into one two three four five six "
                  "seven eight nine ten more less some any all very too also its it's let's lets get give set send "
                  "move turn drive put make have has had know think want need").split())
# link probability of a sound-alike match: logistic over (phonetic similarity, letter similarity, letters in the name); a
# candidate also needs heard words the RNNT was unsure of (min word conf < max_conf: a name it does not know comes out as
# low-confidence pieces, a common word it heard well does not)
ALMOST = dict(min_sim=0.6, min_p=0.6, a=12.0, b=0.72, c=3.0, d=0.06, max_words=4, min_chars=4, max_conf=0.7)


def _letters(s):
    return re.sub('[^a-z]', '', str(s).lower())


def almost_p(heard, form, cfg=None):
    """the link probability of heard words for one list spelling (0 when not similar enough)"""
    import difflib
    c = dict(ALMOST, **(cfg or {}))
    sp = phon_sim(heard, form)
    if sp < c['min_sim']:
        return 0.0
    a, b = _letters(heard), _letters(form)
    sc = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0.0
    nl = len(phon_key(form).replace('A', ''))
    z = c['a'] * (sp - c['b']) + c['c'] * (sc - 0.5) + c['d'] * (min(nl, 10) - 5)
    return 1.0 / (1.0 + math.exp(-z))


def kb_forms(names):
    """names list entries -> [(kb index, entry name, spelling)] (the name and each alias)"""
    out = []
    for i, e in enumerate(names or []):
        nm = str(e.get('name', '')).strip()
        if not nm:
            continue
        for f in [nm] + [str(a) for a in e.get('aliases') or [] if str(a).strip()]:
            out.append((i, nm, f))
    return out


def _suffix_ok(heard, form):
    """a spelling ending in a one-letter word ('Abulfeda E', a satellite crater) needs as many heard words: the heard
    'abulfeda' is the main crater, another entity"""
    fw = str(form).split()
    return not (len(fw) > 1 and len(_letters(fw[-1])) == 1 and len(str(heard).split()) < len(fw))


def common_words(pieces):
    """the RNNT vocabulary's whole-word pieces (lower case): words the model spells as one piece are common words, not
    names (a heard name is split into pieces: 'albyrini' = al + by + rini)"""
    return frozenset(p[1:].lower() for p in getattr(pieces, 'p', pieces) if p.startswith('▁') and len(p) > 1)


def _all_common(ks, common):
    return all((not x) or x in STOP or (common is not None and x in common) for x in ks)


def _unsure(words, max_conf):
    """the min confidence of the words that are not stop words is below max_conf (no conf: unsure)"""
    cs = [w['conf'] for w in words if w.get('conf') is not None and _wk(w.get('text', '')) not in STOP]
    return not cs or min(cs) < max_conf


def best_candidate(heard, forms, cfg=None):
    """heard words -> (kb index, name, p) of the most likely list entry, or None"""
    best = None
    for i, nm, f in forms:
        if not _suffix_ok(heard, f):
            continue
        p = almost_p(heard, f, cfg)
        if p > 0 and (best is None or p > best[2]):
            best = (i, nm, p)
    return best


def almost(words, names, taken=(), cfg=None, common=None):
    """unlinked word spans of one utterance that sound like a list entry -> [dict(word_begin, word_end, heard, candidate,
    kb_index, p)], best first, non-overlapping. words: dicts with text (+ start / end); taken: word indices already in a
    linked name; common (common_words): words that are never a name on their own. Spans: 1 .. max_words words, not only
    stop / common words, not starting or ending with a stop word, >= min_chars letters; a span that IS a list spelling
    is left to the linker."""
    c = dict(ALMOST, **(cfg or {}))
    forms = kb_forms(names)
    if not forms or not words:
        return []
    exact = {_letters(f) for _, _, f in forms}
    k = [_wk(w['text']) for w in words]
    cands = []
    for a in range(len(words)):
        for b in range(a + 1, min(len(words), a + int(c['max_words'])) + 1):
            if any(j in taken for j in range(a, b)) or _all_common(k[a:b], common):
                continue
            if k[a] in STOP or k[b - 1] in STOP:
                continue
            heard = ' '.join(str(w['text']) for w in words[a:b])
            if len(_letters(heard)) < c['min_chars'] or _letters(heard) in exact or not _unsure(words[a:b], c['max_conf']):
                continue
            bc = best_candidate(heard, forms, c)
            if bc and bc[2] >= c['min_p']:
                cands.append((bc[2], a, b, heard, bc))
    cands.sort(key=lambda x: (-x[0], x[1], x[2]))
    used, out = set(), []
    for p, a, b, heard, bc in cands:
        if any(j in used for j in range(a, b)):
            continue
        used.update(range(a, b))
        out.append(dict(word_begin=a, word_end=b, heard=heard, candidate=bc[1], kb_index=bc[0], p=round(p, 4)))
    return sorted(out, key=lambda r: r['word_begin'])


def almost_names(out, names, cfg=None, common=None):
    """predict() result + the names list -> name records with state 'almost' (the best candidate and its p) for heard
    words that sound like a list entry but were not linked; an abstain / not_in_list record over such words becomes
    'almost' (entity stays None: the app decides). Linked records are untouched."""
    recs = list(out.get('names') or [])
    if not names:
        return recs
    forms = kb_forms(names)
    for r in recs:
        if r.get('state') in ('abstain', 'not_in_list') and r.get('heard') and not _all_common([_wk(x) for x in str(r['heard']).split()], common):
            ws = [w for u in out.get('transcript') or [] for w in u.get('words') or []
                  if r.get('start') is not None and w['start'] < r['end'] and w['end'] > r['start']]
            if not _unsure(ws, dict(ALMOST, **(cfg or {}))['max_conf']):
                continue
            bc = best_candidate(r['heard'], forms, cfg)
            if bc and bc[2] >= dict(ALMOST, **(cfg or {}))['min_p']:
                r.update(state='almost', candidate=bc[1], candidate_kb_index=bc[0], candidate_p=round(bc[2], 4))
    add = []
    for u in out.get('transcript') or []:
        ws = u.get('words') or []
        taken = set()
        for j, w in enumerate(ws):
            for r in recs:
                if r.get('start') is not None and w['start'] < r['end'] and w['end'] > r['start'] and r.get('state') in ('linked', 'almost'):
                    taken.add(j)
        for m in almost(ws, names, taken, cfg, common):
            a, b = m['word_begin'], m['word_end']
            add.append(dict(start=ws[a]['start'], end=ws[b - 1]['end'], tag='entity.mention.place', heard=m['heard'],
                            entity=None, kb_index=-1, state='almost', p=m['p'], candidate=m['candidate'],
                            candidate_kb_index=m['kb_index'], candidate_p=m['p'], src='sound_alike', via='sound_alike'))
    return sorted(recs + add, key=lambda r: (r.get('start', 0), r.get('end', 0)))


# ------------------------------------------------------------------------------------------------------------ end of turn
CONT = frozenset(("the a an to and or but of for with at in on from into onto is are was were my your our his her their "
                  "this that um uh er erm like so because then if than very go set give send turn move drive put make "
                  "what where how who why when which more less by about can could would will should let let's").split())
DONE = frozenset(("yes no yeah yep nope okay ok roger copy affirmative negative done stop thanks thank you please "
                  "correct right wrong").split())


def eot_prob(words, pause_s):
    """the probability that the turn is over: the pause since the last voiced frame (logistic, 0.5 at 0.45 s) times a
    lexical factor (0.35 after a word that needs a continuation: 'the', 'to', 'and', 'um', 'set' ...; a short complete
    answer ('roger', 'copy that', 'yes please') ends sooner: 0.5 at 0.3 s). No words: 0."""
    if not words:
        return 0.0
    k = [_wk(w['text']) for w in words]
    last = k[-1]
    short = len(k) <= 3 and (all(x in DONE for x in k) or (k[-2:] == ['copy', 'that'] and all(x in DONE for x in k[:-1])))
    if short:
        return round(1.0 / (1.0 + math.exp(-(pause_s - 0.3) / 0.08)), 4)
    p = 1.0 / (1.0 + math.exp(-(pause_s - 0.45) / 0.08))
    if last in CONT:
        p *= 0.35
    return round(p, 4)
