"""N-best hypotheses from the NFT2 RNNT: a small time-synchronous beam over the same encoder frames as the greedy decode,
with the same names-list boosting (alpha * trie score added to each non-blank token's log p). The greedy decode still writes
the words; the beam only supplies alternatives:

    hyps = beam(rnnt, H, width=4, trie=trie, alpha=2.0)    # [(token ids, score)], best first, distinct token strings
    utt_alts(pieces, hyps, words)  -> [{text, p}]           # other utterance texts, p = softmax over the beam's scores
    word_alts(words, utt_alts)     -> words with 'alts'     # the words other hypotheses have where they differ by one word

Per frame: up to max_sym (2) tokens; the hypotheses that end the frame with blank are merged by token string (max score).
libarc has the same loop (c/src/arc_nft2.c rnnt_beam). Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import difflib

import numpy as np


def _log_softmax(x):
    x = np.asarray(x, np.float64)
    m = x.max(-1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(-1, keepdims=True)))


def beam(rnnt, H, width=4, trie=None, alpha=2.0, max_sym=2):
    """OnnxRNNT + encoder frames H (T, d) -> [(ids tuple, score)] best first (at most width)"""
    H = np.asarray(H.detach().float().cpu().numpy() if hasattr(H, 'detach') else H, np.float32)
    T = H.shape[0]
    if T == 0 or width < 2:
        return []
    Fe = rnnt.jenc.run(None, {'enc': np.ascontiguousarray(H[None])})[0][0]
    blank = rnnt.blank
    g0, hc0 = rnnt._pred(blank, None)
    hyps = [dict(ids=(), s=0.0, g=g0, hc=hc0, b=trie.root if trie is not None else None)]
    for t in range(T):
        A, done = hyps, {}
        for step in range(max_sym + 1):
            cand = []
            for i, a in enumerate(A):
                lp = _log_softmax(rnnt._joint(Fe[t:t + 1], a['g'])[0])
                sb = a['s'] + float(lp[blank])
                d = done.get(a['ids'])
                if d is None or sb > d['s']:
                    done[a['ids']] = dict(a, s=sb)
                if step < max_sym:
                    sc = lp[:blank] + (alpha * trie.scores(a['b']) if trie is not None else 0.0)
                    top = np.argsort(-sc, kind='stable')[:width]
                    cand += [(a['s'] + float(sc[k]), i, int(k)) for k in top]
            if not cand:
                break
            cand.sort(key=lambda x: (-x[0], x[1], x[2]))
            A2 = []
            for s, i, k in cand[:width]:
                g, hc = rnnt._pred(k, A[i]['hc'])
                A2.append(dict(ids=A[i]['ids'] + (k,), s=s, g=g, hc=hc, b=trie.advance(A[i]['b'], k) if trie is not None else None))
            A = A2
        hyps = sorted(done.values(), key=lambda x: (-x['s'], x['ids']))[:width]
    return [(h['ids'], h['s']) for h in hyps]


def _text(pieces, ids):
    return ''.join(pieces.p[k] for k in ids if 0 <= k < len(pieces.p)).replace('▁', ' ').strip()


def _key(text):
    """comparison key: lower case, letters / digits / apostrophes per word (the final words carry case and punctuation)"""
    from .nft2_post import _wk
    return ' '.join(k for k in (_wk(w) for w in str(text).split()) if k)


def utt_alts(pieces, hyps, text, n=None):
    """beam hypotheses -> [{text, p}] that read differently from text (the greedy words), distinct, p = softmax over the beam"""
    if not hyps:
        return []
    s = np.array([x[1] for x in hyps], np.float64)
    p = np.exp(s - s.max())
    p /= p.sum()
    out, seen = [], {_key(text)}
    for (ids, _), q in zip(hyps, p):
        t = ' '.join(_text(pieces, ids).split())
        if _key(t) in seen:
            continue
        seen.add(_key(t))
        out.append(dict(text=t, p=round(float(q), 4)))
    return out[:n] if n else out


def word_alts(words, alts):
    """add each alternative text's word where it replaces exactly one greedy word (or 1-3 words heard in its place) to that
    word's 'alts' (after the runner-up spelling), p = the hypothesis' p"""
    a = [_key(w['text']) for w in words]
    for x in alts:
        b = x['text'].split()
        sm = difflib.SequenceMatcher(None, a, [_key(t) for t in b], autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op != 'replace' or i2 - i1 != 1 or not 1 <= j2 - j1 <= 3:
                continue
            w = words[i1]
            t = ' '.join(b[j1:j2])
            al = w.setdefault('alts', [])
            if _key(t) != a[i1] and all(_key(t) != _key(y['text']) for y in al):
                al.append(dict(text=t, p=x['p']))
    return words
