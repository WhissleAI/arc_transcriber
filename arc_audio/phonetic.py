"""Sound-alike matching of heard words against a names list: a phonetic key per spelling and a weighted edit distance
between keys, so "digo laki" lands near "De Gerlache", "vasilev" near "Vavilov", "tiger" near "Tycho".

    key('De Gerlache') -> 'TAKARLAS' ...      similarity('digo laki', 'De Gerlache') -> 0.7x

The key keeps every consonant sound class and collapses vowels to one symbol ('A'); the distance charges less for
near sounds (b/p, d/t, g/k, v/f, s/z, m/n, l/r, vowels) than for unrelated ones. Word breaks are ignored ("degor
lockheed" vs "De Gerlache"). libarc has the same code (c/src/arc_phonetic.c); tests check both on the same pairs.
Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import re

_RULES = [('tch', 'C'), ('sch', 'sk'), ('ph', 'f'), ('gh', 'g'), ('ck', 'k'), ('qu', 'kw'), ('wh', 'w'), ('sh', 'S'),
          ('ch', 'C'), ('th', 'T'), ('dg', 'j'), ('kn', 'n'), ('wr', 'r'), ('x', 'ks'), ('q', 'k')]
_VOW = set('aeiouy')
# classes of near sounds: a substitution inside a class costs 0.4
_NEAR = ['bp', 'dtT', 'gkCj', 'fvw', 'szSCj', 'mn', 'lr', 'A']


def key(text):
    """spelling -> phonetic key (upper / lower ASCII, vowels = 'A'): letters only, digits spelled out, no spaces"""
    from .nft2 import Pieces
    s = re.sub('[^a-z]', '', Pieces.norm(str(text or '')).replace(' ', ''))
    if not s:
        return ''
    for a, b in _RULES:
        s = s.replace(a, b)
    out = []
    n = len(s)
    for i, c in enumerate(s):
        nx = s[i + 1] if i + 1 < n else ''
        if c == 'c':
            c = 's' if nx in ('e', 'i', 'y') else 'k'
        elif c == 'g' and nx in ('e', 'i', 'y') and i > 0:
            c = 'j'
        elif c == 'h':
            continue                                   # h after a consonant / between vowels: (nearly) silent
        elif c == 'e' and i == n - 1 and n > 3:
            continue                                   # silent final e
        elif c == 'y' and i == 0:
            c = 'y'
        elif c in _VOW:
            c = 'A'
        if out and out[-1] == c:
            continue
        out.append(c)
    return ''.join(out)


def _sub(a, b):
    if a == b:
        return 0.0
    for g in _NEAR:
        if a in g and b in g:
            return 0.4
    return 1.0


def _indel(c):
    return 0.5 if c in 'Ayw' else 1.0


def distance(a, b):
    """weighted edit distance between two keys"""
    n, m = len(a), len(b)
    prev = [0.0] * (m + 1)
    for j in range(1, m + 1):
        prev[j] = prev[j - 1] + _indel(b[j - 1])
    for i in range(1, n + 1):
        cur = [prev[0] + _indel(a[i - 1])] + [0.0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + _indel(a[i - 1]), cur[j - 1] + _indel(b[j - 1]), prev[j - 1] + _sub(a[i - 1], b[j - 1]))
        prev = cur
    return prev[m]


def similarity(x, y):
    """1 - distance / the larger key's own weight (0 .. 1); 0 for an empty side"""
    a, b = key(x), key(y)
    if not a or not b:
        return 0.0
    w = max(sum(_indel(c) for c in a), sum(_indel(c) for c in b))
    return max(0.0, 1.0 - distance(a, b) / w)
