"""libarc's names, numbers and word times against arc_audio's, clip by clip.

    python tools/dump_names.py MINI NAMES.json DOMAIN ref.jsonl clips...                                  # arc_audio
    build/arc-transcribe MINI clips... --domain DOMAIN --names NAMES.json --json > cand.jsonl              # libarc
    python tools/compare_names.py ref.jsonl cand.jsonl [ref2.jsonl cand2.jsonl ...]

Checks per clip: the transcript text; every aligned word (text, start, end to 1 ms, inline tag); every name (heard,
entity, state, tag, src, via, words; start / end to 1.5 ms; p, span_p and the pointer head's entity / p to 2e-3) and
number (heard, kind, unit, value, value2, state, kind_head; start / end to 1.5 ms). Exit status 1 on any difference.

Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import json, sys


def close(a, b, tol):
    return (a is None) == (b is None) and (a is None or abs(a - b) <= tol)


def compare(py, c):
    out = []
    ta, tb = [u['text'] for u in py['transcript']], [u['text'] for u in c['transcript']]
    if ta != tb:
        out.append(f'transcript: {ta} vs {tb}')
    for ua, ub in zip(py['transcript'], c['transcript']):
        if len(ua['words']) != len(ub['words']):
            out.append(f'words of {ua["text"]!r}: {len(ua["words"])} vs {len(ub["words"])}')
        for wa, wb in zip(ua['words'], ub['words']):
            if wa['text'] != wb['text'] or not close(wa['start'], wb['start'], 5e-4) or not close(wa['end'], wb['end'], 5e-4) \
                    or wa.get('tag') != wb.get('tag') or bool(wa.get('tag_begin')) != bool(wb.get('tag_begin')):
                out.append(f'word {wa} vs {wb}')
    if len(py['names']) != len(c['names']):
        out.append(f'names: {len(py["names"])} vs {len(c["names"])}')
    for a, b in zip(py['names'], c['names']):
        for k in ('heard', 'entity', 'state', 'tag', 'src', 'via'):
            if a.get(k) != b.get(k):
                out.append(f'name {a.get("heard")!r}: {k} {a.get(k)!r} vs {b.get(k)!r}')
        for k in ('start', 'end'):
            if not close(a[k], b[k], 1.5e-3):
                out.append(f'name {a.get("heard")!r}: {k} {a[k]} vs {b[k]}')
        for k in ('p', 'span_p'):
            if not close(a.get(k), b.get(k), 2e-3):
                out.append(f'name {a.get("heard")!r}: {k} {a.get(k)} vs {b.get(k)}')
        ma, mb = a.get('model'), b.get('model')
        if (ma is None) != (mb is None) or (ma and (ma['entity'] != mb['entity'] or not close(ma['p'], mb['p'], 2e-3))):
            out.append(f'name {a.get("heard")!r}: model {ma} vs {mb}')
        if [w['w'] for w in a.get('words') or []] != [w['w'] for w in b.get('words') or []]:
            out.append(f'name {a.get("heard")!r}: words {a.get("words")} vs {b.get("words")}')
    if len(py['numbers']) != len(c['numbers']):
        out.append(f'numbers: {len(py["numbers"])} vs {len(c["numbers"])}')
    for a, b in zip(py['numbers'], c['numbers']):
        for k in ('heard', 'kind', 'unit', 'value', 'value2', 'state', 'kind_head'):
            if a.get(k) != b.get(k):
                out.append(f'number {a.get("heard")!r}: {k} {a.get(k)!r} vs {b.get(k)!r}')
        for k in ('start', 'end'):
            if not close(a[k], b[k], 1.5e-3):
                out.append(f'number {a.get("heard")!r}: {k} {a[k]} vs {b[k]}')
    return out


def main(argv):
    clips = bad = names = numbers = words = 0
    for rp, cp in zip(argv[0::2], argv[1::2]):
        cand = {}
        for line in open(cp):
            r = json.loads(line)
            cand[r['clip']] = r
        for line in open(rp):
            py = json.loads(line)
            c = cand.get(py['clip'])
            clips += 1
            names += len(py['names'])
            numbers += len(py['numbers'])
            words += sum(len(u['words']) for u in py['transcript'])
            diff = ['missing from the candidate'] if c is None else compare(py, c)
            if diff:
                bad += 1
                print(py['clip'], *diff, sep='\n  ')
    print(f'{clips} clips ({words} words, {names} names, {numbers} numbers): {clips - bad} identical, {bad} with differences')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
