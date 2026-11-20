"""libarc's names, numbers and word times against arc_audio's, clip by clip.

    python tools/dump_names.py MINI NAMES.json DOMAIN ref.jsonl clips...                                  # arc_audio
    build/arc-transcribe MINI clips... --domain DOMAIN --names NAMES.json --json > cand.jsonl              # libarc
    python tools/compare_names.py [--known-alias-index] ref.jsonl cand.jsonl [ref2.jsonl cand2.jsonl ...]

Checks per clip: the transcript text; every aligned word (text, start, end to 1 ms, inline tag); every name (heard,
entity, state, tag, src, via, words; start / end to 1.5 ms; p, span_p and the pointer head's entity / p to 2e-3) and
number (heard, kind, unit, value, value2, state, kind_head, date, iso; start / end to 1.5 ms); and, when both sides carry
them, the speakers (ids, each group's label, tags; p to 2e-3, last_seen to 0.05 s), the diarization turns (speaker; start /
end to 0.05 s, i.e. the same 100 ms frame) and each utterance's speaker. Exit status 1 on any difference.

--known-alias-index: a KNOWN gap, not a pass. arc_audio's last list check before a name stays unlinked (engine.py
_name_extras with kbsnap.AliasIndex: a heard part of only one entry's name, "Daniel" for "Daniel Brooks", links with
via "partial"; a punctuation / spacing variant with via "alias") is not ported to libarc. With the flag, a name that
arc_audio linked that way and libarc left not_in_list / abstain (same heard text and times) is counted as KNOWN instead
of a difference.

Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import json, sys


def close(a, b, tol):
    return (a is None) == (b is None) and (a is None or abs(a - b) <= tol)


KNOWN_ALIAS = False
known = []


def alias_gap(a, b):
    """arc_audio linked a by its AliasIndex; libarc left b unlinked (the gap --known-alias-index carries)"""
    return (KNOWN_ALIAS and a.get('via') in ('partial', 'alias') and a.get('state') == 'linked' and b.get('state') in ('not_in_list', 'abstain')
            and a.get('heard') == b.get('heard') and close(a['start'], b['start'], 1.5e-3) and close(a['end'], b['end'], 1.5e-3))


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
        if alias_gap(a, b):
            known.append(f'{py["clip"]}: {a["heard"]!r} -> {a["entity"]!r} via {a["via"]} (libarc: {b["state"]})')
            continue
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
        for k in ('heard', 'kind', 'unit', 'value', 'value2', 'state', 'kind_head', 'date', 'iso'):
            if a.get(k) != b.get(k):
                out.append(f'number {a.get("heard")!r}: {k} {a.get(k)!r} vs {b.get(k)!r}')
        for k in ('start', 'end'):
            if not close(a[k], b[k], 1.5e-3):
                out.append(f'number {a.get("heard")!r}: {k} {a[k]} vs {b[k]}')
    out += compare_speakers(py, c)
    return out


def compare_speakers(py, c):
    """speakers / diarization / per-utterance speaker (only when both records carry them)"""
    out = []
    if 'speakers' in py and 'speakers' in c:
        sa, sb = py['speakers'] or {}, c['speakers'] or {}
        if sorted(sa) != sorted(sb):
            out.append(f'speakers: {sorted(sa)} vs {sorted(sb)}')
        for k in sorted(set(sa) & set(sb)):
            a, b = sa[k], sb[k]
            if not close(a.get('last_seen'), b.get('last_seen'), 0.05):
                out.append(f'speaker {k}: last_seen {a.get("last_seen")} vs {b.get("last_seen")}')
            ga, gb = a.get('groups') or {}, b.get('groups') or {}
            if sorted(ga) != sorted(gb):
                out.append(f'speaker {k}: groups {sorted(ga)} vs {sorted(gb)}')
            for g in sorted(set(ga) & set(gb)):
                if ga[g]['label'] != gb[g]['label'] or not close(ga[g]['p'], gb[g]['p'], 2e-3):
                    out.append(f'speaker {k}: {g} {ga[g]} vs {gb[g]}')
            ta, tb = a.get('tags') or {}, b.get('tags') or {}
            if sorted(ta) != sorted(tb) or any(not close(ta[t], tb[t], 2e-3) for t in ta if t in tb):
                out.append(f'speaker {k}: tags {ta} vs {tb}')
    if 'diarization' in py and 'diarization' in c:
        da, db = py['diarization'] or [], c['diarization'] or []
        if len(da) != len(db):
            out.append(f'diarization: {len(da)} vs {len(db)} turns')
        for a, b in zip(da, db):
            if a['speaker'] != b['speaker'] or not close(a['start'], b['start'], 0.05) or not close(a['end'], b['end'], 0.05):
                out.append(f'turn {a} vs {b}')
    for ua, ub in zip(py['transcript'], c['transcript']):
        if 'speaker' in ua and 'speaker' in ub and ua['speaker'] != ub['speaker']:
            out.append(f'speaker of {ua["text"]!r}: {ua["speaker"]} vs {ub["speaker"]}')
    return out


def main(argv):
    global KNOWN_ALIAS
    if argv and argv[0] == '--known-alias-index':
        KNOWN_ALIAS, argv = True, argv[1:]
    clips = bad = names = numbers = words = turns = speakers = 0
    pmax = 0.0
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
            if c is not None and 'speakers' in c and 'speakers' in py:
                turns += len(py.get('diarization') or [])
                speakers += len(py['speakers'] or {})
                for k, a in (py['speakers'] or {}).items():
                    for g, v in (a.get('groups') or {}).items():
                        w = ((c['speakers'] or {}).get(k) or {}).get('groups', {}).get(g)
                        if w and w['label'] == v['label']:
                            pmax = max(pmax, abs(w['p'] - v['p']))
            diff = ['missing from the candidate'] if c is None else compare(py, c)
            if diff:
                bad += 1
                print(py['clip'], *diff, sep='\n  ')
    print(f'{clips} clips ({words} words, {names} names, {numbers} numbers, {speakers} speakers, {turns} diarization turns): '
          f'{clips - bad} identical, {bad} with differences' + (f'; speaker read-out p max |diff| {pmax:.4f}' if speakers else '')
          + (f'; KNOWN alias-index gap: {len(known)} name(s)' if known else ''))
    for k in known:
        print('  KNOWN', k)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
