"""Parity of arc-nft2 (the C library, Arc Transcriber Space) against arc_audio's Python on the same clips.

    python tools/dump_nft2.py NFT2_DIR manifest.json REF_DIR                 # Python: REF_DIR/ref.jsonl + <id>.npz
    build/nft2/arc-nft2 NFT2_DIR --manifest manifest.json --json --dump DUMP > c.jsonl
    python tools/compare_nft2.py REF_DIR c.jsonl DUMP [--md]

Per clip: RNNT token ids, decoder prompt and token ids, transcript words (text, times, tags), names (heard, entity,
state, tag), numbers, events and the device-directed decision must be identical; the float stages (log-mel, encoder
output H, layer tap, decoder memory M, pooled features X / Xt) and probabilities are reported as max abs differences.
Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import argparse
import json
import os
import sys

import numpy as np


def load_dump(d, cid):
    out = {}
    path = os.path.join(d, cid + '.shapes')
    if not os.path.exists(path):
        return out
    for line in open(path):
        name, idx, ty, rows, cols = line.split()
        a = np.fromfile(os.path.join(d, f'{cid}.{name}.{idx}.{ty}'), dtype=np.float32 if ty == 'f32' else np.int32)
        a = a.reshape(int(rows), int(cols)) if ty == 'f32' else a
        out[f'{name}_{idx}' if name not in ('X', 'Xt') else name] = a
    return out


def words_of(t):
    return [(w['text'], round(w['start'], 3), round(w['end'], 3), w.get('tag'), bool(w.get('tag_begin', False)))
            for u in t for w in u['words']]


def name_key(n):
    return (n['heard'], n.get('entity'), n['state'], n['tag'], round(n['start'], 2), round(n['end'], 2))


def num_key(n):
    return (n['heard'], n.get('kind'), n.get('value'), n.get('value2'), n['state'])


def ev_key(e):
    return (e['label'], round(e['start'], 1), round(e['end'], 1))


def maxdiff(a, b):
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return float('inf')
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref')
    ap.add_argument('cand')
    ap.add_argument('dump')
    ap.add_argument('--md', action='store_true', help='a markdown table')
    a = ap.parse_args()
    ref = {}
    for line in open(os.path.join(a.ref, 'ref.jsonl')):
        r = json.loads(line)
        ref[r['id']] = r
    cand = {}
    for line in open(a.cand):
        r = json.loads(line)
        cand[r['id']] = r
    rows, worst, bad = [], {}, 0
    for cid, r in ref.items():
        c = cand.get(cid)
        if c is None:
            rows.append((cid, 'MISSING'))
            bad += 1
            continue
        st = r['stages']
        npz = np.load(os.path.join(a.ref, cid + '.npz'))
        dm = load_dump(a.dump, cid)
        chk = {}
        chk['rnnt'] = all(list(dm.get(f'rnnt_ids_{i}', [])) == u['ids'] and list(dm.get(f'rnnt_frames_{i}', [])) ==
                          u['frames'] for i, u in enumerate(st['rnnt']))
        chk['prompt'] = all(list(dm.get(f'prompt_{i}', [])) == p for i, p in enumerate(st['prompt']))
        chk['dec'] = all(list(dm.get(f'dec_ids_{i}', [])) == u['ids'] for i, u in enumerate(st['dec']))
        chk['dec_text'] = [u.get('decoder_text') for u in c['transcript']] == [u['text'] for u in st['dec']]
        chk['words'] = words_of(r['transcript']) == words_of(c['transcript'])
        chk['text'] = [u['text'] for u in r['transcript']] == [u['text'] for u in c['transcript']]
        chk['names'] = [name_key(n) for n in r['names']] == [name_key(n) for n in c['names']]
        chk['numbers'] = [num_key(n) for n in r['numbers']] == [num_key(n) for n in c['numbers']]
        chk['events'] = [ev_key(e) for e in r['events']] == [ev_key(e) for e in c['events']]
        rdd = [u.get('device_directed', {}).get('directed') for u in r['transcript']]
        cdd = [u.get('device_directed', {}).get('directed') for u in c['transcript']]
        chk['dd'] = rdd == cdd
        diffs = {}
        for k in ('mel', 'H', 'Htap', 'M'):
            v = [maxdiff(npz[f'{k}_{i}'], dm.get(f'{k}_{i}')) for i in range(64) if f'{k}_{i}' in npz.files]
            v = [x for x in v if x is not None]
            diffs[k] = max(v) if v else None
        for k in ('X', 'Xt'):
            diffs[k] = maxdiff(npz[k], dm.get(k)) if k in npz.files else None
        pd = []
        for rn, cn in zip(r['names'], c['names']):
            if rn.get('p') is not None and cn.get('p') is not None:
                pd.append(abs(rn['p'] - cn['p']))
        for ru, cu in zip(r['transcript'], c['transcript']):
            if 'device_directed' in ru and 'device_directed' in cu:
                pd.append(abs(ru['device_directed']['p'] - cu['device_directed']['p']))
        for re_, ce in zip(r['events'], c['events']):
            pd.append(abs(re_['p'] - ce['p']))
        diffs['p'] = max(pd) if pd else 0.0
        for k, v in diffs.items():
            if v is not None:
                worst[k] = max(worst.get(k, 0.0), v)
        ok = all(chk.values())
        bad += not ok
        rows.append((cid, ok, chk, diffs, r['audio_s'], ' | '.join(u['text'] for u in r['transcript'])))
    keys = ['rnnt', 'prompt', 'dec', 'dec_text', 'words', 'text', 'names', 'numbers', 'events', 'dd']
    fk = ['mel', 'H', 'Htap', 'M', 'X', 'Xt', 'p']
    fmt = lambda v: '-' if v is None else f'{v:.1e}'
    if a.md:
        print('| clip | s | ' + ' | '.join(keys) + ' | ' + ' | '.join('Δ' + k for k in fk) + ' |')
        print('|' + '---|' * (2 + len(keys) + len(fk)))
    for row in rows:
        if row[1] == 'MISSING':
            print(row[0], 'MISSING')
            continue
        cid, ok, chk, diffs, secs, txt = row
        if a.md:
            print(f'| {cid} | {secs:.1f} | ' + ' | '.join('=' if chk[k] else '**≠**' for k in keys) + ' | ' +
                  ' | '.join(fmt(diffs[k]) for k in fk) + ' |')
        else:
            fails = [k for k in keys if not chk[k]]
            print(f'{cid:14s} {"OK " if ok else "DIFF"} {",".join(fails):30s} ' +
                  ' '.join(f'{k}={fmt(diffs[k])}' for k in fk))
    print(f'\n{len(rows) - bad}/{len(rows)} clips identical on every discrete output; worst float diffs: ' +
          ', '.join(f'{k} {fmt(v)}' for k, v in worst.items()))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
