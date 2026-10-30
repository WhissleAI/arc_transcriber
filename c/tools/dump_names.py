"""Reference names / numbers / word timestamps from arc_audio's Arc.predict, for tools/compare_names.py.

    python tools/dump_names.py MINI_DIR NAMES.json DOMAIN OUT.jsonl clip.wav [clip.wav ...]

One JSON line per clip: the transcript (text, aligned words with inline tags), the linked names and the numbers, as
predict(audio, domain=DOMAIN, names=NAMES) returns them (events off, frames off). NAMES.json is a list of names
(strings or {"name", "kind", "group", "aliases"}), as examples/example_names.json.

Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import json, os, sys


def main():
    mini, names_path, domain, out = sys.argv[1:5]
    clips = sys.argv[5:]
    import torch
    torch.set_num_threads(4)
    from arc_audio import Arc, load_audio
    arc = Arc.from_pretrained(mini, backbone='mini', device='cpu', threads=4)
    names = json.load(open(names_path))
    with open(out, 'w') as f:
        for c in clips:
            r = arc.predict(load_audio(c), domain=domain, names=names, events=None, return_frames=False)
            keep_w = lambda w: dict(text=w.get('text', w.get('w')), start=w['start'], end=w['end'], **({'tag': w['tag'], 'tag_begin': w.get('tag_begin', False)} if w.get('tag') else {}))
            rec = dict(clip=os.path.basename(c), domain=domain,
                       transcript=[dict(start=u['start'], end=u['end'], text=u['text'], words=[keep_w(w) for w in u.get('words') or []],
                                        tags=[(t['tag'], t['text']) for t in u.get('tags', [])]) for u in r['transcript']],
                       names=r.get('names') or [], numbers=r.get('numbers') or [],
                       utterances=[dict(start=u['start'], end=u['end'], groups=u.get('groups'), tags=u.get('tags')) for u in r.get('utterances') or []])
            f.write(json.dumps(rec, default=float) + '\n')
            f.flush()
            print(c, len(rec['names']), 'names', len(rec['numbers']), 'numbers', file=sys.stderr)


if __name__ == '__main__':
    main()
