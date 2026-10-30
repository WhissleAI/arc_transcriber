"""Reference transcripts: Arc.transcribe() per clip (utterance times, text, tag spans), without and with a context
(names + topic). JSON lines: {"clip", "ctx", "utts": [[start, end, text, [[tag, a, b], ...]], ...]}.

Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import json, sys
from arc_audio import Arc, load_audio
edge, lst, wavdir, out = sys.argv[1:5]
arc = Arc(edge, device='cpu', threads=4)
names = [{'name': 'Priya Raman', 'kind': 'person'}, {'name': 'Doctor Okafor', 'kind': 'person'}, {'name': 'Whissle', 'kind': 'organization'},
         {'name': 'Elizabeth Bennet', 'kind': 'person'}, {'name': 'Mr. Darcy', 'kind': 'person'}]
with open(out, 'w') as f:
    for line in open(lst):
        name = line.split('\t')[0]
        wav = load_audio(f'{wavdir}/{name}.wav')
        for ctx in (0, 1):
            if ctx and not name.startswith('p'):
                continue
            if ctx:
                arc.set_context(names=names, terms=['Arc', 'Pride and Prejudice'], topic='assistant')
            else:
                arc._ctx = None
            utts = arc.transcribe(wav)
            f.write(json.dumps(dict(clip=name, ctx=ctx, utts=[[u['start'], u['end'], u['text'], [list(s) for s in u.get('tag_spans', [])]] for u in utts])) + '\n')
