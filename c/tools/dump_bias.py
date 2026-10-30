"""Reference transcripts with name biasing: Arc.transcribe(wav, names=NAMES, bias_names=True) with the same names as
the decoder context. JSON lines as dump_transcribe.py, ctx = 2.

Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import json, sys
from arc_audio import Arc, load_audio
edge, lst, wavdir, out = sys.argv[1:5]
arc = Arc(edge, device='cpu', threads=4)
names = [{'name': n, 'kind': 'product'} for n in ['WhatsApp', 'Slack', 'Messages', 'Telegram', 'Mail', 'Safari']] + \
        [{'name': 'Priya Raman', 'kind': 'person'}, {'name': 'Doctor Okafor', 'kind': 'person'}, {'name': 'Mr. Darcy', 'kind': 'person'}]
arc.set_context(names=names, terms=['Arc'], topic='assistant')
with open(out, 'w') as f:
    for line in open(lst):
        name = line.split('\t')[0]
        utts = arc.transcribe(load_audio(f'{wavdir}/{name}.wav'), names=names, bias_names=True)
        f.write(json.dumps(dict(clip=name, ctx=2, utts=[[u['start'], u['end'], u['text'], [list(s) for s in u.get('tag_spans', [])]] for u in utts])) + '\n')
