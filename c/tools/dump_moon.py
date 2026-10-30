"""Reference data for the C Moonshine port: for each clip, the utterances Arc's builtin transcript decodes (the audio
segment as the decoder sees it, the encoder output, the final beam token ids and text), saved as .npy + one JSON.
  python dump_moon.py ARC_EDGE_DIR LIST.tsv WAV_DIR OUT_DIR [max_clips]

Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import json, os, sys, numpy as np, torch
from arc_audio import Arc, load_audio
from arc_audio import transcribe as T
from arc_audio.moon import generate_kwargs

edge, lst, wavdir, out = sys.argv[1:5]
mx = int(sys.argv[5]) if len(sys.argv) > 5 else 10**9
os.makedirs(out, exist_ok=True)
arc = Arc(edge, device='cpu', threads=4)
rec = []
orig = T.Transcriber.decode_audio

def spy(self, wav, span_types=None, final=False):
    w = wav if len(wav) >= 3200 else np.pad(wav, (0, 3200 - len(wav)))
    x = torch.as_tensor(w)[None].float()
    kw = generate_kwargs(self.m, x)
    with torch.no_grad():
        H = self.m.model.encoder(kw['input_values'], attention_mask=kw['attention_mask']).last_hidden_state[0].numpy()
    txt = orig(self, wav, span_types, final)
    rec.append(dict(wav=np.asarray(w, np.float32), H=H, ids=list(self.last_ids), text=txt, final=bool(final),
                    prompt=self._prompt_ids(), max_len=self._max_len(len(w))))
    return txt

T.Transcriber.decode_audio = spy
meta = []
for i, line in enumerate(open(lst)):
    if i >= mx:
        break
    name = line.split('\t')[0]
    rec.clear()
    utts = arc.transcribe(load_audio(os.path.join(wavdir, name + '.wav')))
    for j, r in enumerate(rec):
        stem = f'{name}_u{j}'
        np.save(os.path.join(out, stem + '.wav.npy'), r['wav'])
        np.save(os.path.join(out, stem + '.H.npy'), r['H'])
        meta.append(dict(stem=stem, clip=name, ids=r['ids'], text=r['text'], final=r['final'], prompt=r['prompt'],
                         max_len=r['max_len'], n=len(r['wav']), utt_text=[u['text'] for u in utts]))
json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
print(len(meta), 'utterances')
