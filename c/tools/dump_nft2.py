"""Reference outputs of arc_audio's NFT2 path (Arc Transcriber Space) for tools/compare_nft2.py, stage by stage.

    python tools/dump_nft2.py NFT2_DIR MANIFEST.json OUT_DIR [--threads 4]

NFT2_DIR is the model folder (config.json, adapters.safetensors, model.q4.safetensors, tags.json, decoder/,
encoder_onnx/). MANIFEST.json is a list of clips:

    [{"id": "s1", "wav": "path.wav", "domain": "assistant", "names": [{"name": "Tycho", "kind": "lunar_feature"}],
      "context": {"dom": "space"}}, ...]

(paths relative to the manifest's folder). Each clip runs Arc.predict(wav, domain, transcript="nft2", names,
context) with the ONNX encoder (backbone encoder="onnx"), as the app does. Written to OUT_DIR:

    ref.jsonl     one line per clip: the result (transcript with words / tags / device_directed, names, numbers,
                  events) and per utterance the RNNT token ids and frames, the decoder prompt and token ids, the
                  decoder text and the tag spans on the RNNT words
    <id>.npz      stage arrays: mel_<u>, H_<u>, Htap_<u> (encoder output and layer tap of utterance u), M_<u> (the
                  decoder memory), X, Xt (pooled Arc features, unrounded) for the numeric diffs

`build/nft2/arc-nft2 NFT2_DIR --manifest MANIFEST.json --json --dump DUMP` writes the same from libarc; with
--mel-from DUMP the reference encodes libarc's log-mel frames (see compare_nft2.py); with --stream it runs Arc.stream
(compare with arc-nft2 --stream: the utterances, their RNNT / decoder stages and words). Needs onnxruntime and onnx.
Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import argparse, json, os, sys

import numpy as np


def load_wav(path):
    import soundfile as sf
    w, sr = sf.read(path, dtype='float32', always_2d=True)
    w = w.mean(1)
    if sr != 16000:
        raise SystemExit(f'{path}: {sr} Hz (16 kHz expected)')
    return np.ascontiguousarray(w, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('manifest')
    ap.add_argument('out')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--stream', action='store_true', help='Arc.stream(transcript="nft2") fed 100 ms at a time, then flush, '
                    'instead of Arc.predict (compare with arc-nft2 --stream; only the transcript is comparable)')
    ap.add_argument('--mel-from', help="a DIR of arc-nft2 --dump: feed libarc's log-mel frames to the encoder instead of "
                    "torch's (isolates the front end: everything after it must then match exactly)")
    ap.add_argument('--almost', action='store_true', help="predict(almost=True): sound-alike 'almost' name records")
    ap.add_argument('--nbest', type=int, default=None, help='predict(nbest=N): RNNT beam alternatives')
    ap.add_argument('--no-guard', action='store_true', help='no no-speech / tail guard (transcription.nft2.guard false)')
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    from arc_audio import Arc
    from arc_audio import nft2 as N
    from arc_audio import server as S
    from arc_audio import api as A
    from arc_audio.transcribe import Transcriber

    os.makedirs(a.out, exist_ok=True)
    root = os.path.dirname(os.path.abspath(a.manifest))
    clips = json.load(open(a.manifest))
    model = os.path.abspath(a.model)
    parent, sub = os.path.split(model.rstrip('/'))
    os.environ['ARC_NFT2_ENCODER'] = 'onnx'
    if a.no_guard:
        _init = N.Nft2Finals.__init__
        def init(self, *x, **k):
            _init(self, *x, **k)
            self.c['guard'] = False
        N.Nft2Finals.__init__ = init
    os.environ['ARC_NFT2_ONNX_THREADS'] = str(a.threads)
    arc = Arc(model, device='cpu', threads=a.threads) if os.path.exists(os.path.join(model, 'config.json')) else \
        Arc.from_pretrained(model, backbone='nft2', device='cpu', threads=a.threads)

    cap, cur = {}, {}
    enc_mel = N.OnnxEncoder.encode_mel

    def encode_mel(self, M):
        if a.mel_from:   # libarc's frames for this encoder input: same shape, closest (torch and libarc differ by < 1e-3)
            best = None
            for f in sorted(os.listdir(a.mel_from)):
                if f.startswith(cur['id'] + '.mel.') and f.endswith('.f32'):
                    Mc = np.fromfile(os.path.join(a.mel_from, f), np.float32)
                    if Mc.size == M.size:
                        d = float(np.abs(Mc.reshape(M.shape) - M).max())
                        if d < 1e-2 and (best is None or d < best[0]):
                            best = (d, Mc.reshape(M.shape))
            if best is not None:
                M = best[1]
        H, Ht = enc_mel(self, M)
        cap.setdefault('enc', []).append((np.array(M), H, Ht))
        return H, Ht
    N.OnnxEncoder.encode_mel = encode_mel
    utt = N.Nft2Backbone.utterance

    def utterance(self, wav, trie=None, alpha=2.0, t0=0.0, want_tap=False):
        r = utt(self, wav, trie=trie, alpha=alpha, t0=t0, want_tap=want_tap)
        cap.setdefault('rnnt', []).append(dict(ids=[int(x) for x in r['ids']], frames=[int(x) for x in r['frames']], n=int(r['n']),
                                               words=[dict(w) for w in r['words']]))
        return r
    N.Nft2Backbone.utterance = utterance
    mwp = S.mem_with_pos

    def mem_with_pos(heads, pos_emb, P):
        M = mwp(heads, pos_emb, P)
        cap.setdefault('mem', []).append(M[0].detach().float().cpu().numpy())
        return M
    S.mem_with_pos = mem_with_pos
    pids = Transcriber._prompt_ids

    def prompt_ids(self):
        P = pids(self)
        cap.setdefault('prompt', []).append([int(x) for x in P])
        return P
    Transcriber._prompt_ids = prompt_ids
    dec = N.Nft2Finals.decode

    def decode(self, y, t0=0.0, f0=None, f1=None, want=False, tail=False):
        u = dec(self, y, t0, f0, f1, want=True, tail=tail)
        cap.setdefault('dec', []).append(dict(ids=[int(x) for x in u.pop('_dec_ids')], text=u['decoder_text'],
                                              spans=[list(s) for s in u['tag_spans']], t0=float(t0), n=len(y)))
        u.pop('_rnnt_ids', None)
        u.pop('_P', None)
        return u
    N.Nft2Finals.decode = decode
    feats = A.Arc._feats

    def _feats(self, wav, tap=False):
        x, xe, xt = feats(self, wav, tap)
        cap['X'] = np.array(xe, np.float32)
        cap['Xt'] = None if xt is None else np.array(xt, np.float32)
        return x, xe, xt
    A.Arc._feats = _feats

    with open(os.path.join(a.out, 'ref.jsonl'), 'w') as f:
        for c in clips:
            cap.clear()
            cur['id'] = c['id']
            w = load_wav(os.path.join(root, c['wav']))
            if a.stream:
                st = arc.stream(domain=c.get('domain', 'assistant'), transcript='nft2', names=c.get('names') or None,
                                context=c.get('context') or None, background=False, partials=False)
                outs = [st.feed(w[i:i + 1600]) for i in range(0, len(w), 1600)] + [st.flush()]
                st.close()
                out = {k: [x for o in outs for x in (o.get(k) or [])] for k in ('transcript', 'names', 'numbers', 'audio_events')}
            else:
                out = arc.predict(w, domain=c.get('domain', 'assistant'), transcript='nft2', names=c.get('names') or None,
                                  context=c.get('context') or None, return_frames=False, almost=a.almost, nbest=a.nbest)
            ww = lambda x: {k: x.get(k) for k in ('text', 'start', 'end', 'tag', 'tag_begin', 'conf', 'disfl', 'alts') if k in x}
            tr = [dict(start=u['start'], end=u['end'], text=u['text'], words=[ww(x) for x in u.get('words') or []], tags=u.get('tags') or [],
                       device_directed=u.get('device_directed'), conf=u.get('conf'), clean_text=u.get('clean_text'),
                       n_dropped=u.get('n_dropped'), quantities=u.get('quantities') or [], alts=u.get('alts') or []) for u in out['transcript']]
            nk = ('tag', 'heard', 'entity', 'state', 'p', 'src', 'via', 'start', 'end', 'span_p', 'snap_score', 'decoder_tag',
                  'candidate', 'candidate_kb_index', 'candidate_p')
            names = [dict({k: n.get(k) for k in nk}, model=n.get('model'), words=n.get('words')) for n in out.get('names') or []]
            stages = dict(rnnt=cap.get('rnnt', []), prompt=cap.get('prompt', []), dec=cap.get('dec', []))
            rec = dict(id=c['id'], audio_s=round(len(w) / 16000, 3), transcript=tr, names=names, numbers=out.get('numbers') or [],
                       events=[{k: e.get(k) for k in ('label', 'start', 'end', 'p')} for e in out.get('audio_events') or []], stages=stages)
            f.write(json.dumps(rec, default=float) + '\n')
            f.flush()
            arr = {}
            for i, (M, H, Ht) in enumerate(cap.get('enc', [])):
                arr[f'mel_{i}'], arr[f'H_{i}'] = M.astype(np.float32), H.astype(np.float32)
                if Ht is not None:
                    arr[f'Htap_{i}'] = Ht.astype(np.float32)
            for i, M in enumerate(cap.get('mem', [])):
                arr[f'M_{i}'] = M.astype(np.float32)
            if cap.get('X') is not None:
                arr['X'] = cap['X']
            if cap.get('Xt') is not None:
                arr['Xt'] = cap['Xt']
            np.savez(os.path.join(a.out, f"{c['id']}.npz"), **arr)
            print(c['id'], ' | '.join(u['text'] for u in tr), '|', [(n['heard'], n['entity'], n['state']) for n in names], file=sys.stderr)


if __name__ == '__main__':
    main()
