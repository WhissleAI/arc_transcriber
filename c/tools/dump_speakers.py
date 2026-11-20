"""Fixtures for c/tests/test_speaker.c: arc_audio's speakers on multi-speaker clips (two macOS `say` voices and more,
concatenated; written to files, never played).

    python tools/dump_speakers.py MINI_DIR tests/data/speakers.jsonl DOMAIN clip.wav [clip.wav ...]

One JSON line per clip: the per-frame inputs of the diarization as arc_audio's Engine sees them (the speaker index the
slot state machine gives, meta.speech >= 0.5, the slot logits and the diar embedding; float32, base64) and the
diarization arc_audio's predict() returns for that clip in each mode (slots, fused, voice: diarization=...), with the
TrackBank settings of the domain. The C test replays the inputs through arc_spk and compares the turns exactly.

Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import base64, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, '<f4').tobytes()).decode()


def main():
    mini, out, domain = sys.argv[1:4]
    clips = sys.argv[4:]
    import torch
    torch.set_num_threads(2)
    from arc_audio import Arc, load_audio
    from arc_audio import engine as E
    arc = Arc.from_pretrained(mini, backbone='mini', device='cpu', threads=2)
    rec = {}
    orig_bank, orig_dec = E.Engine._bank_step, E.Engine._decode_frame

    def bank(self, f, e, slot_z, is_speech):
        rec.setdefault('emb', {})[f] = np.asarray(e, np.float32)
        rec.setdefault('slot', {})[f] = np.asarray(slot_z, np.float32)
        return orig_bank(self, f, e, slot_z, is_speech)

    def dec(self, f, p, spk, is_speech, bio, start, close):
        rec.setdefault('spk', {})[f] = int(spk)
        rec.setdefault('speech', {})[f] = bool(is_speech)
        return orig_dec(self, f, p, spk, is_speech, bio, start, close)

    E.Engine._bank_step, E.Engine._decode_frame = bank, dec
    fb = arc.cfg['diarization']['fused_bank']
    with open(out, 'w') as fo:
        for c in clips:
            wav = load_audio(c)
            rec.clear()
            r = {}
            for mode in ('voice', 'fused', 'slots'):   # the bank modes record the embedding and slot logits (same in each)
                r[mode] = arc.predict(wav, domain=domain, diarization=mode, events=None, return_frames=False, kb_snap=False)['diarization']
            n = max(rec['spk']) + 1
            assert sorted(rec['spk']) == list(range(n)) and sorted(rec['emb']) == list(range(n))
            line = dict(clip=os.path.basename(c), domain=domain, n_frames=n, n_slot=len(rec['slot'][0]), d_emb=len(rec['emb'][0]),
                        bank=dict(beta=float(fb['beta_by_domain'].get(domain, fb['beta_default'])), tau_new=fb['tau_new'],
                                  tau_upd=fb['tau_upd'], confirm=fb['confirm'], switch=fb['switch'], smooth=fb['smooth']),
                        spk=[rec['spk'][f] for f in range(n)], speech=[int(rec['speech'][f]) for f in range(n)],
                        slot=b64(np.stack([rec['slot'][f] for f in range(n)])), emb=b64(np.stack([rec['emb'][f] for f in range(n)])),
                        diarization={m: [dict(speaker=d['speaker'], start=round(d['start'], 1), end=round(d['end'], 1)) for d in r[m]]
                                     for m in r})
            fo.write(json.dumps(line) + '\n')
            print(c, n, 'frames', {m: len(v) for m, v in r.items()}, 'turns', file=sys.stderr)


if __name__ == '__main__':
    main()
