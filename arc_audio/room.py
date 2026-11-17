"""Room read-outs for Arc Transcriber Space (backbone nft2): one small network on the encoder's Arc frames plus the microphone audio
and, when given, the device's own playback (reference) audio.

Per 80 ms frame: app_playback, media_playback, live_speech, end_of_turn, room_tone, addressee (device / other_person /
reading_aloud), safety evidence. Per utterance: act (question / command / confirmation / chatter / statement), addressee, safety.

BETA outputs (config.json room.<output>.beta): addressee and safety. Read their notes (Room.notes) before using them.
The safety output is not a safety system: never use it as the only check or for any safety-critical decision."""
import os

import numpy as np

SR = 16000


class Room:

    def __init__(self, path, cfg, threads=None):
        import onnxruntime as ort
        from .nft2 import _no_telemetry
        _no_telemetry(ort)
        self.cfg = cfg
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads or cfg.get('threads') or 1)
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(os.path.join(path, cfg.get('file', 'room.onnx')), so, providers=['CPUExecutionProvider'])
        self.frame_s = float(cfg.get('frame_s', 0.08))
        self.frames = list(cfg['frames'])
        self.addr = list(cfg['addressee']['labels'])
        self.act = list(cfg['act']['labels'])
        self.safety = list(cfg['safety']['labels'])
        self.safety_frames = list(cfg['safety'].get('frame_labels') or [])
        self.thr = dict(cfg['safety'].get('thresholds') or {})
        self.beta = sorted(k for k, v in cfg.items() if isinstance(v, dict) and v.get('beta'))
        self.notes = {k: v.get('note') for k, v in cfg.items() if isinstance(v, dict) and v.get('note')}

    def run(self, tap, mic, ref=None):
        """tap (T, d) Arc frames of `mic` (drop0, 80 ms), mic / ref float32 16 kHz (ref aligned sample for sample, or None)
        -> dict of numpy arrays: frames (T, 5), addressee (T, 3), act (5,), safety (15,), safety_frames (T, 14)"""
        tap = np.asarray(tap, np.float32)
        mic = np.asarray(mic, np.float32).reshape(-1)
        n = max(len(mic), 401)
        m = np.zeros(n, np.float32)
        m[:len(mic)] = mic
        r = np.zeros(n, np.float32)
        on = 0.0
        if ref is not None:
            ref = np.asarray(ref, np.float32).reshape(-1)[:n]
            r[:len(ref)] = ref
            on = 1.0 if np.any(ref) else 0.0
        o = self.sess.run(None, dict(tap=tap[None], mic=m[None], ref=r[None], ref_on=np.array([on], np.float32)))
        return dict(frames=o[0][0], addressee=o[1][0], act=o[2][0], safety=o[3][0], safety_frames=o[4][0], ref_on=bool(on))

    def summary(self, raw, t0=0.0, frames=False):
        """utterance read-out of run()'s arrays; frames=True adds the per-frame tracks (t = frame centre time in seconds)"""
        F = raw['frames']
        T = len(F)
        live = F[:, self.frames.index('live_speech')] >= 0.5 if T else np.zeros(0, bool)
        A = raw['addressee'][live] if live.any() else raw['addressee']
        la = np.log(np.clip(A, 1e-9, 1.0)).mean(0) if len(A) else np.log(np.full(len(self.addr), 1.0 / len(self.addr)))
        pa = np.exp(la - la.max())
        pa = pa / pa.sum()
        act = raw['act']
        su = raw['safety']
        k = self.frames.index('end_of_turn')
        out = dict(
            act=dict(label=self.act[int(act.argmax())], p=round(float(act.max()), 4), probs={a: round(float(p), 4) for a, p in zip(self.act, act)}),
            addressee=dict(label=self.addr[int(pa.argmax())], p=round(float(pa.max()), 4), probs={a: round(float(p), 4) for a, p in zip(self.addr, pa)}, beta=True),
            safety=dict(above_threshold=[s for s, p in zip(self.safety, su) if p >= self.thr.get(s, 0.5)],
                        probs={s: round(float(p), 4) for s, p in zip(self.safety, su)}, beta=True),
            end_of_turn=round(float(F[-6:, k].max()), 4) if T else None,
            playback=round(float((F[:, self.frames.index('app_playback')] >= 0.5).mean()), 4) if T else None,
            media=round(float((F[:, self.frames.index('media_playback')] >= 0.5).mean()), 4) if T else None,
            reference=raw.get('ref_on', False))
        if frames:
            out['frames'] = dict(t=[round(t0 + (i + 1.5) * self.frame_s, 3) for i in range(T)], labels=self.frames,
                                 probs=np.round(F, 4).tolist(), addressee=np.round(raw['addressee'], 4).tolist(),
                                 safety_labels=self.safety_frames, safety=np.round(raw['safety_frames'], 4).tolist())
        return out


def load_room(path, cfg, threads=None):
    """config.json 'room' section -> Room, or None when the package has none"""
    rc = (cfg or {}).get('room')
    if not rc or not os.path.exists(os.path.join(path, rc.get('file', 'room.onnx'))):
        return None
    return Room(path, rc, threads)

