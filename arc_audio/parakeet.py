import queue, threading
import numpy as np
MODEL_ID = 'nvidia/parakeet-tdt-0.6b-v2'
SR = 16000

class ParakeetFinals:

    def __init__(self, device='cpu', background=False, model_id=MODEL_ID, backend=None):
        self.backend = backend or self._nemo(device, model_id)
        self.q, self.res = (queue.Queue(), queue.Queue())
        self.th = None
        if background:
            self.th = threading.Thread(target=self._loop, daemon=True)
            self.th.start()

    @staticmethod
    def _nemo(device, model_id):
        try:
            import logging
            logging.getLogger('nemo_logger').setLevel(logging.ERROR)
            import torch
            import nemo.collections.asr as nemo_asr
        except Exception as e:
            raise RuntimeError("transcript='parakeet' needs NeMo: pip install 'arc-audio[parakeet]'") from e
        m = nemo_asr.models.ASRModel.from_pretrained(model_id, map_location='cpu')
        m = (m.half().cuda() if str(device).startswith('cuda') else m).eval()

        def run(y):
            with torch.inference_mode():
                h = m.transcribe([np.asarray(y, np.float32)], batch_size=1, timestamps=True, verbose=False)
            h = h[0] if isinstance(h, tuple) else h
            ws = (h[0].timestamp or {}).get('word', []) if hasattr(h[0], 'timestamp') else []
            return [dict(word=w['word'], start=float(w['start']), end=float(w['end'])) for w in ws]
        return run

    def _decode(self, y, t0, f0, f1):
        ws = self.backend(y)
        words = [dict(text=w['word'], start=round(t0 + w['start'], 3), end=round(t0 + max(w['end'], w['start'] + 0.04), 3), is_final=True) for w in ws]
        return dict(start=round(f0 / 10, 1), end=round((f1 + 1) / 10, 1), f0=f0, f1=f1, text=' '.join((w['text'] for w in words)), words=words, final=True, source='parakeet')

    def _loop(self):
        while True:
            job = self.q.get()
            if job is None:
                return
            self.res.put(self._decode(*job))

    def submit(self, y, t0, f0, f1):
        if self.th is None:
            self.res.put(self._decode(y, t0, f0, f1))
        else:
            self.q.put((y, t0, f0, f1))

    def poll(self, words):
        out = []
        while True:
            try:
                u = self.res.get_nowait()
            except queue.Empty:
                break
            words.add(u['words'], 'parakeet')
            words.mark_covered(u['end'])
            out.append(u)
        return out

    def drain(self, words):
        if self.th is not None:
            self.q.put(None)
            self.th.join()
        return self.poll(words)

    def transcribe_offline(self, wav, arc):
        from .transcribe import UttSegmenter
        from .api import frame_db, FR
        seg = UttSegmenter(arc.cfg.get('transcription', {}))
        closed = []
        n = (len(wav) + FR - 1) // FR
        for f in range(n):
            closed += seg.feed(f, frame_db(wav[f * FR:(f + 1) * FR]))
        closed += seg.flush()
        out = []
        for f0, f1, _ in closed:
            a = max(0, f0 * FR - 2400)
            b = min(len(wav), (f1 + 1) * FR + 2400)
            out.append(self._decode(wav[a:b], a / SR, f0, f1))
        return out
