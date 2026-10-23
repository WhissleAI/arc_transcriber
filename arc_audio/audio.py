import shutil, subprocess
import numpy as np
SR = 16000

def resample(x, sr_in, sr_out=SR):
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    try:
        import torch, torchaudio.functional as AF
        return AF.resample(torch.from_numpy(np.ascontiguousarray(x, np.float32)), sr_in, sr_out).numpy()
    except Exception:
        pass
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr_in), int(sr_out))
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    except Exception:
        import librosa
        return librosa.resample(x.astype(np.float32), orig_sr=sr_in, target_sr=sr_out)

def to_mono(x):
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.mean(1) if x.shape[0] > x.shape[1] else x.mean(0)
    if x.dtype.kind in 'iu':
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    return x.astype(np.float32, copy=False)

def _ffmpeg(path):
    if not shutil.which('ffmpeg'):
        raise RuntimeError(f'cannot decode {path}: soundfile failed and ffmpeg is not on PATH')
    raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()

def load_audio(src, sr=None):
    if isinstance(src, (str, bytes)) or hasattr(src, '__fspath__'):
        try:
            import soundfile as sf
            x, sr_in = sf.read(str(src), dtype='float32', always_2d=False)
        except Exception:
            return _ffmpeg(src)
        return resample(to_mono(x), sr_in)
    x = to_mono(src)
    return resample(x, sr or SR)
