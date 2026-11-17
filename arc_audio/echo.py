"""The device's own playback in the microphone (echo / self-speech): a partitioned-block frequency-domain NLMS echo canceller
(overlap-save), a per-utterance playback_leak score, and the barge-in rule.

    ec = EchoCanceller()                    # 16 kHz; blocks of 256 samples (16 ms), 32 partitions = 512 ms of echo path
    ec.feed_ref(pcm, t)                     # what the device plays, t = the stream time (s, mic clock) of its first sample
    ec.feed_mic(pcm)                        # mic audio in arrival order; the residual (mic - echo estimate) grows with it
    ec.leak(t0, t1)                         # share of the mic energy in [t0, t1) the playback explains (0..1); None: nothing played

The reference is aligned to the mic clock by the caller (t); the filter learns 0..512 ms of further delay (the speaker, the room,
audio buffers). Each 256-sample block: the filter output is the overlap-save convolution of the reference with 32 partitions of
256 taps (FFT 512), the error is the mic minus it (the residual, exact in time), and the partitions are updated by normalised
LMS (step 0.03, normalised by the mean reference power per bin over the partitions) with the gradient constraint (each
partition's taps stay 256 long); once it has seen 1 s of playback, a block whose residual is more than twice the echo estimate
(double talk: someone speaking over the playback) does not update it. leak = 1 - E(residual) / E(mic) over the blocks with playback (reference above -55 dBFS); blocks without
playback count as unexplained. libarc has the same canceller (c/src/arc_echo.c; arc_nft2_stream_feed_ref).
Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import numpy as np

SR = 16000
ECHO = dict(block=256, parts=32, mu=0.03, ref_db=-55.0, eps=1e-6, threshold=0.7, dt_ratio=2.0, dt_after_s=1.0)


class EchoCanceller:
    def __init__(self, cfg=None):
        self.c = dict(ECHO, **(cfg or {}))
        B, P = self.c['block'], self.c['parts']
        self.K = B + 1
        self.W = np.zeros((P, self.K), np.complex128)
        self.X = np.zeros((P, self.K), np.complex128)    # reference spectra of the last P blocks, newest first
        self.ref = np.zeros(0, np.float32)                # reference on the mic clock (sample index = mic sample index)
        self.pend = np.zeros(0, np.float32)              # mic samples not yet in a block
        self.n_mic = 0
        self._res, self._res_cat = [], None               # residual blocks
        self.nb = 0                                       # blocks done (block j = samples [j B, j B + B))
        self.e_mic, self.e_res, self.e_ref = [], [], []
        self.n_on = self.n_dt = 0                         # blocks with playback, blocks with double talk
        self.any_ref = False

    def feed_ref(self, pcm, t=None):
        """reference audio; t (s): where its first sample sits on the mic clock (None: right after the last reference sample)"""
        x = np.asarray(pcm, np.float32).reshape(-1)
        a = len(self.ref) if t is None else max(0, int(round(float(t) * SR)))
        if a + len(x) > len(self.ref):
            self.ref = np.concatenate([self.ref, np.zeros(a + len(x) - len(self.ref), np.float32)])
        self.ref[a:a + len(x)] += x
        if len(x) and float(np.max(np.abs(x))) > 0:
            self.any_ref = True

    def feed_mic(self, pcm):
        """mic audio -> the residual of the blocks it completed"""
        x = np.asarray(pcm, np.float32).reshape(-1)
        self.pend = np.concatenate([self.pend, x])
        self.n_mic += len(x)
        B = self.c['block']
        out = []
        while len(self.pend) >= B:
            out.append(self._block(self.pend[:B]))
            self.pend = self.pend[B:]
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def flush(self):
        """the last partial block (zero padded); the residual is then as long as the mic"""
        B = self.c['block']
        if len(self.pend):
            m = len(self.pend)
            e = self._block(np.concatenate([self.pend, np.zeros(B - m, np.float32)]))
            self._res[-1] = e[:m]
            self.pend = np.zeros(0, np.float32)
        return self.residual()

    def _ref(self, a, b):
        r = np.zeros(b - a, np.float64)
        lo, hi = max(a, 0), min(b, len(self.ref))
        if hi > lo:
            r[lo - a:hi - a] = self.ref[lo:hi]
        return r

    def _block(self, mic):
        c = self.c
        B, P = c['block'], c['parts']
        a = self.nb * B
        d = mic.astype(np.float64)
        x = self._ref(a - B, a + B)
        self.X = np.concatenate([np.fft.rfft(x)[None], self.X[:-1]], 0)
        y = np.fft.irfft((self.W * self.X).sum(0), 2 * B)[B:]
        e = d - y
        er, ey = float(np.sum(e * e)), float(np.sum(y * y))
        thr = 10 ** (c['ref_db'] / 10)
        if self.n_on * B / SR >= c['dt_after_s'] and er > c['dt_ratio'] * ey:
            self.n_dt += 1                                # double talk (speech that is not the playback): no update
        else:
            E = np.fft.rfft(np.concatenate([np.zeros(B), e]))
            Pw = (np.abs(self.X) ** 2).sum(0) / P
            g = np.fft.irfft(c['mu'] * np.conj(self.X) * (E / (Pw + c['eps'] * 2 * B))[None], 2 * B, axis=1)
            g[:, B:] = 0.0
            self.W += np.fft.rfft(g, axis=1)
        self.e_mic.append(float(np.sum(d * d)))
        self.e_res.append(er)
        self.e_ref.append(float(np.mean(x[B:] ** 2)))
        self.n_on += self.e_ref[-1] > thr
        self.nb += 1
        e = e.astype(np.float32)
        self._res.append(e)
        self._res_cat = None
        return e

    def residual(self):
        """the residual so far, sample-aligned with the mic"""
        if self._res_cat is None or len(self._res) > 1:
            self._res_cat = np.concatenate(self._res) if self._res else np.zeros(0, np.float32)
            self._res = [self._res_cat] if len(self._res_cat) else []
        return self._res_cat

    def leak(self, t0, t1):
        """playback_leak over [t0, t1) s: 1 - E(res) / E(mic) over the blocks with playback; 0 when nothing in it was played;
        None when no reference was ever fed"""
        if not self.any_ref:
            return None
        B = self.c['block']
        j0 = max(0, int(np.floor(t0 * SR / B)))
        j1 = min(self.nb, int(np.ceil(t1 * SR / B)))
        if j1 <= j0:
            return 0.0
        thr = 10 ** (self.c['ref_db'] / 10)
        em = np.array(self.e_mic[j0:j1])
        er = np.array(self.e_res[j0:j1])
        on = np.array(self.e_ref[j0:j1]) > thr
        tot = float(em.sum())
        if tot <= 0 or not on.any():
            return 0.0
        explained = float(np.maximum(em[on] - er[on], 0.0).sum())
        return round(min(1.0, max(0.0, explained / tot)), 4)

    def playing(self, t, span=0.3):
        """1 when the reference has playback in [t - span, t)"""
        a, b = max(0, int((t - span) * SR)), max(0, int(t * SR))
        r = self.ref[a:b]
        return bool(len(r)) and float(np.mean(r.astype(np.float64) ** 2)) > 10 ** (self.c['ref_db'] / 10)

    def trained_s(self):
        """seconds of playback the filter has seen"""
        return self.n_on * self.c['block'] / SR


def barge_in(ec, t_open, t_now, min_s=0.2, max_leak=0.5, trained_s=1.0, win_s=0.3):
    """the barge-in rule: an utterance open for >= min_s while the device plays, whose last win_s (or since it opened) the
    playback does not explain (leak < max_leak), once the canceller has seen trained_s of playback"""
    if t_now - t_open < min_s or ec.trained_s() < trained_s or not ec.playing(t_now, max(min_s, min(win_s, t_now - t_open))):
        return False
    lk = ec.leak(max(t_open, t_now - win_s), t_now)
    return lk is not None and lk < max_leak
