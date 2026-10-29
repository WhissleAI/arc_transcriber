import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
LABELS = ['scene.cheer', 'scene.applause', 'scene.crowd_ambience', 'scene.boo_jeer', 'scene.chant', 'scene.music', 'scene.laughter', 'scene.horn', 'scene.whistle', 'scene.buzzer', 'scene.gunshot_explosion', 'scene.silence', 'scene.background_speech', 'scene.siren', 'scene.phone_ring', 'scene.typing', 'scene.door_knock', 'scene.beep', 'scene.alarm', 'scene.wind', 'scene.traffic', 'scene.ambience.cafe', 'scene.ambience.office', 'scene.ambience.street', 'scene.noise_other', 'bio_ev.cough', 'bio_ev.throat_clear', 'bio_ev.sneeze', 'meta.degraded_audio', 'meta.overlap', 'meta.channel.telephone', 'meta.channel.narrowband', 'meta.channel.mulaw', 'meta.channel.mp3', 'meta.channel.opus', 'meta.room.small', 'meta.room.medium', 'meta.room.large', 'meta.clipping', 'meta.snr.high', 'meta.snr.mid', 'meta.snr.low', 'scene.scream', 'scene.crying', 'scene.glass_break', 'scene.distress_call', 'scene.distress', 'meta.people_present', 'meta.media_command', 'meta.hidden_command']
CTX = 3

class EventHead(nn.Module):

    def __init__(self, d_in=1280, k=len(LABELS), hid=256):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.c1 = nn.Conv1d(d_in, hid, 5, padding=2)
        self.c2 = nn.Conv1d(hid, k, 3, padding=1)

    def forward(self, x):
        z = self.ln(x).transpose(1, 2)
        return self.c2(F.gelu(self.c1(z))).transpose(1, 2)

    @torch.no_grad()
    def window(self, x, lo, hi, n_total):
        a = max(0, lo - CTX)
        b = hi + CTX if n_total is None else min(n_total, hi + CTX)
        z = self.ln(x[a:b]).T[None]
        if lo - CTX < 0:
            z = F.pad(z, (CTX - lo, 0))
        r = 0 if n_total is None else max(0, hi + CTX - n_total)
        if r:
            z = F.pad(z, (0, r))
        h1 = F.gelu(F.conv1d(z, self.c1.weight, self.c1.bias))
        off = lo - CTX + 2
        idx = torch.arange(h1.shape[-1]) + off
        valid = (idx >= 0) & (idx < (n_total if n_total is not None else 1 << 30))
        h1 = h1 * valid.to(h1.dtype)
        y = F.conv1d(h1, self.c2.weight, self.c2.bias)
        return y[0].T[:hi - lo]

class DirectedHead(nn.Module):

    def __init__(self, d_in=640):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.fc = nn.Linear(d_in, 1)

    @torch.no_grad()
    def prob(self, X):
        if len(X) == 0:
            return None
        z = self.fc(self.ln(X.float()))[..., 0]
        return float(torch.sigmoid(z.mean()))

class EventTracker:

    def __init__(self, head, thr, status, enabled):
        self.head = head
        self.thr = np.array([thr.get(l, 0.5) for l in LABELS], np.float32)
        self.status = status
        self.on = np.array([l in enabled for l in LABELS])
        self.X = []
        self.x0 = 0
        self.n_done = 0
        self.open = {}
        self.out = []
        self.probs = []

    def push(self, x, t):
        self.X += [np.concatenate([a, b]) for a, b in zip(x, t)]

    def step(self, final=False):
        n = self.x0 + len(self.X)
        hi = n if final else n - CTX
        new = []
        if hi > self.n_done:
            base = max(self.x0, self.n_done - CTX)
            X = torch.as_tensor(np.stack(self.X[base - self.x0:]), dtype=torch.float32)
            z = self.head.window(X, self.n_done - base, hi - base, n - base if final else None)
            p = torch.sigmoid(z).numpy()
            self.probs.append(p)
            on = np.nonzero(self.on)[0]
            for i in range(len(p)):
                f = self.n_done + i
                for j in on:
                    lab = LABELS[j]
                    if p[i, j] >= self.thr[j]:
                        o = self.open.get(lab)
                        if o is None:
                            self.open[lab] = dict(label=lab, start=round(f / 10, 1), end=round((f + 1) / 10, 1), p=round(float(p[i, j]), 4), status=self.status.get(lab, 'experimental'))
                        else:
                            o['end'] = round((f + 1) / 10, 1)
                            o['p'] = max(o['p'], round(float(p[i, j]), 4))
                    elif lab in self.open:
                        new.append(self.open.pop(lab))
            self.n_done = hi
            cut = self.n_done - CTX - self.x0
            if cut > 0:
                self.X = self.X[cut:]
                self.x0 += cut
        if final:
            new += self._close_all()
        self.out += new
        return new

    def _close_all(self):
        out = list(self.open.values())
        self.open = {}
        return out

    def frames(self):
        return np.concatenate(self.probs) if self.probs else np.zeros((0, len(LABELS)), np.float32)
