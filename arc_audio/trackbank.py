import numpy as np

class TrackBank:

    def __init__(self, beta=2.0, tau_new=0.5, tau_upd=0.5, confirm=20, switch=10, smooth=5, max_tracks=64):
        self.beta, self.tau_new, self.tau_upd, self.confirm, self.switch, self.smooth = (beta, tau_new, tau_upd, confirm, switch, smooth)
        self.max_tracks = max_tracks
        self.S = None
        self.nt = 0
        self.ring = []
        self.cur = 0
        self.pend = None
        self.pend_fr = []
        self.newbuf = []
        self.new_fr = []

    def measure(self, e, slot_z):
        z = np.asarray(slot_z, np.float32)
        P = np.exp(z - z.max())
        P /= P.sum()
        Ps = P[1:] / max(P[1:].sum(), 1e-06)
        m = np.concatenate([np.asarray(e, np.float32), self.beta * Ps])
        return m / max(np.linalg.norm(m), 1e-06)

    def step(self, f, e, slot_z, speech):
        if not speech:
            return (0, [])
        E = self.measure(e, slot_z)
        if self.S is None:
            self.S = np.zeros((self.max_tracks, len(E)), np.float32)
        self.ring.append(E)
        self.ring = self.ring[-self.smooth:]
        eb = np.mean(self.ring, 0)
        eb /= max(1e-06, np.linalg.norm(eb))
        nt = self.nt
        revised = []
        sims = self.S[:nt] / np.maximum(1e-06, np.linalg.norm(self.S[:nt], axis=1, keepdims=True)) @ eb if nt else np.zeros(0)
        sb = float(sims.max()) if nt else -1.0
        cand = int(sims.argmax()) + 1 if sb >= self.tau_new else -1
        if cand == -1:
            self.newbuf.append(E)
            self.new_fr.append(f)
            self.pend = None
            self.pend_fr = []
            if len(self.newbuf) >= self.confirm and nt < self.max_tracks:
                self.S[nt] = np.sum(self.newbuf, 0)
                self.nt += 1
                self.cur = self.nt
                revised = list(self.new_fr)
                self.newbuf = []
                self.new_fr = []
                self.ring = self.ring[-1:]
        else:
            self.newbuf = []
            self.new_fr = []
            if cand != self.cur:
                if self.pend != cand:
                    self.pend = cand
                    self.pend_fr = []
                self.pend_fr.append(f)
                if len(self.pend_fr) >= self.switch or self.cur == 0:
                    self.cur = cand
                    revised = list(self.pend_fr)
                    self.pend = None
                    self.pend_fr = []
                    self.ring = self.ring[-1:]
            else:
                self.pend = None
                self.pend_fr = []
        if self.cur > 0:
            cc = self.S[self.cur - 1] / max(1e-06, np.linalg.norm(self.S[self.cur - 1]))
            if float(cc @ (E / max(1e-06, np.linalg.norm(E)))) >= self.tau_upd:
                self.S[self.cur - 1] += E
        return (self.cur, revised)
