import re
import numpy as np
import torch
STOPNAME = {'i', "i'm", 'sorry', 'thank', 'you', 'the', 'and', 'is', 'was', 'please', 'go', 'ahead', 'yes', 'no', 'okay', 'well', 'so', 'may', 'it', 'that', 'this', 'a', 'an', 'to', 'of', 'in', 'for', 'on', 'but', 'your', 'honor'}

class VoiceRoles:

    def __init__(self, eng, dom, cfg=None):
        c = dict(min_p=0.7, min_s=3.0, hide_uniform=['court_appellate'], hide_domains=['news'])
        c.update(cfg or {})
        self.c = c
        self.eng, self.dom = (eng, dom)
        m, R = (eng.m, eng.R)
        self.head = m.head
        cols = R.cols['speaker']
        self.rl = [j for j, i in enumerate(cols) if R.tab.ids[i].startswith('speaker.role.') and i in R.status]
        self.names = [R.tab.ids[cols[j]].split('.')[-1] for j in self.rl]
        self.la = np.asarray(R.la[[cols[j] for j in self.rl]], np.float64)
        self.acc = {}
        self.done = {}
        self.p = {}
        self.cue = {}
        self.pending = []
        self.seen_cue = set()
        self.segs = []
        self.todo = []

    def add_segment(self, voice, start, end):
        self.segs.append((str(voice), float(start), float(end)))
        self.todo.append([str(voice), float(start), float(end), int(start * 10)])
        self.step()

    def step(self):
        keep = []
        for seg in self.todo:
            v, a, b, nxt = seg
            f1 = min(int(np.ceil(b * 10)) - 1, self.eng.f - 1)
            if f1 >= nxt:
                self._add(v, list(range(nxt, f1 + 1)))
                seg[3] = f1 + 1
            if seg[3] < int(np.ceil(b * 10)):
                keep.append(seg)
        self.todo = keep
        self._resolve()

    def voice_at(self, t):
        best = None
        for v, a, b in self.segs:
            if a - 1e-06 <= t <= b + 1e-06:
                best = v
        return best

    @torch.no_grad()
    def _add(self, voice, frames):
        if not frames or not self.rl:
            return
        dn = self.done.setdefault(voice, set())
        frames = [f for f in frames if f not in dn and f >= self.eng.ring_f0]
        if not frames:
            return
        dn.update(frames)
        h = self.eng.audio_h_at(frames).float()
        z = h + self.head.pool_in['speaker'](h)
        a = self.acc.setdefault(voice, [torch.zeros(z.shape[1], dtype=torch.float64), 0])
        a[0] += z.double().sum(0).cpu()
        a[1] += len(frames)
        mean = (a[0] / a[1]).float()[None, None].to(h.device)
        lg = self.head.logits_q('speaker', mean, torch.tensor([self.eng.dom_i], device=h.device))[0, 0].float().cpu().numpy()
        zz = lg[self.rl].astype(np.float64) + self.la
        p = np.exp(zz - zz.max())
        self.p[voice] = p / p.sum()

    def raw(self, voice):
        p = self.p.get(voice)
        if p is None:
            return (None, 0.0)
        j = int(np.argmax(p))
        return (self.names[j], float(p[j]))

    def words(self, ws):
        if self.dom != 'court_appellate' or not ws:
            return
        toks = [re.sub("[^\\w']", '', w['w'] if 'w' in w else w['text']).lower() for w in ws]
        txt = [w['w'] if 'w' in w else w['text'] for w in ws]
        text = ' '.join(toks)
        for m in re.finditer('\\bmay it please the court\\b', text):
            k = text[:m.start()].count(' ')
            t = ws[min(k, len(ws) - 1)]['start']
            key = ('mpc', round(t, 1))
            if key in self.seen_cue:
                continue
            self.seen_cue.add(key)
            v = self.voice_at(t + 0.5)
            if v is not None:
                self.cue[v] = dict(role='counsel', name=None, why='may it please the Court')
        for i, tk in enumerate(toks):
            if tk != 'justice':
                continue
            t = ws[i]['start']
            key = ('addr', round(t, 1))
            if key in self.seen_cue:
                continue
            near_mpc = any((toks[j] == 'please' for j in range(max(0, i - 6), min(len(toks), i + 7))))
            if i > 0 and toks[i - 1] == 'chief':
                if near_mpc:
                    continue
                name = 'Chief Justice'
            elif i + 1 < len(toks) and toks[i + 1] and (toks[i + 1] not in STOPNAME) and txt[i + 1][:1].isupper():
                name = 'Justice ' + re.sub("[^\\w'\\-]", '', txt[i + 1]).replace("'s", '')
            else:
                continue
            self.seen_cue.add(key)
            self.pending.append((ws[i]['end'], name, self.voice_at(t)))

    def _resolve(self):
        keep = []
        segs = sorted(self.segs, key=lambda s: s[1])
        for t, name, who in self.pending:
            nxt = next((s for s in segs if t - 0.3 <= s[1] <= t + 10 and s[0] != who), None)
            if nxt is None:
                if not segs or segs[-1][2] < t + 10:
                    keep.append((t, name, who))
                continue
            if self.cue.get(nxt[0], {}).get('role') != 'counsel':
                self.cue[nxt[0]] = dict(role='judge', name=name, why='addressed as ' + name)
        self.pending = keep

    def roles(self):
        c = self.c
        out = {}
        for v, (_, n) in self.acc.items():
            role, p = self.raw(v)
            show = p >= c['min_p'] and n >= c['min_s'] * 10 and (self.dom not in c['hide_domains'])
            out[v] = dict(role=role, p=round(p, 3), s=round(n / 10, 1), label=role if show else None)
        if self.dom in c['hide_uniform']:
            big = [r for r in out.values() if r['s'] >= c['min_s'] and r['role']]
            if len(big) >= 2 and len({r['role'] for r in big}) == 1:
                for r in out.values():
                    if r['label']:
                        r['label'] = None
                        r['why'] = 'same role for every voice'
        for v, cu in self.cue.items():
            r = out.setdefault(v, dict(role=None, p=None, s=0.0, label=None))
            r['cue'] = cu['why']
            r['label'] = cu['name'] if cu['role'] == 'judge' and cu.get('name') else cu['role']
        return out
