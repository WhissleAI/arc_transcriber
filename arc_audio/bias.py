import re
from difflib import SequenceMatcher
import torch
try:
    from transformers import LogitsProcessor
except Exception:
    LogitsProcessor = object
DEFAULT = dict(start_bonus=2.0, cont_bonus=1.5, inside=1.0, outside=0.3, rewrite_margin=0.5, rewrite_sim=0.6, name_types=['player', 'team', 'official', 'venue'], format_numbers=True)

class Trie:

    def __init__(self, tok, vocab):
        self.tok = tok
        self.seqs = []
        self.prefix = {}
        self.starts = {}
        for ty, terms in (vocab or {}).items():
            for t in terms:
                for form in {t, t.lower()} if ty not in ('player', 'team', 'official', 'venue') else {t}:
                    ids = tok(form).input_ids
                    if ids and ids[0] == getattr(tok, 'bos_token_id', None):
                        ids = ids[1:]
                    ids = tuple((i for i in ids if i not in (tok.eos_token_id,)))
                    if not ids:
                        continue
                    self.seqs.append((ids, ty))
                    self.starts.setdefault(ids[0], set()).add(ty)
                    for k in range(1, len(ids)):
                        self.prefix.setdefault(ids[:k], {}).setdefault(ids[k], set()).add(ty)
        self.maxlen = max((len(s) for s, _ in self.seqs), default=0)

class BiasProcessor(LogitsProcessor):

    def __init__(self, trie, weights, start_bonus, cont_bonus, prompt_len=1):
        self.t, self.w, self.sb, self.cb, self.p0 = (trie, weights, start_bonus, cont_bonus, prompt_len)
        self.start = {tokid: self.sb * max((self.w.get(ty, 0.0) for ty in tys)) for tokid, tys in self.t.starts.items()}
        self._dev = None

    def _start_tensors(self, dev, dtype):
        if self._dev != (dev, dtype):
            ks = list(self.start)
            self._si = torch.tensor(ks, device=dev, dtype=torch.long)
            self._sv = torch.tensor([self.start[k] for k in ks], device=dev, dtype=dtype)
            self._dev = (dev, dtype)
        return (self._si, self._sv)

    def __call__(self, input_ids, scores):
        gen = input_ids[0, self.p0:].tolist()
        cont = {}
        for k in range(min(len(gen), self.t.maxlen - 1), 0, -1):
            nxt = self.t.prefix.get(tuple(gen[-k:]))
            if nxt:
                for tokid, tys in nxt.items():
                    cont[tokid] = max(cont.get(tokid, 0.0), self.cb * max((self.w.get(ty, 0.0) for ty in tys)))
                break
        if not self.start and (not cont):
            return scores
        scores = torch.log_softmax(scores.float(), -1).to(scores.dtype)
        if self.start:
            si, sv = self._start_tensors(scores.device, scores.dtype)
            scores[0, si] += sv
        ex = {t: b - self.start.get(t, 0.0) for t, b in cont.items() if b > self.start.get(t, 0.0)}
        if ex:
            scores[0, torch.tensor(list(ex), device=scores.device)] += torch.tensor(list(ex.values()), device=scores.device, dtype=scores.dtype)
        return scores

class RowBias(LogitsProcessor):

    def __init__(self, base):
        self.b = base

    def __call__(self, input_ids, scores):
        return torch.cat([self.b(input_ids[i:i + 1], scores[i:i + 1].clone()) for i in range(input_ids.shape[0])], 0)

def weights_for(span_types, all_types, inside, outside):
    return {ty: inside if ty in span_types else outside for ty in all_types}
