"""References for arc_score and the n-best (tests/test_score.c): for each clip in a list, Arc Mini's teacher-forced
log P(text | audio) for given texts (plain, and with the brace span inside ENTITY_PERSON / ENTITY_WORK: each variant
separately) and the final decode's beams (generate, 4 beams, num_return_sequences 4, whole clip as one utterance), the
prompt holding only the domain "assistant". JSON lines: {"clip", "texts": [...], "plain": [...], "person": [...],
"work": [...], "ntok": [...], "alts": [...], "alt_scores": [...]} (null where a text has no brace span).
  python -I dump_score_ref.py MINI_DIR WAV_DIR LIST.tsv OUT.jsonl
LIST: clip<TAB>text<TAB>text ...; a clip may be a stretch of a file: name@start-end (seconds), e.g. example_synth@0-7.25
Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE."""
import sys, json
import numpy as np, torch
from arc_audio import Arc, load_audio
from arc_audio.tagdec import TID
from arc_audio.moon import generate_kwargs
from arc_audio.api import context_dict
from arc_audio.transcribe import loopfix

mini, wavdir, lst, outp = sys.argv[1:5]
arc = Arc(mini, device='cpu', threads=4)
tr = arc._transcriber()
tr.set_context(context_dict(None, None, 'assistant', None, None))
P = tr._prompt_ids()
tok, m = arc.tok, tr.m
st, eos = m.config.decoder_start_token_id, m.config.eos_token_id
enc_t = lambda s: tok(s.strip(), add_special_tokens=False).input_ids if s.strip() else []

def seqs(t):
    if '{' not in t or '}' not in t[t.index('{'):]:
        return enc_t(t), None, None
    i = t.index('{'); j = t.index('}', i)
    pre, mid, post = enc_t(t[:i]), enc_t(t[i + 1:j]), enc_t(t[j + 1:])
    return pre + mid + post, pre + [TID['ENTITY_PERSON']] + mid + [TID['END']] + post, pre + [TID['ENTITY_WORK']] + mid + [TID['END']] + post

with open(outp, 'w') as out:
    for line in open(lst):
        f = line.rstrip('\n').split('\t')
        clip, texts = f[0], f[1:]
        name, _, span = clip.partition('@')
        w = load_audio(f'{wavdir}/{name}.wav')
        if span:
            a, b = (float(x) for x in span.split('-'))
            w = w[int(round(a * 16000)):int(round(b * 16000))]
        w = np.pad(w, (0, max(0, 3200 - len(w))))
        with torch.no_grad():
            kw = generate_kwargs(m, torch.as_tensor(w)[None].float())
            enc = m.get_encoder()(kw['input_values'], attention_mask=kw['attention_mask'])
            h0 = enc.last_hidden_state.clone()
            def score(s):
                if s is None: return None
                # a fresh copy each call: the decoder adds its positions to the encoder states in place
                e2 = type(enc)(last_hidden_state=h0.clone(), attention_mask=getattr(enc, 'attention_mask', None))
                lp = torch.log_softmax(m(encoder_outputs=e2, decoder_input_ids=torch.tensor([P + [st] + s])).logits.float()[0], -1)
                return float(sum(lp[len(P) + k, x] for k, x in enumerate(s + [eos])))
            r = dict(clip=clip, texts=texts, plain=[], person=[], work=[], ntok=[])
            for t in texts:
                a, b, c = seqs(t)
                r['plain'].append(score(a)); r['person'].append(score(b)); r['work'].append(score(c)); r['ntok'].append(len(a) + 1)
            # the transcriber's own logits processors (tag offsets, suppressed ids, repeat guard), as its final decode
            from transformers import LogitsProcessorList
            pre, guards = tr._guards(len(P) + 1)
            lp = pre + guards
            g = m.generate(**dict(kw, decoder_input_ids=torch.tensor([P + [st]]), decoder_start_token_id=P[0]) if P else kw,
                           max_length=tr._max_len(len(w)) + len(P), do_sample=False, num_beams=4, num_return_sequences=4,
                           output_scores=True, return_dict_in_generate=True, **({'logits_processor': LogitsProcessorList(lp)} if lp else {}))
            alts, sc = [], []
            for i in range(g.sequences.shape[0]):
                txt = tok.decode([int(x) for x in g.sequences[i].tolist()][len(P):], skip_special_tokens=True).strip()
                if txt not in alts: alts.append(txt); sc.append(float(g.sequences_scores[i]))
            r['alts'], r['alt_scores'] = alts, sc
        out.write(json.dumps(r) + '\n')
        print(clip, alts[:2], flush=True)
