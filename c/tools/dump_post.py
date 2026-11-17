"""Fixtures for test_post (c/tests/data/post.jsonl): arc_audio's quantities, sound-alike similarity, disfluency tags,
end-of-turn probabilities, sound-alike name candidates and the guard on fixed inputs.
    python c/tools/dump_post.py > c/tests/data/post.jsonl
Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from arc_audio.quantity import quantities
from arc_audio.phonetic import key, similarity
from arc_audio.nft2_post import disfluency, eot_prob, almost, guard

Q = ['set power to seventy five percent', 'give rover 45 more per cent', 'shields to 40', 'send the rover two kilometers north',
     'forty five percent', 'a little more resea rch please um then give protection 80 more percent as well', 'this one is good',
     'one hundred and five meters', 'two point five km', 'minus one hundred seventy degrees celsius', 'give it ten more',
     '5,000 feet', 'set to 40% now', 'two and a half hours', 'five nine five three', 'twenty twenty', 'a hundred and twelve',
     'drive at thirty kilometers per hour', 'turn ninety degrees left', 'three thousand four hundred and two', 'one more time',
     'ten times faster', 'We need 2.5 kilograms', 'eleven point oh five', "it's 12km away", 'nineteen sixty nine',
     'go fourteen hundred meters', 'one', 'a', 'two hundred fifty six', 'power at 100%.', 'oh no', 'Apollo sixteen rover']
P = [('digo laki', 'De Gerlache'), ('degor lockheed', 'De Gerlache'), ('vasilev', 'Vavilov'), ('sappelov', 'Vavilov'),
     ('tiger', 'Tycho'), ('jerrys', 'Antares'), ('gigi shannon', 'Eugene Cernan'), ('puzz aldrin', 'Buzz Aldrin'),
     ('roaming', 'Roving'), ('pythagoramus', 'Pythagoras'), ('where is', 'Werner'), ('the moon', 'Tumon'),
     ('Shackleton', 'shackleton'), ('Apollo 12', 'apollo twelve'), ('knight', 'night'), ('Xavier', 'zavier'),
     ('Schmidt', 'Smith'), ('church', 'kirk'), ('phone', 'fone'), ('whale', 'wail'), ('', 'Tycho'), ('Mare Crisium', 'mary chrism')]
D = ['a little more resea research please um then give protection 80 more percent as well', 'okay okay go to go to tycho',
     'you know what i mean the rover', 'do you know where it is', 'i mean the far side', 'um like you know i want ice',
     'the the rover', 'ro- rover go', 'uh uh what', 'i mean it', 'you know', 'go to Tycho, you know, quickly']
E = [('roger', 0.35), ('copy that', 0.35), ('go to the', 0.6), ('set power to seventy', 0.6), ('where is tycho', 0.6),
     ('where is tycho', 0.3), ('yes please', 0.4), ('', 1.0), ('um', 0.9), ('copy', 0.2), ('no no no', 0.5), ('roger that', 0.5)]
NAMES = [dict(name='De Gerlache'), dict(name='Vavilov'), dict(name='Tycho'), dict(name='Antares', aliases=['Antares crater']),
         dict(name='Eugene Cernan'), dict(name='Buzz Aldrin'), dict(name='Shackleton'), dict(name='Pythagoras')]
A = ['drive the rover from shackleton to digo laki', 'roger vasilev', 'tell me about tiger', 'where is jerrys',
     'tell me about gigi shannon', 'tell me about puzz aldrin', 'where is the moon', 'i like the rover', 'go to pythagoramus now',
     'show me vavilov and vasilev', 'tell me about the weather', 'degor lockheed monk']

out = []
for t in Q:
    out.append(dict(k='qty', w=t.split(), q=[dict(x, value=float(x['value'])) for x in quantities(t.split())]))
out.append(dict(k='qty', w='Apollo sixteen rover'.split(), tags=['ENTITY_MISSION', 'ENTITY_MISSION', None], q=[]))
for a, b in P:
    out.append(dict(k='phon', a=a, b=b, ka=key(a), kb=key(b), s=similarity(a, b)))
for t in D:
    out.append(dict(k='disfl', w=t.split(), t=[w['disfl'] for w in disfluency([dict(text=x) for x in t.split()])]))
for t, p in E:
    out.append(dict(k='eot', w=t.split(), pause=p, p=eot_prob([dict(text=x) for x in t.split()], p)))
for t in A:
    hits = almost([dict(text=x) for x in t.split()], NAMES)
    out.append(dict(k='almost', w=t.split(), names=NAMES, hits=[[h['word_begin'], h['word_end'], h['kb_index'], h['p']] for h in hits]))
def signal(n, base, loud, a, b):
    """deterministic test audio (libarc's test makes the same): a sine with a slow wobble, louder in [a, b)"""
    i = np.arange(n, dtype=np.float64)
    y = base * np.sin(i * 0.05) * (1 + 0.3 * np.sin(i * 0.0007))
    y[a:b] *= loud
    return y.astype(np.float32)


for base, loud in [(0.0003, 1), (0.05, 1), (0.002, 30), (0.02, 30), (0.0, 1), (0.01, 1)]:
    sig = dict(n=48000, base=base, loud=loud, a=4000, b=20000)
    y = signal(**sig)
    words = [dict(text=x, start=s, end=s + 0.3, conf=c) for x, s, c in [('you', 0.5, 0.9), ('you', 0.9, 0.2), ('you', 1.3, 0.3), ('go', 1.6, 0.8), ('vavilov', 2.9, 0.3)]]
    keep, why = guard(words, y, 0.0)
    out.append(dict(k='guard', sig=sig, words=words, why=why, kept=[w['text'] for w in keep]))
for o in out:
    print(json.dumps(o))
