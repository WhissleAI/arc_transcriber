"""Offline tests of the NFT2 read-outs (no model download): typed quantities, sound-alike keys and 'almost' candidates, the
no-speech / tail / repetition guard, disfluency tags, the end-of-turn heuristic, the echo canceller's playback_leak and
barge-in rule, and the n-best helpers. libarc has the same code; c/tests/test_post.c and test_echo.c check it against these
functions' outputs."""
import numpy as np

from arc_audio import nft2_post as P
from arc_audio.echo import EchoCanceller, barge_in
from arc_audio.phonetic import key, similarity
from arc_audio.quantity import quantities
from arc_audio.rnnt_beam import utt_alts, word_alts

SR = 16000


def q1(text, tags=None):
    return [(q['type'], q['value'], q['unit'], q['relative'], q['written']) for q in quantities(text.split(), tags)]


def test_quantities():
    assert q1('give rover seventy five percent power') == [('percent', 75, '%', 0, '75%')]
    assert q1('give rover 45 more per cent') == [('percent', 45, '%', 1, '45%')]
    assert q1('drive two kilometers north') == [('distance', 2, 'km', 0, '2 km')]
    assert q1('shields to 40') == [('number', 40, None, 0, '40')]
    assert q1('five nine five three') == [('number', 5, None, 0, '5'), ('number', 9, None, 0, '9'),
                                          ('number', 5, None, 0, '5'), ('number', 3, None, 0, '3')]
    assert q1('apollo twelve', ['ENTITY_MISSION', 'ENTITY_MISSION']) == []   # inside a name
    assert q1('go to the crater') == []


def test_phonetic():
    assert key('Vavilov') == key('vavilov') and similarity('vavilof', 'Vavilov') > 0.8
    assert similarity('vasilev', 'Vavilov') > 0.6
    assert similarity('tiger', 'Tycho') < similarity('tycho', 'Tycho') == 1.0
    assert similarity('', 'Tycho') == 0.0


def test_almost():
    names = [dict(name='Vavilov'), dict(name='Buzz Aldrin'), dict(name='Abulfeda E')]
    hits = P.almost([dict(text=w) for w in 'tell me about puzz aldrin'.split()], names)
    assert [(h['word_begin'], h['word_end'], h['candidate']) for h in hits] == [(3, 5, 'Buzz Aldrin')]
    # a spelling that IS on the list is the linker's, not a sound-alike
    assert P.almost([dict(text=w) for w in 'show me vavilov'.split()], names) == []
    # a satellite crater ('Abulfeda E') needs its letter
    assert P.almost([dict(text='abulfeda')], names) == []
    # words the RNNT heard well (conf >= max_conf) are not sound-alikes
    assert P.almost([dict(text='roger', conf=0.99), dict(text='vasilev', conf=0.95)], names) == []
    assert P.almost([dict(text='roger', conf=0.99), dict(text='vasilev', conf=0.3)], names)[0]['candidate'] == 'Vavilov'
    # common words never are, on their own
    assert P.almost([dict(text='live', conf=0.3)], [dict(name='Love H')], common={'live'}) == []


def test_almost_names_state():
    out = dict(transcript=[dict(words=[dict(text='go', start=0.0, end=0.2, conf=0.99),
                                       dict(text='to', start=0.2, end=0.3, conf=0.99),
                                       dict(text='pythagoramus', start=0.3, end=0.9, conf=0.2)])],
               names=[dict(start=0.3, end=0.9, heard='pythagoramus', entity=None, state='not_in_list', tag='entity.mention.place')])
    recs = P.almost_names(out, [dict(name='Pythagoras')])
    assert len(recs) == 1 and recs[0]['state'] == 'almost' and recs[0]['candidate'] == 'Pythagoras'
    assert recs[0]['entity'] is None and recs[0]['candidate_p'] >= P.ALMOST['min_p']


def _sig(n, loud=None, base=0.01):
    i = np.arange(n)
    y = base * np.sin(i * 0.05) * (1 + 0.3 * np.sin(i * 0.0007))
    if loud:
        a, b, g = loud
        y[a:b] *= g
    return y.astype(np.float32)


def test_guard():
    words = [dict(text='you', start=0.5, end=0.8, conf=0.9), dict(text='you', start=0.9, end=1.2, conf=0.2),
             dict(text='go', start=1.6, end=1.9, conf=0.8)]
    kept, why = P.guard(words, np.zeros(SR * 3, np.float32))                   # digital silence: nothing is speech
    assert kept == [] and set(why) == {'no_speech'}
    y = _sig(3 * SR, (4000, 20000, 30.0))                                       # speech-level audio 0.25 .. 1.25 s
    kept, why = P.guard(words, y)
    assert [w['text'] for w in kept] == ['you', 'go'] or 'repeat' in why       # the low-confidence repeat goes


def test_disfluency():
    ws = P.disfluency([dict(text=w) for w in 'um go go to the the crater'.split()])
    assert [w.get('disfl') for w in ws][:2] == ['filler', 'repeat']
    assert P.clean_text(ws) == 'go to the crater'


def test_eot():
    ws = [dict(text=w) for w in 'go to the'.split()]
    assert P.eot_prob(ws, 0.8) < P.eot_prob([dict(text=w) for w in 'go to tycho'.split()], 0.8)
    assert P.eot_prob([dict(text='roger')], 0.4) > 0.7
    assert P.eot_prob([], 1.0) == 0.0


def _speechlike(n, f0, seed):
    r = np.random.default_rng(seed)
    t = np.arange(n) / SR
    ph = 2 * np.pi * np.cumsum(f0 * (1 + 0.1 * np.sin(2 * np.pi * 0.5 * t))) / SR
    x = sum(np.sin(k * ph) / k for k in range(1, 10)) + 0.8 * r.standard_normal(n)
    return (0.05 * x * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def test_echo_leak_and_barge_in():
    n = 6 * SR
    ref = _speechlike(n, 140, 1)
    rir = np.zeros(800)
    rir[0] = 1.0
    rir[1:] = np.random.default_rng(2).standard_normal(799) * np.exp(-np.arange(1, 800) / 200) * 0.5
    rir /= np.sqrt((rir ** 2).sum())
    echo = np.zeros(n, np.float32)
    echo[800:] = (np.convolve(ref, rir)[:n - 800] * 0.5).astype(np.float32)
    talk = np.zeros(n, np.float32)
    talk[4 * SR:5 * SR] = _speechlike(SR, 250, 3) * 2
    for mic, want_talk in ((echo, False), (echo + talk, True)):
        ec = EchoCanceller()
        ec.feed_ref(ref, 0.0)
        for a in range(0, n, 1600):
            ec.feed_mic(mic[a:a + 1600])
        ec.flush()
        assert len(ec.residual()) == n
        assert ec.leak(3.0, 4.0) > 0.8                                         # converged: the playback is explained
        lk = ec.leak(4.0, 5.0)
        assert (lk < 0.5) == want_talk
        assert barge_in(ec, 4.0, 4.4) == want_talk
    assert EchoCanceller().leak(0, 1) is None                                  # no reference: no score


def test_nbest_helpers():
    class Pc:
        p = ['▁go', '▁to', '▁ty', 'cho', '▁tie', '▁co']
    hyps = [((0, 1, 2, 3), -1.0), ((0, 1, 4, 5), -2.0), ((0, 1, 2, 3), -3.0)]
    alts = utt_alts(Pc, hyps, 'Go to Tycho.')
    assert [a['text'] for a in alts] == ['go to tie co']
    ws = word_alts([dict(text='Go'), dict(text='to'), dict(text='Tycho.')], alts)
    assert ws[2]['alts'] == [dict(text='tie co', p=alts[0]['p'])]
