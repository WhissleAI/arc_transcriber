"""Fixtures for c/tests/test_echo.c: arc_audio/echo.py on the bundled examples/example_synth.wav as the playback (its first 8 s),
its echo through a synthetic room response (60 ms delay, 100 ms decay, from a small LCG the C test repeats), with and without
another voice over it (1.5 s of the same file, reversed, at 5.5 s). The fixture keeps the expected playback_leak and barge-in
decision per window; the C test rebuilds the signals.

    python tools/dump_echo.py tests/data/echo.json
Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from arc_audio.echo import EchoCanceller, barge_in  # noqa: E402

SR = 16000


def lcg(seed, n):
    x, out = seed, np.empty(n)
    for i in range(n):
        x = (1103515245 * x + 12345) % 2147483648
        out[i] = x / 2147483648.0 - 0.5
    return out


def signals(case):
    import soundfile as sf
    y, _ = sf.read(os.path.join(HERE, '..', '..', 'examples', 'example_synth.wav'), dtype='float32')
    n = 8 * SR
    ref = y[:n].astype(np.float64)
    rir = np.zeros(1600)
    rir[0] = 1.0
    rir[1:] = lcg(7, 1599) * np.exp(-np.arange(1, 1600) / 400)
    rir /= np.sqrt((rir ** 2).sum())
    echo = np.zeros(n)
    echo[960:] = np.convolve(ref, rir)[:n - 960] * 0.5
    mic = echo + lcg(11, n) * 3e-4
    if case == 'echo_talk':
        mic[int(5.5 * SR):7 * SR] += y[20 * SR:int(21.5 * SR)][::-1]
    return mic.astype(np.float32), ref.astype(np.float32)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'tests/data/echo.json'
    cases = []
    for cid in ('echo', 'echo_talk'):
        mic, ref = signals(cid)
        ec = EchoCanceller()
        ec.feed_ref(ref, 0.0)
        for a in range(0, len(mic), 1600):
            ec.feed_mic(mic[a:a + 1600])
        ec.flush()
        wins = [(a, a + 1.0) for a in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)] + [(5.5, 5.8), (5.5, 7.0), (3.0, 3.5)]
        cases.append(dict(id=cid, windows=[[a, b, ec.leak(a, b), int(barge_in(ec, a, b))] for a, b in wins]))
        print(cid, [w[2:] for w in cases[-1]['windows']])
    json.dump(dict(cases=cases), open(out, 'w'), indent=0)


if __name__ == '__main__':
    main()
