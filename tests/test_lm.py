"""N-gram LM shallow fusion (arc_audio.lm, libarc's arc_set_lm).

    pytest -q tests/test_lm.py                       # format, maths, errors: no weights
    ARC_MODEL=/local/arc-v1.1 pytest -q tests/test_lm.py   # + Python vs C on the golden clip (needs make -C c)

The parity test builds a toy LM here (a small backoff bigram, enough to move the decoder; real LMs are trained
elsewhere and only need to follow the ARCLM1 format in docs/LIBARC.md).
"""
import json, math, os, socket, subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import torch

from arc_audio.lm import ArcLM, LMFusion, fuse, lm_key, write_lm

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "examples" / "example_synth.wav"
NAMES = ROOT / "examples" / "example_names.json"
C_BIN = ROOT / "c" / "build" / "arc-transcribe"

# the toy trigram of c/tests/test_lm.c: vocab 10, 1 = <s>, 2 = </s>, 3-7 text, 8-9 tags
TOY = {(3,): (-1.0, -0.5), (4,): (-2.0, -0.25), (5,): (-3.0, 0.0), (6,): (-4.0, 0.0), (7,): (-5.0, 0.0), (2,): (-2.5, 0.0),
       (1,): (-20.0, -0.1), (1, 3): (-0.2, -0.7), (3, 4): (-0.3, -0.6), (1, 3, 4): (-0.05, 0.0), (4, 5): (1.0, -0.4)}


@pytest.fixture
def toy(tmp_path):
    p = tmp_path / "toy.arclm"
    write_lm(p, 3, 10, TOY, unscored=-3.0, scored=range(1, 8))
    return p


def test_key_matches_libarc():
    assert lm_key([1]) == 0x0F6EE3214F2608B7
    assert lm_key([1, 5, 7]) == 0xD249F93DD7FB53AF
    assert lm_key([32599, 3]) == 0x26FDE75946D1BEC9


def test_load_and_backoff(toy):
    lm = ArcLM(toy)
    assert (lm.order, lm.vocab, lm.entries) == (3, 10, len(TOY))
    assert lm.is_scored(3) and not lm.is_scored(8) and not lm.is_scored(0) and not lm.is_scored(99)
    close = lambda a, b: abs(float(a) - b) < 1e-6
    assert close(lm.logp([1, 3], 4), -0.05)
    assert close(lm.logp([7, 3], 4), -0.3)
    assert close(lm.logp([3], 5), -3.5)
    assert close(lm.logp([1, 3], 5), -4.2)
    assert close(lm.logp([4, 5], 6), -4.4)              # (4 5) is a context only: its bow, never its logp
    assert close(lm.logp([1, 3, 4], 5), -3.85)          # only the last order-1 tokens count


@pytest.mark.parametrize("patch, msg", [
    (lambda b: b[:10], "too short"),
    (lambda b: b"GGUF\0\0\0\0" + b[8:], "bad magic"),
    (lambda b: b"ARCLM2\0\0" + b[8:], "unsupported ARCLM version"),
    (lambda b: b[:8] + (9).to_bytes(4, "little") + b[12:], "order"),
    (lambda b: b[:20] + (24).to_bytes(4, "little") + b[24:], "power of 2"),
    (lambda b: b[:-16], "size"),
    (lambda b: b + bytes(16), "size"),
])
def test_bad_files(toy, tmp_path, patch, msg):
    p = tmp_path / "bad.arclm"
    p.write_bytes(patch(toy.read_bytes()))
    with pytest.raises(ValueError, match=msg):
        ArcLM(p)


def test_full_table(tmp_path):
    p = tmp_path / "full.arclm"
    write_lm(p, 1, 10, {(3,): (-1, 0), (4,): (-1, 0)})    # cap 4 for 2 grams: fill the rest by hand
    b = bytearray(p.read_bytes())
    for o in range(24 + 16, len(b), 16):
        if not any(b[o:o + 8]):
            b[o:o + 8] = (0xFFFF0001 + o).to_bytes(8, "little")
    p.write_bytes(bytes(b))
    with pytest.raises(ValueError, match="no empty slot"):
        ArcLM(p)


def test_fusion_maths(toy):
    lm = ArcLM(toy)
    raw = torch.tensor([-9.0, -9.0, 1.0, 3.0, 2.9, 2.5, -0.5, -9.0, 2.95, -9.0])
    assert fuse(lm, 0.0, 16, raw, [1, 3], 2) is raw                  # weight 0: untouched
    ref = torch.log_softmax(raw, -1)
    w = 0.5
    lp = {t: float(lm.logp([1, 3], t)) for t in (3, 4, 5, 6)}
    top = max(lp.values())
    want = ref - w * 6.0
    for t in lp:
        want[t] = ref[t] + w * max(lp[t] - top, -6.0)
    want[8] = ref[8]                                                 # a tag: neutral
    want[2] = ref[2] + w * max(lp[3] - top, -6.0)                    # the end token follows the audio's best
    got = fuse(lm, w, 16, raw, [1, 3], 2)
    assert torch.allclose(got, want, atol=1e-5)
    assert got[4] > got[3]
    got2 = fuse(lm, w, 2, raw, [1, 3], 2)                             # only 3 and 8 are candidates
    assert abs(float(got2[4] - (ref[4] - w * 6.0))) < 1e-5 and abs(float(got2[2] - ref[2])) < 1e-5
    # the processor: batch rows, the prompt before the start token ignored
    p = LMFusion(lm, w, 16, start=2, eos=2)
    ids = torch.tensor([[7, 7, 1, 3], [9, 9, 1, 3]])
    out = p(ids, torch.stack([raw, raw]))
    assert torch.allclose(out[0], want, atol=1e-5) and torch.allclose(out[1], want, atol=1e-5)


# ---------------------------------------------------------------- parity with libarc (needs the weights and make -C c)

def _mini_dir():
    m = os.environ.get("ARC_MODEL")
    if m and os.path.isdir(m):
        return m if os.path.exists(os.path.join(m, "config.json")) else os.path.join(m, "mini")
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
    except OSError:
        return None
    from arc_audio import MODEL_REVISION
    from huggingface_hub import snapshot_download
    return os.path.join(snapshot_download("WhissleAI/arc-v1.1", revision=MODEL_REVISION, allow_patterns=["mini/*", "mini/*/*"]), "mini")


def _toy_lm(mini, path):
    """a backoff bigram (absolute discounting) over Arc Mini's ids from text that spells a name differently from the
    golden clip's list, so the fusion has something to change"""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(mini, "moonshine-streaming-tiny", "tokenizer.json"))
    V = tok.get_vocab_size()
    text = ["Delaney to Theo Brant.", "Brant drives, he scores.", "What a move by Theo Brant.", "Brant has twenty six points tonight.",
            "That puts the Harbor City Hawks ahead 102 to 98.", "Welcome back, folks."]
    uni, bi = Counter(), Counter()
    for e in tok.encode_batch(text, add_special_tokens=False):
        s = [1] + e.ids + [2]
        uni.update(s[1:])
        bi.update(zip(s, s[1:]))
    scored = [t for t in range(3, 32600)] + [1, 2]
    n, D = sum(uni.values()), 0.5
    floor = D * len(uni) / n / len(scored)
    grams = {(t,): (math.log(max(uni[t] - D, 0) / n + floor), 0.0) for t in scored}
    ctx = defaultdict(Counter)
    for (a, b), c in bi.items():
        ctx[a][b] = c
    for a, nx in ctx.items():
        tot = sum(nx.values())
        gam = D * len(nx) / tot
        for b, c in nx.items():
            grams[(a, b)] = (math.log((c - D) / tot + gam * math.exp(grams[(b,)][0])), 0.0)
        lp, _ = grams.get((a,), (1.0, 0.0))
        grams[(a,)] = (lp, math.log(gam))
    write_lm(path, 2, V, grams, unscored=-4.0, scored=scored)


@pytest.fixture(scope="module")
def mini():
    m = _mini_dir()
    if not m:
        pytest.skip("offline and no ARC_MODEL")
    return m


@pytest.fixture(scope="module")
def arc(mini):
    from arc_audio import Arc
    return Arc.from_pretrained(mini, threads=4)


def _c_texts(mini, lm, w):
    if not C_BIN.exists():
        pytest.skip("make -C c first")
    args = [str(C_BIN), mini, str(CLIP), "--names", str(NAMES), "--domain", "nba", "--json"]
    if lm:
        args += ["--lm", str(lm), "--lm-weight", str(w)]
    out = json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)
    return [(u["text"], [tuple(t) for t in u.get("tags", [])]) for u in out["transcript"]]


def _py_texts(out):
    return [(u["text"], [(t["tag"], t["text"]) for t in u.get("tags", [])]) for u in out["transcript"]]


def test_parity_with_libarc(arc, mini, tmp_path):
    lm = tmp_path / "toy.arclm"
    _toy_lm(mini, lm)
    names = json.load(open(NAMES))
    arc.set_lm(lm, 0.3)
    try:
        base = _py_texts(arc.predict(str(CLIP), names=names, domain="nba", return_frames=False, lm_weight=0.0))
        assert base == _c_texts(mini, None, 0)
        assert base == _c_texts(mini, lm, 0.0)                     # weight 0: identical to no LM
        changed = False
        for w in (0.3, 1.0, 3.0):
            py = _py_texts(arc.predict(str(CLIP), names=names, domain="nba", return_frames=False, lm_weight=w))
            assert py == _c_texts(mini, lm, w), f"weight {w}"
            changed |= py != base
        assert changed, "the toy LM never changed the transcript: the test does not exercise fusion"
    finally:
        arc.set_lm(None)


def _stream_texts(arc, wav, **kw):
    s = arc.stream(background=False, **kw)
    outs = [s.feed(wav[i:i + 1600]) for i in range(0, len(wav), 1600)] + [s.flush()]
    return [u["text"] for o in outs for u in (o or {}).get("transcript") or []]


def test_stream_finals_follow_the_lm(arc, mini, tmp_path):
    lm = tmp_path / "toy.arclm"
    _toy_lm(mini, lm)
    from arc_audio import load_audio
    wav = load_audio(str(CLIP))
    arc.set_lm(lm, 1.0)
    try:
        on = [u["text"] for u in arc.predict(wav, return_frames=False)["transcript"]]
        off = [u["text"] for u in arc.predict(wav, return_frames=False, lm_weight=0.0)["transcript"]]
        assert on != off
        assert _stream_texts(arc, wav) == on
        assert _stream_texts(arc, wav, lm_weight=0.0) == off
    finally:
        arc.set_lm(None)


def test_set_lm_checks(arc, tmp_path):
    p = tmp_path / "small.arclm"
    write_lm(p, 1, 10, {(3,): (-1.0, 0.0)})
    with pytest.raises(ValueError, match="another tokenizer"):
        arc.set_lm(p)
    with pytest.raises(ValueError):
        arc.set_lm_weight(0.5)                                    # no LM loaded
    assert arc.lm is None and arc.domains


def test_cli_flags():
    import subprocess, sys
    out = subprocess.run([sys.executable, "-m", "arc_transcriber", "--help"], capture_output=True, text=True, cwd=ROOT).stdout
    assert "--lm " in out and "--lm-weight" in out and "--lm-topk" in out
