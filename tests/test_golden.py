"""The reference package reproduces the golden outputs in tests/golden/ (same checks a port must pass: scripts/check_parity.py).

    ARC_MODEL=/local/arc-v1.1 pytest -q tests/test_golden.py         # Arc Mini; add ARC_TEST_PARAKEET=1 for Arc Transcriber (full)
"""
import json, os, socket, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
MODEL = os.environ.get("ARC_MODEL")


def online():
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
        return True
    except OSError:
        return False


@pytest.mark.parametrize("backbone", ["mini", pytest.param("full", marks=pytest.mark.skipif(
    not os.environ.get("ARC_TEST_PARAKEET"), reason="ARC_TEST_PARAKEET=1 runs Arc Transcriber (full)"))])
def test_golden(backbone, tmp_path):
    if not MODEL and not online():
        pytest.skip("offline and no ARC_MODEL")
    import check_parity as CP
    gold = json.load(open(ROOT / "tests" / "golden" / f"{backbone}.json"))
    cand = CP.run(backbone, MODEL, "cpu", 4)
    errs, _ = CP.compare(gold, json.loads(json.dumps(cand, default=float)))
    assert not errs, "\n".join(errs)


def test_compare_detects_changes():
    import check_parity as CP
    gold = json.load(open(ROOT / "tests" / "golden" / "mini.json"))
    bad = json.loads(json.dumps(gold))
    bad["output"]["transcript"][0]["text"] += " extra"
    bad["stages"]["features"]["head"][0] += 1.0
    errs, _ = CP.compare(gold, bad)
    assert len(errs) >= 2
    assert not CP.compare(gold, gold)[0]
