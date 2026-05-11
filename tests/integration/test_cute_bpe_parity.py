"""Token-id parity test: cute-bpe vs HF Tokenizer on real holdout files.

This is the load-bearing gate for the 1.1.0 release. cute-bpe must
produce byte-identical token IDs to the current HuggingFace-backed
encoder on at least 200 random Python files from the Stack holdout.
Any divergence is a release blocker.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from cute_tokenizer._accel import BPEEncoder

from cute_tokenizer import CUTETokenizerFast

pytestmark = pytest.mark.integration

HOLDOUT_DIR = Path(r"C:\Users\husse\Downloads\CUTE\holdout\python")
N_FILES = 200
SEED = 42


def _sample_files() -> list[Path]:
    if not HOLDOUT_DIR.exists():
        pytest.skip(f"holdout dir not available at {HOLDOUT_DIR}")
    files = sorted(p for p in HOLDOUT_DIR.rglob("*.py") if p.is_file())
    rng = random.Random(SEED)
    rng.shuffle(files)
    return files[:N_FILES]


@pytest.fixture(scope="module")
def cute_bpe() -> BPEEncoder:
    return BPEEncoder("model/tokenizer.json", "model/cute_mapping.json")


@pytest.fixture(scope="module")
def cute_tok(monkeypatch_module) -> CUTETokenizerFast:
    """Reference tokenizer forced onto the HF-backed `fast_encode` path.

    cute-bpe (the 1.1.0 default) MUST match the HF-tokenizers-backed
    encoder byte-for-byte. We force the HF path via env var so we are
    actually comparing the two implementations and not cute-bpe to itself.
    """
    monkeypatch_module.setenv("CUTE_USE_HF_BACKEND", "1")
    return CUTETokenizerFast(
        tokenizer_file="model/tokenizer.json",
        cute_mapping_file="model/cute_mapping.json",
    )


@pytest.fixture(scope="module")
def monkeypatch_module(request):
    """Module-scoped monkeypatch (default pytest fixture is function-scoped)."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    request.addfinalizer(mp.undo)
    return mp


def test_token_id_parity_on_holdout(cute_bpe: BPEEncoder, cute_tok: CUTETokenizerFast) -> None:
    files = _sample_files()
    diffs: list[tuple[Path, int, int, int]] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        cb_ids = list(cute_bpe.encode(text))
        ref_ids = cute_tok.fast_encode(text)
        if cb_ids != ref_ids:
            differing = sum(1 for a, b in zip(cb_ids, ref_ids, strict=False) if a != b)
            diffs.append((path, len(cb_ids), len(ref_ids), differing))

    if diffs:
        sample = "\n".join(
            f"  {p.name}: cute={n_cb} hf={n_ref} differing={d}" for p, n_cb, n_ref, d in diffs[:5]
        )
        pytest.fail(
            f"cute-bpe diverges from CUTETokenizerFast.fast_encode on "
            f"{len(diffs)}/{len(files)} files.\nFirst few:\n{sample}"
        )
