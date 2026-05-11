"""CUTE — Compact Unicode Token Encoding.

Public API:
    build_cute      — train a CUTE tokenizer from a corpus directory.
    CUTEConfig      — all knobs for the build pipeline.
    CUTETokenizerFast — HuggingFace-compatible inference wrapper.
    PUAMapping      — word ↔ PUA character mapping.
    load_default_tokenizer — load the bundled production-ready tokenizer.
"""

from __future__ import annotations

import os as _os
from pathlib import Path

# Silence the "None of PyTorch, TensorFlow, or Flax have been found" warning
# that `transformers` emits at import. CUTE only needs the tokenizer layer, not
# the model layer, so this warning is irrelevant. Must be set BEFORE the first
# `transformers` import — putting it here covers both library and CLI paths.
_os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from ._version import __version__
from .baseline import BaselineTokenizer, Cl100kBaseline, NullBaseline, get_default_baseline
from .config import CUTEConfig
from .pua import PUAMapping
from .selection import compute_candidate_score, select_by_savings
from .tokenizer import CUTETokenizerFast
from .trainer import build_cute, load_mapping, save_mapping


def _bundled_data_dir() -> Path:
    """Directory containing tokenizer.json and cute_mapping.json.

    Resolution order:
    1. Package ``data/`` (wheel / after prepare copies into the package tree).
    2. Repository ``model/`` (editable install from a checkout with ``src/`` layout).
    """
    pkg_dir = Path(__file__).resolve().parent
    embedded = pkg_dir / "data"
    if (embedded / "tokenizer.json").is_file() and (embedded / "cute_mapping.json").is_file():
        return embedded
    if pkg_dir.parent.name == "src":
        repo_root = pkg_dir.parent.parent
        dev = repo_root / "model"
        if (dev / "tokenizer.json").is_file() and (dev / "cute_mapping.json").is_file():
            return dev
    raise RuntimeError(
        "Bundled tokenizer files not found. Expected tokenizer.json and cute_mapping.json in "
        "the package data directory, or in ./model at the repository root. "
        "For releases, ensure model/ is populated before building the wheel."
    )


def load_default_tokenizer() -> CUTETokenizerFast:
    """Load the bundled production-ready CUTE tokenizer.

    This tokenizer was trained on a large code corpus (The Stack subset)
    with 80k vocab and 90% coverage target. It is ready to use immediately
    after `pip install cute-tokenizer` — no training required.

    Returns
    -------
    CUTETokenizerFast
        Pre-trained tokenizer instance.

    Example
    -------
        >>> from cute_tokenizer import load_default_tokenizer
        >>> tok = load_default_tokenizer()
        >>> ids = tok("def hello(): return 42", add_special_tokens=False).input_ids
        >>> len(ids)
        6
    """
    data_dir = _bundled_data_dir()
    tokenizer_file = data_dir / "tokenizer.json"
    mapping_file = data_dir / "cute_mapping.json"

    return CUTETokenizerFast(
        tokenizer_file=str(tokenizer_file),
        cute_mapping_file=str(mapping_file),
    )


__all__ = [
    "BaselineTokenizer",
    "CUTEConfig",
    "CUTETokenizerFast",
    "Cl100kBaseline",
    "NullBaseline",
    "PUAMapping",
    "__version__",
    "build_cute",
    "compute_candidate_score",
    "get_default_baseline",
    "load_default_tokenizer",
    "load_mapping",
    "save_mapping",
    "select_by_savings",
]
