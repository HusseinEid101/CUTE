"""Baseline tokenizer adapters for benchmarking.

Each adapter exposes ``encode(text) -> list[int]`` and ``decode(ids) -> str``
plus a stable ``name`` and ``vocab_size``. Adapters that fail to load
(missing dependency, missing model files, gated download) are silently
skipped — the runner records their status as "skipped" with the reason.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Baseline:
    """A loadable tokenizer adapter."""

    name: str
    encode: "EncodeFn"
    decode: "DecodeFn"
    vocab_size: int
    note: str = ""


class EncodeFn(Protocol):
    def __call__(self, text: str) -> list[int]: ...


class DecodeFn(Protocol):
    def __call__(self, ids: list[int]) -> str: ...


def _load_cute(tokenizer_dir: Path) -> Baseline | None:
    """Load CUTE 1.1.0 via the production fast paths.

    `fast_encode` / `fast_decode` auto-use the purpose-built `cute-bpe`
    Rust encoder when the extension is loaded — that's what we want to
    measure in the comparison runner. Falls back to the HF wrapper path
    when the user sets `CUTE_USE_HF_BACKEND=1`.
    """
    try:
        from cute_tokenizer import CUTETokenizerFast
        tok = CUTETokenizerFast(
            tokenizer_file=tokenizer_dir / "tokenizer.json",
            cute_mapping_file=tokenizer_dir / "cute_mapping.json",
        )
        backend = "cute-bpe" if tok._cute_bpe is not None else "HF tokenizers"
        return Baseline(
            name="CUTE",
            encode=lambda t: tok.fast_encode(t),
            decode=lambda ids: tok.fast_decode(ids),
            vocab_size=tok.vocab_size,
            note=f"V2 model, {backend} backend, from {tokenizer_dir}",
        )
    except Exception as e:  # noqa: BLE001 — diagnostic context only
        warnings.warn(f"CUTE adapter failed: {e}", stacklevel=2)
        return None


def _load_cl100k() -> Baseline | None:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return Baseline(
            name="cl100k_base",
            encode=lambda t: enc.encode(t, disallowed_special=()),
            decode=lambda ids: enc.decode(ids),
            vocab_size=enc.n_vocab,
            note="OpenAI cl100k via tiktoken",
        )
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"cl100k adapter failed: {e}", stacklevel=2)
        return None


def _load_o200k() -> Baseline | None:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return Baseline(
            name="o200k_base",
            encode=lambda t: enc.encode(t, disallowed_special=()),
            decode=lambda ids: enc.decode(ids),
            vocab_size=enc.n_vocab,
            note="OpenAI o200k via tiktoken (GPT-4o)",
        )
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"o200k adapter failed: {e}", stacklevel=2)
        return None


def _load_gpt2() -> Baseline | None:
    try:
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        return Baseline(
            name="gpt2",
            encode=lambda t: tok(t, add_special_tokens=False)["input_ids"],
            decode=lambda ids: tok.decode(ids, skip_special_tokens=True),
            vocab_size=tok.vocab_size,
            note="HuggingFace gpt2",
        )
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"gpt2 adapter failed: {e}", stacklevel=2)
        return None


def _load_codellama() -> Baseline | None:
    try:
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
        return Baseline(
            name="codellama",
            encode=lambda t: tok(t, add_special_tokens=False)["input_ids"],
            decode=lambda ids: tok.decode(ids, skip_special_tokens=True),
            vocab_size=tok.vocab_size,
            note="codellama/CodeLlama-7b-hf",
        )
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"codellama adapter failed: {e}", stacklevel=2)
        return None


def _load_starcoder2() -> Baseline | None:
    try:
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
        return Baseline(
            name="starcoder2",
            encode=lambda t: tok(t, add_special_tokens=False)["input_ids"],
            decode=lambda ids: tok.decode(ids, skip_special_tokens=True),
            vocab_size=tok.vocab_size,
            note="bigcode/starcoder2-3b",
        )
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"starcoder2 adapter failed: {e}", stacklevel=2)
        return None


def _load_llama3() -> Baseline | None:
    """Meta's Llama-3 tokenizer (SentencePiece BPE, 128k vocab).

    Gated; requires HF auth to download. Skipped if not accessible.
    """
    try:
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        return Baseline(
            name="llama3 (SentencePiece BPE)",
            encode=lambda t: tok(t, add_special_tokens=False)["input_ids"],
            decode=lambda ids: tok.decode(ids, skip_special_tokens=True),
            vocab_size=tok.vocab_size,
            note="meta-llama/Meta-Llama-3-8B",
        )
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"llama3 adapter failed: {e}", stacklevel=2)
        return None


def _load_xlmr() -> Baseline | None:
    """XLM-RoBERTa (SentencePiece Unigram, 250k vocab) — public, multilingual.

    Represents the SentencePiece *Unigram* algorithm family (vs everyone
    else here who is BPE-style). Not code-tuned, so compression will be
    poor — but it's the only public Unigram comparator in the field.
    """
    try:
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
        return Baseline(
            name="xlm-roberta (SentencePiece Unigram)",
            encode=lambda t: tok(t, add_special_tokens=False)["input_ids"],
            decode=lambda ids: tok.decode(ids, skip_special_tokens=True),
            vocab_size=tok.vocab_size,
            note="xlm-roberta-base",
        )
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"xlm-roberta adapter failed: {e}", stacklevel=2)
        return None


def _load_t5() -> Baseline | None:
    """T5 (SentencePiece Unigram, 32k vocab) — small, public.

    Another SentencePiece-Unigram comparator at a smaller vocab size.
    """
    try:
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("t5-base")
        return Baseline(
            name="t5 (SentencePiece Unigram)",
            encode=lambda t: tok(t, add_special_tokens=False)["input_ids"],
            decode=lambda ids: tok.decode(ids, skip_special_tokens=True),
            vocab_size=tok.vocab_size,
            note="t5-base",
        )
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"t5 adapter failed: {e}", stacklevel=2)
        return None


def load_all(cute_dir: Path) -> list[Baseline]:
    """Load every adapter that's available. Order = report order."""
    candidates = [
        _load_cute(cute_dir),
        _load_cl100k(),
        _load_o200k(),
        _load_gpt2(),
        _load_codellama(),
        _load_starcoder2(),
        _load_llama3(),
        _load_xlmr(),
        _load_t5(),
    ]
    return [b for b in candidates if b is not None]


__all__ = ["Baseline", "load_all"]
