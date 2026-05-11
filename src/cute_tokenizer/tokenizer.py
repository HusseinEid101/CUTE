"""HuggingFace-compatible tokenizer wrapper.

Subclasses `PreTrainedTokenizerFast` so users get a drop-in replacement for
`AutoTokenizer` in any HF training/inference loop.

Usage:
    from cute_tokenizer import CUTETokenizerFast
    tok = CUTETokenizerFast.from_pretrained("./output")
    ids = tok("def hello(): return 42").input_ids
    text = tok.decode(ids)

The two-line wrapper UX the user asked for. Pre-tokenization runs in Python
(Aho-Corasick over the PUA mapping); the trained byte-level BPE handles
the residual stream in Rust.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from transformers import PreTrainedTokenizerFast

from ._accel_loader import USE_RUST, accel, prepare_mapping
from .decode import reverse_pua_substitute
from .pretokenizer import pretokenize_to_string
from .pua import PUAMapping
from .trainer import load_mapping

if TYPE_CHECKING:
    pass


_MAPPING_FILENAME = "cute_mapping.json"


class CUTETokenizerFast(PreTrainedTokenizerFast):
    """CUTE wrapper. Performs Python-side PUA substitution before delegating
    encoding to the underlying byte-level BPE tokenizer.

    Inherits everything else from `PreTrainedTokenizerFast` — padding,
    truncation, batch encoding, special tokens, save/load semantics.
    """

    # Static map of constructor kwarg name → on-disk filename. Required by
    # PreTrainedTokenizerFast machinery; values are immutable strings, so
    # the class-level dict is safe despite RUF012's general advice.
    vocab_files_names = {  # type: ignore[assignment]  # noqa: RUF012
        "tokenizer_file": "tokenizer.json",
        "cute_mapping_file": _MAPPING_FILENAME,
    }

    def __init__(
        self,
        tokenizer_file: str | Path | None = None,
        cute_mapping_file: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        if tokenizer_file is None:
            raise ValueError("CUTETokenizerFast requires `tokenizer_file`")
        if cute_mapping_file is None:
            raise ValueError("CUTETokenizerFast requires `cute_mapping_file`")

        # PreTrainedTokenizerFast loads the underlying tokenizer.json itself.
        super().__init__(tokenizer_file=str(tokenizer_file), **kwargs)

        self._cute_mapping: PUAMapping = load_mapping(Path(cute_mapping_file))
        self._cute_mapping_file = str(cute_mapping_file)

        # 1.1.0: build the purpose-built `cute-bpe` encoder once. The
        # `fast_encode` / `fast_decode` hot paths use it instead of HF's
        # BPE for ~2x lower encode latency at byte-equal output. Falls
        # back to the HF-backed path if the Rust extension didn't load,
        # or when `CUTE_USE_HF_BACKEND=1` is set by the user.
        import os as _os

        force_hf = _os.environ.get("CUTE_USE_HF_BACKEND", "") not in (
            "",
            "0",
            "false",
            "False",
        )
        self._cute_bpe: Any | None = None
        if USE_RUST and not force_hf and hasattr(accel, "BPEEncoder"):
            try:
                self._cute_bpe = accel.BPEEncoder(str(tokenizer_file), str(cute_mapping_file))
            except Exception:
                self._cute_bpe = None

    # ---------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------

    @property
    def cute_mapping(self) -> PUAMapping:
        return self._cute_mapping

    def _save_pretrained(  # type: ignore[override]
        self,
        save_directory: Any,
        file_names: tuple[str, ...],
        legacy_format: bool | None = None,
        filename_prefix: str | None = None,
    ) -> tuple[str, ...]:
        """Save the BPE tokenizer.json + cute_mapping.json + tokenizer_config.json.

        We hook here (rather than `save_vocabulary`) because Fast tokenizers
        bypass `save_vocabulary` entirely — `_save_pretrained` is the real
        extension point.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Let the parent write tokenizer.json + tokenizer_config.json + special_tokens_map.json
        out = super()._save_pretrained(
            str(save_directory),
            file_names=file_names,
            legacy_format=legacy_format,
            filename_prefix=filename_prefix,
        )

        prefix = f"{filename_prefix}-" if filename_prefix else ""
        mapping_path = save_directory / f"{prefix}{_MAPPING_FILENAME}"

        from .trainer import save_mapping

        save_mapping(self._cute_mapping, mapping_path)

        return (*out, str(mapping_path))

    # ---------------------------------------------------------------------
    # Encode path (override at the lowest convenient level)
    # ---------------------------------------------------------------------

    def _cute_pretokenize(self, text: str) -> str:
        """Run PUA substitution + identifier splitting on a single string.

        Hot path: skip the wrapper-level `pretokenize_to_string` redirection
        and call `_accel.pretokenize_to_string` directly with the cached
        prepared mapping. Saves one Python frame per encode.
        """
        if USE_RUST:
            return accel.pretokenize_to_string(text, self._prepared_mapping)
        return pretokenize_to_string(text, self._cute_mapping)

    def _batch_encode_plus(  # type: ignore[override]
        self,
        batch_text_or_text_pairs: Any,
        **kwargs: Any,
    ) -> Any:
        return super()._batch_encode_plus(
            self._preprocess_batch(batch_text_or_text_pairs), **kwargs
        )

    def _encode_plus(  # type: ignore[override]
        self,
        text: Any,
        text_pair: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        text = self._preprocess_one(text)
        if text_pair is not None:
            text_pair = self._preprocess_one(text_pair)
        return super()._encode_plus(text, text_pair=text_pair, **kwargs)

    def _preprocess_one(self, x: Any) -> Any:
        if isinstance(x, str):
            return self._cute_pretokenize(x)
        # Pre-tokenized input (list of strings) — substitute each piece.
        if isinstance(x, list) and all(isinstance(p, str) for p in x):
            # Batch FFI hop: one allow_threads + Rayon over all pieces.
            if USE_RUST and hasattr(accel, "pretokenize_batch"):
                return list(accel.pretokenize_batch(x, self._prepared_mapping))
            return [self._cute_pretokenize(p) for p in x]
        return x

    def _preprocess_batch(self, batch: Any) -> Any:
        if not isinstance(batch, list):
            return batch
        # Fast path: a homogeneous list of plain strings can go through the
        # batched Rust API in one FFI hop (Rayon-parallel inside).
        if (
            USE_RUST
            and hasattr(accel, "pretokenize_batch")
            and all(isinstance(b, str) for b in batch)
        ):
            return list(accel.pretokenize_batch(batch, self._prepared_mapping))
        return [self._preprocess_pair_or_text(b) for b in batch]

    def _preprocess_pair_or_text(self, item: Any) -> Any:
        if isinstance(item, tuple) and len(item) == 2:
            a, b = item
            return (self._preprocess_one(a), self._preprocess_one(b))
        return self._preprocess_one(item)

    @property
    def _prepared_mapping(self) -> Any:
        """Lazily-built, instance-cached `_accel.PreparedMapping`. Avoids
        the global id-keyed cache lookup on every encode call.
        """
        cached = getattr(self, "_prepared_mapping_cache", None)
        if cached is not None:
            return cached
        prepared = prepare_mapping(self._cute_mapping)
        # Bypass dataclass-frozen-style attribute checks via __dict__.
        self.__dict__["_prepared_mapping_cache"] = prepared
        return prepared

    # ---------------------------------------------------------------------
    # Decode path
    # ---------------------------------------------------------------------

    def _decode(  # type: ignore[override]
        self,
        token_ids: Any,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs: Any,
    ) -> str:
        text = super()._decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
        if USE_RUST:
            return accel.reverse_pua_substitute(text, self._prepared_mapping)
        return reverse_pua_substitute(text, self._cute_mapping)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:  # type: ignore[override]
        text = super().convert_tokens_to_string(tokens)
        if USE_RUST:
            return accel.reverse_pua_substitute(text, self._prepared_mapping)
        return reverse_pua_substitute(text, self._cute_mapping)

    # ------------------------------------------------------------------
    # Fast paths — skip PreTrainedTokenizerFast machinery
    # ------------------------------------------------------------------
    #
    # `__call__` / `_encode_plus` build a `BatchEncoding` (input_ids +
    # attention_mask + special-token insertion + optional padding /
    # truncation). That machinery costs ~400 µs per call on top of the
    # actual BPE encode. When all you want is `list[int]` of token ids,
    # use `fast_encode` / `fast_decode` to skip the wrapper and call the
    # raw HF Tokenizer directly. Trade-off: no attention_mask, no
    # special-token insertion, no padding/truncation logic. For raw
    # inference / training pipelines that's usually fine.

    def fast_encode(self, text: str) -> list[int]:
        """Return token ids for ``text``. Skips the ``BatchEncoding`` wrapper.

        Uses the 1.1.0 purpose-built ``cute-bpe`` Rust encoder when
        available (~2x lower latency than the HF-backed fallback,
        byte-equal output). Set ``CUTE_USE_HF_BACKEND=1`` to force the
        previous HF-tokenizers path.
        """
        if self._cute_bpe is not None:
            return list(self._cute_bpe.encode(text))
        if USE_RUST:
            pre = accel.pretokenize_to_string(text, self._prepared_mapping)
        else:
            pre = pretokenize_to_string(text, self._cute_mapping)
        return self._tokenizer.encode(pre, add_special_tokens=False).ids

    def fast_encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Batched ``fast_encode``. Single Rayon-parallel FFI hop when
        the ``cute-bpe`` backend is loaded.
        """
        if self._cute_bpe is not None:
            return [list(ids) for ids in self._cute_bpe.encode_batch(texts)]
        if USE_RUST and hasattr(accel, "pretokenize_batch"):
            pres = list(accel.pretokenize_batch(texts, self._prepared_mapping))
        else:
            pres = [pretokenize_to_string(t, self._cute_mapping) for t in texts]
        return [enc.ids for enc in self._tokenizer.encode_batch(pres, add_special_tokens=False)]

    def fast_decode(self, ids: list[int]) -> str:
        """Decode token ids to text. Skips the wrapper.

        Uses the 1.1.0 ``cute-bpe`` decoder (table-lookup + byte-level
        inverse + reverse-PUA in one Rust call) when available.
        """
        if self._cute_bpe is not None:
            return self._cute_bpe.decode(list(ids))
        text = self._tokenizer.decode(ids, skip_special_tokens=True)
        if USE_RUST:
            return accel.reverse_pua_substitute(text, self._prepared_mapping)
        return reverse_pua_substitute(text, self._cute_mapping)

    def fast_decode_batch(self, ids_list: list[list[int]]) -> list[str]:
        """Batched decode. One Rayon-parallel FFI hop when ``cute-bpe``
        is loaded.
        """
        if self._cute_bpe is not None:
            return list(self._cute_bpe.decode_batch([list(ids) for ids in ids_list]))
        texts = self._tokenizer.decode_batch(ids_list, skip_special_tokens=True)
        if USE_RUST and hasattr(accel, "reverse_pua_batch"):
            return list(accel.reverse_pua_batch(texts, self._prepared_mapping))
        return [reverse_pua_substitute(t, self._cute_mapping) for t in texts]


__all__ = ["CUTETokenizerFast"]
