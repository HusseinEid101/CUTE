"""End-to-end build pipeline: corpus → frequency → savings selection → PUA → BPE.

Architecture (post-refit):

  Corpus shards
    │
    ▼
  Frequency counter ─────────────────────────┐
    │                                        │
    ▼                                        ▼
  PUA candidate selection (savings-based)  cl100k baseline
    │
    ▼
  PUAMapping (word ↔ PUA char)
    │
    ▼
  PUA-substituted text stream  ──►  BpeTrainer  ──►  tokenizer.json (raw)
                                                       │
                                                       ▼
                                          merge_policy audit + invariants
                                                       │
                                                       ▼
                                                 tokenizer.json (final)

The substitution stream is the load-bearing fix: it lets the BPE trainer
actually see PUA chars in the symbol stream so merges like
``[Ġ][⟦return⟧]`` (whitespace-prefix + PUA) can be learned. The previous
implementation registered PUA chars as `AddedToken`s only after training,
which made all whitespace+PUA merges impossible.

We still register PUA chars as `AddedToken`s as a *safety net* — any PUA
char that BPE didn't see often enough to merge is still guaranteed an
atomic vocab id.
"""

from __future__ import annotations

import json
import random
import time
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tokenizers import AddedToken, Tokenizer, decoders, models, trainers
from tokenizers.pre_tokenizers import ByteLevel

from .baseline import BaselineTokenizer, get_default_baseline
from .config import CUTEConfig
from .corpus import ingest_corpus, iter_shard_texts
from .frequency import count_frequencies
from .manifest import (
    hash_corpus_shards,
    hash_vocab,
    make_manifest,
)
from .merge_policy import audit_and_filter_tokenizer_file
from .pretokenizer import pretokenize_to_string
from .pua import PUAMapping, assign_pua_mapping
from .selection import (
    coverage_of,
    select_by_coverage,
    select_by_savings,
)

# ---------------------------------------------------------------------------
# Mapping persistence
# ---------------------------------------------------------------------------


def save_mapping(mapping: PUAMapping, path: Path) -> None:
    """Write the mapping as JSON. Word → PUA codepoint integer for clarity."""
    payload = {
        "version": 1,
        "size": mapping.size,
        "skipped_codepoints": list(mapping.skipped_codepoints),
        "word_to_codepoint": {w: ord(c) for w, c in mapping.word_to_pua.items()},
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_mapping(path: Path) -> PUAMapping:
    """Inverse of `save_mapping`."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    word_to_pua = {w: chr(cp) for w, cp in payload["word_to_codepoint"].items()}
    pua_to_word = {c: w for w, c in word_to_pua.items()}
    return PUAMapping(
        word_to_pua=word_to_pua,
        pua_to_word=pua_to_word,
        skipped_codepoints=tuple(payload.get("skipped_codepoints", [])),
    )


# ---------------------------------------------------------------------------
# Training-stream substitution
# ---------------------------------------------------------------------------


def _substituted_iter(
    texts: Iterable[str],
    mapping: PUAMapping,
) -> Iterator[str]:
    """Yield each text with PUA substitution applied. Empty mapping is a no-op."""
    if not mapping.word_to_pua:
        yield from texts
        return
    for text in texts:
        yield pretokenize_to_string(text, mapping)


# ---------------------------------------------------------------------------
# BPE training
# ---------------------------------------------------------------------------


def _build_bpe_tokenizer() -> Tokenizer:
    """Construct an untrained Tokenizer with vanilla ByteLevel pre-tokenizer."""
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True, trim_offsets=True)
    tok.decoder = decoders.ByteLevel()
    return tok


def _train_bpe(
    tokenizer: Tokenizer,
    shards_dir: Path,
    mapping: PUAMapping,
    config: CUTEConfig,
) -> None:
    """Run BPE training on the *PUA-substituted* shard stream, then add any
    PUA chars that BPE didn't pick up as `AddedToken`s for safety.

    Vocab budget split:
        bpe_vocab_size = config.vocab_size - len(mapping)
        bpe_merge_budget = bpe_vocab_size - len(special_tokens) - 256
    """
    bpe_vocab_size = config.vocab_size - len(mapping.word_to_pua)
    bpe_merge_budget = bpe_vocab_size - len(config.special_tokens) - 256

    if bpe_merge_budget < config.min_bpe_budget:
        raise ValueError(
            f"BPE merge budget too small: {bpe_merge_budget} < {config.min_bpe_budget}. "
            f"Reduce pua_budget (currently {config.pua_budget}) or raise "
            f"vocab_size (currently {config.vocab_size})."
        )

    trainer = trainers.BpeTrainer(
        vocab_size=bpe_vocab_size,
        special_tokens=list(config.special_tokens),
        initial_alphabet=list(ByteLevel.alphabet()),
        min_frequency=config.min_frequency,
        show_progress=False,
    )

    # THE FIX: substitute PUAs into the training stream so BPE actually
    # sees them. Without this, ByteLevel pre-tokenizes raw UTF-8 bytes
    # which never contain PUA chars, and no whitespace+PUA merge can
    # ever be learned.
    substituted = _substituted_iter(iter_shard_texts(shards_dir), mapping)
    tokenizer.train_from_iterator(substituted, trainer=trainer)

    # Safety net: any PUA char that wasn't picked up as a vocab entry by
    # BPE training (unlikely but possible for very rare ones) is registered
    # explicitly so it has an atomic id. Already-known chars are no-ops.
    existing_vocab = tokenizer.get_vocab()
    pua_added = [
        AddedToken(
            ch,
            single_word=False,
            lstrip=False,
            rstrip=False,
            normalized=False,
            special=False,
        )
        for ch in mapping.pua_chars
        if ch not in existing_vocab
    ]
    if pua_added:
        tokenizer.add_tokens(pua_added)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def build_cute(
    corpus_dir: Path,
    output_dir: Path,
    config: CUTEConfig | None = None,
    *,
    baseline: BaselineTokenizer | None = None,
) -> Path:
    """Run the full CUTE build. Returns the path to the manifest file.

    Idempotent: re-running with the same inputs reproduces the same artifacts.
    """
    if config is None:
        config = CUTEConfig()

    random.seed(config.seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}

    # Phase 1 — ingest corpus
    t0 = time.perf_counter()
    ingest_stats = ingest_corpus(
        corpus_dir=Path(corpus_dir),
        out_dir=output_dir,
        extensions=config.extensions,
        shard_size_bytes=config.shard_size_bytes,
        enable_secret_scrub=config.enable_secret_scrub,
        enable_license_filter=config.enable_license_filter,
        license_allowlist=config.license_allowlist,
    )
    timing["ingest"] = time.perf_counter() - t0
    shards_dir = output_dir / "shards"

    # Phase 2 — frequency analysis
    t0 = time.perf_counter()
    freq = count_frequencies(
        shards_dir=shards_dir,
        boost_weight=config.boost_weight,
        max_token_len=config.max_token_len,
        workers=config.workers,
    )
    timing["frequency"] = time.perf_counter() - t0

    if not freq:
        raise RuntimeError(
            f"Corpus at {corpus_dir} produced zero tokens. "
            "Check that the directory contains files matching config.extensions."
        )

    # Phase 3 — selection + PUA assignment
    t0 = time.perf_counter()
    if config.use_savings_selection:
        if baseline is None:
            baseline = get_default_baseline()
        selected = select_by_savings(
            freq,
            baseline,
            vocab_budget=config.pua_budget,
            max_len=config.max_token_len,
            allow_supplementary_pua=config.allow_supplementary_pua,
        )
    else:
        warnings.warn(
            "use_savings_selection=False — falling back to legacy frequency-based "
            "selection. Production builds should use savings-based scoring.",
            stacklevel=2,
        )
        selected = select_by_coverage(
            freq,
            coverage_target=config.coverage_target,
            max_len=config.max_token_len,
            max_tokens=config.pua_budget if config.pua_budget > 0 else None,
        )

    coverage = coverage_of(freq, selected)
    mapping = assign_pua_mapping(
        selected,
        corpus_pua_codepoints=ingest_stats.pua_codepoints_in_corpus,
        skip_bmp=config.pua_skip_bmp,
    )
    save_mapping(mapping, output_dir / "cute_mapping.json")
    timing["selection_and_pua"] = time.perf_counter() - t0

    # Phase 4 — BPE training (on PUA-substituted stream)
    t0 = time.perf_counter()
    tok = _build_bpe_tokenizer()
    _train_bpe(tok, shards_dir, mapping, config)
    tokenizer_path = output_dir / "tokenizer.json"
    tok.save(str(tokenizer_path))
    timing["bpe_training"] = time.perf_counter() - t0

    # Phase 4b — invariant audit + optional PUA-PUA merge filter
    t0 = time.perf_counter()
    audit_stats = audit_and_filter_tokenizer_file(
        tokenizer_path,
        mapping,
        strict=config.strict_pua_atomicity,
    )
    timing["merge_audit"] = time.perf_counter() - t0

    # Phase 5 — tokenizer_config.json (so HF auto_map works)
    _write_tokenizer_config(output_dir, config)

    # Phase 6 — manifest
    t0 = time.perf_counter()
    # Re-load vocab after potential rewrite.
    final_tok = Tokenizer.from_file(str(tokenizer_path))
    vocab = final_tok.get_vocab()
    baseline_name = baseline.name if baseline is not None else "n/a"
    manifest = make_manifest(
        config=config.to_dict(),
        corpus_hash=hash_corpus_shards(shards_dir),
        vocab_hash=hash_vocab(vocab),
        pua_mapping_size=mapping.size,
        pua_codepoints_in_corpus=sorted(ingest_stats.pua_codepoints_in_corpus),
        coverage_achieved=coverage,
        timing_seconds=timing,
        ingest_stats={
            k: (sorted(v) if isinstance(v, frozenset) else v)
            for k, v in asdict(ingest_stats).items()
        },
    )
    # Stash baseline + audit info on the manifest config dict so it's
    # captured without growing the BuildManifest dataclass surface.
    manifest.config = {
        **manifest.config,
        "baseline_name": baseline_name,
        "merge_audit": audit_stats,
    }
    manifest_path = output_dir / "build_manifest.json"
    manifest.write(manifest_path)
    timing["manifest"] = time.perf_counter() - t0

    return manifest_path


def _write_tokenizer_config(output_dir: Path, config: CUTEConfig) -> None:
    """Write `tokenizer_config.json` so `AutoTokenizer.from_pretrained` works.

    We deliberately do NOT set `bos_token` / `eos_token` / `pad_token` /
    `unk_token` because the conventional defaults (`<s>`, `</s>`, `<pad>`,
    `<unk>`) collide with natural text in code corpora — making them
    special tokens causes silent roundtrip loss whenever those substrings
    appear in a real file. Users who need padding / sequence boundaries
    should pick from the pipe-style markers (`<|endoftext|>`, etc.) which
    are guaranteed not to appear in real code.
    """
    cfg: dict[str, Any] = {
        "tokenizer_class": "CUTETokenizerFast",
        "auto_map": {
            "AutoTokenizer": [None, "cute_tokenizer.tokenizer.CUTETokenizerFast"],
        },
        "model_max_length": 1_000_000,
        "padding_side": "right",
        "truncation_side": "right",
        # Use <|endoftext|> as the sequence boundary if user enables special tokens.
        "eos_token": "<|endoftext|>",
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(cfg, indent=2),
        encoding="utf-8",
    )


__all__ = ["build_cute", "load_mapping", "save_mapping"]
