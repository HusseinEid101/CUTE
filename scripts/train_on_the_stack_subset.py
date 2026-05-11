#!/usr/bin/env python3
"""Train CUTE on a subset of The Stack (BigCode) with 80k vocab target.

Downloads a representative sample (not the full 3TB) using HuggingFace datasets,
filters for permissive licenses, and trains with configs/default.toml settings.

Usage:
    .venv\Scripts\python.exe scripts\train_on_the_stack_subset.py \
        --lang python \
        --max-files 50000 \
        --output output_stack_80k
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cute_tokenizer import CUTEConfig, build_cute


def download_stack_subset(
    *,
    languages: list[str],
    max_files: int,
    license_allowlist: tuple[str, ...],
    cache_dir: Path | None = None,
) -> Path:
    """Stream The Stack from HuggingFace, filter licenses, write to corpus dir."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("Install: pip install datasets") from e

    corpus_dir = Path("corpus_stack_subset")
    corpus_dir.mkdir(exist_ok=True)

    total_written = 0
    seen_hashes: set[str] = set()

    for lang in languages:
        print(f"\n=== Streaming The Stack: {lang} ===")

        # Use streaming to avoid downloading full dataset
        ds = load_dataset(
            "bigcode/the-stack",
            data_dir=f"data/{lang}",
            split="train",
            streaming=True,
            cache_dir=str(cache_dir) if cache_dir else None,
        )

        lang_dir = corpus_dir / lang
        lang_dir.mkdir(exist_ok=True)

        for i, row in enumerate(ds):
            if total_written >= max_files:
                break

            # Check license
            license_spdx = (row.get("license") or "Unknown").strip()
            if license_spdx not in license_allowlist:
                continue

            # Deduplication via content hash (simple)
            content = row.get("content", "")
            if not content or len(content) < 50:
                continue

            import hashlib

            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            # Write file
            rel_path = row.get("path", f"file_{i}.py")
            safe_name = rel_path.replace("/", "_").replace("\\", "_")[:100]
            out_file = lang_dir / f"{total_written:06d}_{safe_name}"
            out_file.write_text(content, encoding="utf-8")

            total_written += 1
            if total_written % 1000 == 0:
                print(f"  Written {total_written} files...")

    print(f"\nTotal corpus files: {total_written}")
    return corpus_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Train CUTE on The Stack subset")
    ap.add_argument(
        "--lang",
        default="python",
        help="Comma-separated languages (python,javascript,typescript,go,rust)",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=50000,
        help="Max files to download (default: 50000 ≈ 2-5GB of code)",
    )
    ap.add_argument(
        "--output",
        default="output_stack_80k",
        help="Output directory for tokenizer",
    )
    ap.add_argument(
        "--config",
        default="configs/default.toml",
        help="TOML config (default: 80k vocab)",
    )
    args = ap.parse_args()

    # Match default.toml license allowlist
    license_allowlist = (
        "MIT",
        "Apache-2.0",
        "BSD-3-Clause",
        "BSD-2-Clause",
        "ISC",
        "Apache 2.0",
        "Apache License 2.0",
        "CC0-1.0",
        "Unlicense",
    )

    languages = [l.strip() for l in args.lang.split(",")]

    print("=" * 60)
    print("CUTE Training: The Stack (BigCode) Subset")
    print("=" * 60)
    print(f"Languages: {languages}")
    print(f"Target files: {args.max_files}")
    print(f"Vocab target: 80,000 (from {args.config})")
    print(f"License filter: {len(license_allowlist)} permissive licenses")
    print("")

    # Step 1: Download subset
    corpus = download_stack_subset(
        languages=languages,
        max_files=args.max_files,
        license_allowlist=license_allowlist,
    )

    # Step 2: Train with production config
    print(f"\n{'=' * 60}")
    print("Starting CUTE training...")
    print(f"{'=' * 60}")

    output_dir = Path(args.output)
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        return 1

    manifest = build_cute(
        corpus_dir=corpus,
        output_dir=output_dir,
        config=CUTEConfig.from_toml(config_path) if hasattr(CUTEConfig, 'from_toml') else None,
    )

    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"{'=' * 60}")
    print(f"Tokenizer: {output_dir}")
    print(f"Manifest: {manifest}")
    print("")
    print("Validate with:")
    print(f"  .venv\\Scripts\\python.exe benchmarks\\compression.py \\")
    print(f"      --tokenizer .\\{output_dir} --holdout .\\holdout --max-files 2000")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
