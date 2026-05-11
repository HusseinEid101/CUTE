#!/usr/bin/env python3
"""Prepare CUTE for PyPI release with bundled production tokenizer.

This script:
1. Trains CUTE on a large corpus (The Stack subset or your corpus)
2. Copies tokenizer artifacts to src/cute_tokenizer/data/
3. Packages everything for pip install

Usage:
    # Option 1: Train on The Stack subset (recommended)
    .venv\Scripts\python.exe scripts\prepare_for_pypi_release.py \
        --train --stack --max-files 100000

    # Option 2: Use existing trained tokenizer (default: ./model)
    .venv\Scripts\python.exe scripts\prepare_for_pypi_release.py \
        --tokenizer-dir .\model

    # Option 3: Train on your own corpus
    .venv\Scripts\python.exe scripts\prepare_for_pypi_release.py \
        --train --corpus .\training_corpus

After running, build and upload:
    python -m build
    python -m twine upload dist/*
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))


def copy_tokenizer_to_package(tokenizer_dir: Path, pkg_data_dir: Path) -> None:
    """Optional: mirror tokenizer files under src/cute_tokenizer/data for editable installs.

    Wheels bundle ``model/*`` via Hatch ``force-include`` in pyproject.toml; this copy is only
    needed if you want ``src/cute_tokenizer/data`` populated on disk (e.g. older tooling).
    """
    pkg_data_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "tokenizer.json",
        "cute_mapping.json",
        "tokenizer_config.json",
    ]

    for fname in files_to_copy:
        src = tokenizer_dir / fname
        dst = pkg_data_dir / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {fname}")
        else:
            print(f"  ⚠ Missing {fname}")

    print(f"\nOptional mirror written to: {pkg_data_dir}")


def train_on_stack(max_files: int) -> Path:
    """Train on The Stack subset and return output directory."""
    from cute_tokenizer import build_cute, CUTEConfig

    # First download subset
    from train_on_the_stack_subset import download_stack_subset

    license_allowlist = (
        "MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause",
        "ISC", "Apache 2.0", "Apache License 2.0", "CC0-1.0", "Unlicense",
    )

    corpus = download_stack_subset(
        languages=["python"],
        max_files=max_files,
        license_allowlist=license_allowlist,
    )

    output_dir = REPO_ROOT / "output_release"
    # Default CUTEConfig already matches default.toml (80k vocab, 90% coverage)
    config = CUTEConfig()

    print(f"\nTraining with config: 80k vocab, 90% coverage target...")
    manifest = build_cute(corpus_dir=corpus, output_dir=output_dir, config=config)
    print(f"Training complete: {manifest}")

    return output_dir


def train_on_corpus(corpus_dir: Path) -> Path:
    """Train on user-provided corpus."""
    from cute_tokenizer import build_cute, CUTEConfig

    output_dir = REPO_ROOT / "output_release"
    # Default CUTEConfig already matches default.toml (80k vocab, 90% coverage)
    config = CUTEConfig()

    print(f"Training on: {corpus_dir}")
    print(f"Output: {output_dir}")
    manifest = build_cute(corpus_dir=corpus_dir, output_dir=output_dir, config=config)
    print(f"Training complete: {manifest}")

    return output_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare CUTE for PyPI with bundled tokenizer")
    ap.add_argument(
        "--tokenizer-dir",
        type=Path,
        help="Use existing trained tokenizer (skip training)",
    )
    ap.add_argument(
        "--train",
        action="store_true",
        help="Train a new tokenizer",
    )
    ap.add_argument(
        "--stack",
        action="store_true",
        help="Train on The Stack subset (requires --train)",
    )
    ap.add_argument(
        "--corpus",
        type=Path,
        help="Train on custom corpus directory (requires --train)",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=100000,
        help="Max files for The Stack training (default: 100k)",
    )
    ap.add_argument(
        "--mirror-to-src",
        action="store_true",
        help="Also copy tokenizer JSON into src/cute_tokenizer/data (optional)",
    )
    args = ap.parse_args()

    pkg_data_dir = REPO_ROOT / "src" / "cute_tokenizer" / "data"

    print("=" * 60)
    print("CUTE PyPI Release Preparation")
    print("=" * 60)

    # Determine tokenizer source
    if args.tokenizer_dir:
        tokenizer_dir = args.tokenizer_dir
        print(f"\nUsing existing tokenizer: {tokenizer_dir}")
    elif args.train:
        if args.stack:
            print(f"\nTraining on The Stack subset ({args.max_files} files)...")
            tokenizer_dir = train_on_stack(args.max_files)
        elif args.corpus:
            print(f"\nTraining on custom corpus: {args.corpus}")
            tokenizer_dir = train_on_corpus(args.corpus)
        else:
            print("ERROR: --train requires either --stack or --corpus")
            return 1
    else:
        default_model = REPO_ROOT / "model"
        if default_model.exists() and (default_model / "tokenizer.json").exists():
            tokenizer_dir = default_model
            print(f"\nUsing tokenizer in: {tokenizer_dir}")
        else:
            print("ERROR: No tokenizer found. Use --tokenizer-dir, --train, or populate ./model/")
            return 1

    # Validate tokenizer files exist
    required = ["tokenizer.json", "cute_mapping.json"]
    for f in required:
        if not (tokenizer_dir / f).exists():
            print(f"ERROR: Missing required file: {tokenizer_dir / f}")
            return 1

    if args.mirror_to_src:
        print(f"\nMirroring tokenizer into src/cute_tokenizer/data...")
        copy_tokenizer_to_package(tokenizer_dir, pkg_data_dir)
    else:
        print(f"\nWheel will bundle ./model via Hatch force-include (see pyproject.toml).")

    # Verify
    print(f"\nVerification:")
    from cute_tokenizer import load_default_tokenizer
    try:
        tok = load_default_tokenizer()
        print(f"  ✓ load_default_tokenizer() works")
        print(f"  ✓ Vocab size: {len(tok.get_vocab())}")
        test = "def hello(): return 42"
        ids = tok(test, add_special_tokens=False).input_ids
        print(f"  ✓ Test encode: '{test}' → {len(ids)} tokens")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 1

    print("\n" + "=" * 60)
    print("Ready for PyPI!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. python -m build")
    print("  2. python -m twine upload dist/*")
    print(f"\nUsers can now:")
    print(f"  pip install cute-tokenizer")
    print(f"  from cute_tokenizer import load_default_tokenizer")
    print(f"  tok = load_default_tokenizer()  # Production-ready!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
