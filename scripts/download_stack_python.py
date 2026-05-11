#!/usr/bin/env python3
"""Download a Python subset of The Stack and split into train + holdout.

Streams `bigcode/the-stack` (gated dataset — must be HF-logged-in and have
accepted terms on the dataset page), filters by permissive license,
dedupes by content hash, and writes to:

    corpus/python/        # training set (~95% of files)
    holdout/python/       # held-out set (~5%, never used in training)

Split is deterministic: a file goes to holdout iff
sha256(content)[:1] == "0" (1/16 of files), so re-running with the same
files reproduces the same split.

Usage:
    .venv\\Scripts\\python.exe scripts\\download_stack_python.py --max-files 10000
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

DEFAULT_LICENSES = (
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


def download_and_split(
    *,
    max_files: int,
    license_allowlist: tuple[str, ...],
    train_dir: Path,
    holdout_dir: Path,
    languages: tuple[str, ...] = ("python",),
    holdout_hex_prefix: str = "0",
    min_content_len: int = 100,
    max_content_len: int = 200_000,
) -> tuple[int, int]:
    """Stream The Stack subset across one or more languages and split deterministically."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("Install: pip install datasets") from e

    train_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)

    # Concatenate one streamed dataset per language. Each language's dataset
    # is loaded lazily and we round-robin through them so the corpus stays
    # balanced if `max_files` is hit before some languages are exhausted.
    print(f"streaming bigcode/the-stack across {languages} (target {max_files:,} files)")
    datasets = []
    for lang in languages:
        datasets.append(
            iter(load_dataset(
                "bigcode/the-stack",
                data_dir=f"data/{lang}",
                split="train",
                streaming=True,
            ))
        )

    def _round_robin():
        active = list(range(len(datasets)))
        while active:
            for i in list(active):
                try:
                    yield next(datasets[i])
                except StopIteration:
                    active.remove(i)

    ds = _round_robin()

    seen_hashes: set[str] = set()
    n_train = n_holdout = 0
    n_seen = n_dropped_license = n_dropped_dedup = n_dropped_size = 0

    license_set = {x.lower() for x in license_allowlist}

    def _row_license_ok(row: dict) -> bool:
        # Stack v1 schema: licenses live in three list-valued fields. We accept
        # if ANY of them contains an allow-listed SPDX id.
        for field in ("max_stars_repo_licenses", "max_issues_repo_licenses", "max_forks_repo_licenses"):
            licenses = row.get(field) or []
            for spdx in licenses:
                if isinstance(spdx, str) and spdx.strip().lower() in license_set:
                    return True
        return False

    for row in ds:
        if n_train + n_holdout >= max_files:
            break
        n_seen += 1

        if not _row_license_ok(row):
            n_dropped_license += 1
            continue

        content = row.get("content", "")
        if not content or not (min_content_len <= len(content) <= max_content_len):
            n_dropped_size += 1
            continue

        full_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if full_hash in seen_hashes:
            n_dropped_dedup += 1
            continue
        seen_hashes.add(full_hash)

        # Deterministic split: holdout iff hash starts with the chosen hex prefix.
        is_holdout = full_hash.startswith(holdout_hex_prefix)
        target_dir = holdout_dir if is_holdout else train_dir

        rel_path = (
            row.get("max_stars_repo_path")
            or row.get("max_issues_repo_path")
            or row.get("max_forks_repo_path")
            or f"file_{n_seen}.py"
        ).replace("/", "_").replace("\\", "_")[:80]
        # Idx prefix prevents filename collisions; use a short hash slice for uniqueness.
        idx = n_holdout if is_holdout else n_train
        out = target_dir / f"{idx:06d}_{full_hash[:8]}_{rel_path}"
        try:
            out.write_text(content, encoding="utf-8")
        except (OSError, UnicodeError):
            continue

        if is_holdout:
            n_holdout += 1
        else:
            n_train += 1

        if (n_train + n_holdout) % 500 == 0:
            print(
                f"  written {n_train + n_holdout:,} files "
                f"(train={n_train:,}, holdout={n_holdout:,}, "
                f"seen={n_seen:,}, dropped: license={n_dropped_license}, "
                f"dedup={n_dropped_dedup}, size={n_dropped_size})"
            )

    print(
        f"\nFinal: train={n_train:,}, holdout={n_holdout:,}, "
        f"streamed={n_seen:,} rows, "
        f"dropped: license={n_dropped_license}, dedup={n_dropped_dedup}, size={n_dropped_size}"
    )
    return n_train, n_holdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-files",
        type=int,
        default=10_000,
        help="Total files to keep (train + holdout). Default 10k ≈ 300-500MB.",
    )
    ap.add_argument(
        "--train-dir",
        default="corpus/python",
        help="Training corpus directory (default: corpus/python)",
    )
    ap.add_argument(
        "--holdout-dir",
        default="holdout/python",
        help="Held-out corpus directory (default: holdout/python)",
    )
    ap.add_argument(
        "--holdout-prefix",
        default="0",
        help="Hex prefix for holdout split (default '0' = 1/16 ≈ 6.25%)",
    )
    ap.add_argument(
        "--lang",
        default="python",
        help=(
            "Comma-separated languages from The Stack v1 "
            "(e.g. 'python,javascript,typescript,rust,go'). Default: python"
        ),
    )
    args = ap.parse_args()

    languages = tuple(lang.strip() for lang in args.lang.split(",") if lang.strip())

    n_train, n_holdout = download_and_split(
        max_files=args.max_files,
        license_allowlist=DEFAULT_LICENSES,
        train_dir=Path(args.train_dir),
        holdout_dir=Path(args.holdout_dir),
        languages=languages,
        holdout_hex_prefix=args.holdout_prefix,
    )

    if n_train == 0:
        print("ERROR: no training files written. Check HF auth + dataset access.")
        return 1

    print(f"\nNext steps:")
    print(f"  cute build --corpus {args.train_dir} --output ./output_v2 --config configs/default.toml")
    print(
        f"  python -m benchmarks.compression --tokenizer ./output_v2 "
        f"--holdout {args.holdout_dir} --max-files 1000"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
