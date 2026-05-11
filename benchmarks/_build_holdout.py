"""Build a small holdout corpus from locally-available source files for the
benchmark runner. Local-only: copies real .py/.rs/.ts/.js files into
``benchmark-corpus/`` so the runner has something to chew on.

Usage::

    python benchmarks/_build_holdout.py --target benchmark-corpus --max 500

Sources used (in priority order):
  1. The repository's own src/, tests/, rust/, benchmarks/.
  2. The active venv's site-packages (Python).
  3. The user's cargo registry cache (Rust).
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

CODE_EXTS = {".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".java", ".c", ".cpp",
             ".rb", ".go", ".php", ".cs", ".swift", ".kt"}

REPO_ROOT = Path(__file__).resolve().parent.parent


def _gather_from(root: Path, max_files: int, *, min_bytes: int = 200,
                 max_bytes: int = 200_000) -> list[Path]:
    """Walk ``root`` and return up to ``max_files`` source files.

    Filters out: oversized files, generated files, vendored test fixtures,
    dist-info dirs, __pycache__.
    """
    if not root.exists():
        return []
    skip_dirs = {"__pycache__", ".git", "target", "node_modules", "build",
                 "dist", ".venv", ".pytest_cache", ".mypy_cache",
                 "vendor", "third_party"}
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in skip_dirs and not d.endswith(".dist-info")]
        for name in filenames:
            ext = Path(name).suffix.lower()
            if ext not in CODE_EXTS:
                continue
            p = Path(dirpath) / name
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size < min_bytes or size > max_bytes:
                continue
            out.append(p)
            if len(out) >= max_files * 4:  # gather more than needed; sample below
                return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="benchmark-corpus")
    ap.add_argument("--max", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    target = REPO_ROOT / args.target
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    rng = random.Random(args.seed)

    sources = []
    # 1. Repo itself.
    for sub in ("src", "tests", "rust", "benchmarks", "scripts"):
        sources.extend(_gather_from(REPO_ROOT / sub, args.max))

    # 2. Venv site-packages (Python). Limit per-source so one big lib doesn't dominate.
    venv_sp = REPO_ROOT.parent.parent / ".venv" / "Lib" / "site-packages"
    if not venv_sp.exists():
        # fallback: walk sys.path
        venv_sp = Path(sys.exec_prefix) / "Lib" / "site-packages"
    py_libs = ["requests", "hypothesis", "pytest", "regex", "tokenizers",
               "transformers", "numpy", "ahocorasick", "tqdm", "click", "rich",
               "attr", "anyio", "aiohttp", "altair", "build"]
    for lib in py_libs:
        lib_dir = venv_sp / lib
        if lib_dir.exists():
            files = _gather_from(lib_dir, 80)
            rng.shuffle(files)
            sources.extend(files[:60])

    # 3. Cargo registry (Rust). Sample a handful of crates.
    cargo_reg = Path.home() / ".cargo" / "registry" / "src"
    if cargo_reg.exists():
        for index_dir in cargo_reg.iterdir():
            if not index_dir.is_dir():
                continue
            crates = list(index_dir.iterdir())[:30]
            for crate in crates:
                if not crate.is_dir():
                    continue
                src = crate / "src"
                files = _gather_from(src if src.exists() else crate, 12)
                sources.extend(files[:8])

    # Dedup by path; sample final.
    unique = list({str(p): p for p in sources}.values())
    rng.shuffle(unique)
    chosen = unique[: args.max]

    # Copy into target/<lang>/<flat-index>.<ext>
    counts: dict[str, int] = {}
    for p in chosen:
        ext = p.suffix.lower()
        lang = ext.lstrip(".")
        lang_dir = target / lang
        lang_dir.mkdir(exist_ok=True)
        idx = counts.get(lang, 0)
        dest = lang_dir / f"{idx:04d}{ext}"
        try:
            shutil.copyfile(p, dest)
            counts[lang] = idx + 1
        except OSError:
            continue

    total = sum(counts.values())
    print(f"Holdout corpus written to {target}/")
    print(f"  total files: {total}")
    for lang, n in sorted(counts.items()):
        print(f"  {lang}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
