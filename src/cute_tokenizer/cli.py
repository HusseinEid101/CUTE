"""Command-line interface for CUTE.

cute build --corpus ./corpus --output ./output [--config configs/default.toml]
cute roundtrip-check --tokenizer ./output --corpus ./holdout
cute info --tokenizer ./output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Silence the noisy "None of PyTorch, TensorFlow >= 2.0, or Flax have been found"
# warning from `transformers`. We only use the tokenizer, not the model layer,
# so this warning is irrelevant. Set BEFORE any transformers import below.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from ._version import __version__
from .config import CUTEConfig
from .corpus import iter_corpus_files
from .manifest import BuildManifest
from .tokenizer import CUTETokenizerFast
from .trainer import build_cute


def _cmd_build(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config)) if args.config else CUTEConfig()
    manifest_path = build_cute(
        corpus_dir=Path(args.corpus),
        output_dir=Path(args.output),
        config=config,
    )
    print(f"Build complete. Manifest: {manifest_path}")
    return 0


def _cmd_roundtrip_check(args: argparse.Namespace) -> int:
    tok_dir = Path(args.tokenizer)
    tok = CUTETokenizerFast(
        tokenizer_file=tok_dir / "tokenizer.json",
        cute_mapping_file=tok_dir / "cute_mapping.json",
    )

    corpus_dir = Path(args.corpus)
    files_checked = files_failed = 0
    for path in iter_corpus_files(
        corpus_dir,
        extensions=(".py", ".js", ".ts", ".java", ".c", ".cpp", ".rs", ".go", ".rb", ".php"),
    ):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        files_checked += 1
        ids = tok(text, add_special_tokens=False).input_ids
        decoded = tok.decode(ids, skip_special_tokens=True)
        if decoded != text:
            files_failed += 1
            print(f"FAIL {path.relative_to(corpus_dir)}")
            if args.verbose:
                _show_diff(text, decoded)
        if files_checked >= args.max_files:
            break

    print(f"Round-trip check: {files_checked - files_failed}/{files_checked} OK")
    return 0 if files_failed == 0 else 1


def _cmd_info(args: argparse.Namespace) -> int:
    tok_dir = Path(args.tokenizer)
    manifest = BuildManifest.read(tok_dir / "build_manifest.json")
    if args.summary:
        # Compact summary instead of full JSON dump.
        d = manifest.to_dict()
        cfg = d.get("config", {})
        print(f"cute_version: {d.get('cute_version')}")
        print(f"baseline:     {cfg.get('baseline_name', 'n/a')}")
        print(f"vocab_size:   {cfg.get('vocab_size')}")
        print(f"pua_budget:   {cfg.get('pua_budget')}")
        print(f"pua_assigned: {d.get('pua_mapping_size')}")
        print(f"vocab_hash:   {d.get('vocab_hash')}")
        print(f"corpus_hash:  {d.get('corpus_hash')}")
        audit = cfg.get("merge_audit", {})
        if audit:
            print(
                f"merge_audit:  found={audit.get('pua_pua_merges_found', 0)}, "
                f"removed={audit.get('pua_pua_merges_removed', 0)}"
            )
    else:
        print(json.dumps(manifest.to_dict(), indent=2))
    return 0


def _show_diff(expected: str, got: str) -> None:
    """Print a brief diff for round-trip failures."""
    for i, (e, g) in enumerate(zip(expected, got, strict=False)):
        if e != g:
            ctx = max(0, i - 20)
            print(f"  first diff at offset {i}:")
            print(f"    expected: ...{expected[ctx : i + 20]!r}")
            print(f"    got     : ...{got[ctx : i + 20]!r}")
            return
    if len(expected) != len(got):
        print(f"  length differs: expected={len(expected)}, got={len(got)}")


def _load_config(path: Path) -> CUTEConfig:
    """Load a config from TOML. Lazy import — `tomllib` is stdlib in 3.11+."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib  # type: ignore[no-redef]
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return CUTEConfig(**data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cute", description="CUTE tokenizer builder")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Train a CUTE tokenizer from a corpus")
    p_build.add_argument("--corpus", required=True, help="Corpus directory")
    p_build.add_argument("--output", required=True, help="Output directory")
    p_build.add_argument("--config", help="Optional TOML config path")
    p_build.set_defaults(func=_cmd_build)

    p_rt = sub.add_parser("roundtrip-check", help="Verify byte-equal round-trip")
    p_rt.add_argument("--tokenizer", required=True, help="Trained tokenizer dir")
    p_rt.add_argument("--corpus", required=True, help="Held-out corpus to check")
    p_rt.add_argument("--max-files", type=int, default=10_000)
    p_rt.add_argument("--verbose", action="store_true")
    p_rt.set_defaults(func=_cmd_roundtrip_check)

    p_info = sub.add_parser("info", help="Print build manifest")
    p_info.add_argument("--tokenizer", required=True)
    p_info.add_argument(
        "--summary",
        action="store_true",
        help="Print a compact human-readable summary instead of full JSON",
    )
    p_info.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
