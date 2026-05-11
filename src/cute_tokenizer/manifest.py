"""Build manifest — captures what the build did, for reproducibility audits.

Two builds with the same input + code version should produce manifests that
differ only in `build_host_info` and `timing_seconds`.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._version import __version__


@dataclass
class BuildManifest:
    """Captures the inputs, outputs, and environment of a CUTE build."""

    cute_version: str
    config: dict[str, Any]
    corpus_hash: str
    vocab_hash: str
    pua_mapping_size: int
    pua_codepoints_in_corpus: list[int]
    library_versions: dict[str, str]
    build_host_info: dict[str, str]
    timing_seconds: dict[str, float] = field(default_factory=dict)
    ingest_stats: dict[str, Any] = field(default_factory=dict)
    coverage_achieved: float = 0.0
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> BuildManifest:
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(**d)


def hash_corpus_shards(shards_dir: Path) -> str:
    """Hash all shards in deterministic order. Used as the corpus identity."""
    h = hashlib.sha256()
    for shard in sorted(shards_dir.glob("shard_*.jsonl.gz")):
        h.update(shard.name.encode("utf-8"))
        h.update(b":")
        with shard.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\n")
    return h.hexdigest()


def hash_vocab(token_to_id: dict[str, int]) -> str:
    """Hash the (token, id) vocab in sorted-by-id order."""
    h = hashlib.sha256()
    for tok, idx in sorted(token_to_id.items(), key=lambda kv: kv[1]):
        h.update(f"{idx}\t".encode())
        h.update(tok.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def collect_library_versions() -> dict[str, str]:
    """Snapshot versions of dependencies that affect tokenizer behavior."""
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for mod_name in ("tokenizers", "transformers", "regex", "ahocorasick", "xxhash", "orjson"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[mod_name] = "not_installed"
    return versions


def collect_host_info() -> dict[str, str]:
    """Capture host info — included for diagnostics, IGNORED for determinism."""
    import os

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": str(os.cpu_count() or 1),
        "python_implementation": platform.python_implementation(),
    }


def make_manifest(
    config: dict[str, Any],
    corpus_hash: str,
    vocab_hash: str,
    pua_mapping_size: int,
    pua_codepoints_in_corpus: list[int],
    coverage_achieved: float,
    timing_seconds: dict[str, float] | None = None,
    ingest_stats: dict[str, Any] | None = None,
) -> BuildManifest:
    return BuildManifest(
        cute_version=__version__,
        config=config,
        corpus_hash=corpus_hash,
        vocab_hash=vocab_hash,
        pua_mapping_size=pua_mapping_size,
        pua_codepoints_in_corpus=sorted(pua_codepoints_in_corpus),
        library_versions=collect_library_versions(),
        build_host_info=collect_host_info(),
        timing_seconds=timing_seconds or {},
        ingest_stats=ingest_stats or {},
        coverage_achieved=coverage_achieved,
    )


def determinism_diff(a: BuildManifest, b: BuildManifest) -> list[str]:
    """Return human-readable list of fields that differ in ways that BREAK
    determinism. `build_host_info` and `timing_seconds` are exempt."""
    diffs: list[str] = []
    ignore = {"build_host_info", "timing_seconds", "timestamp_utc"}
    da = a.to_dict()
    db = b.to_dict()
    for k in sorted(set(da) | set(db)):
        if k in ignore:
            continue
        if da.get(k) != db.get(k):
            diffs.append(f"{k}: {da.get(k)!r} != {db.get(k)!r}")
    return diffs


__all__ = [
    "BuildManifest",
    "collect_host_info",
    "collect_library_versions",
    "determinism_diff",
    "hash_corpus_shards",
    "hash_vocab",
    "make_manifest",
]
