"""Corpus pipeline: stream files, dedupe by content hash, scrub secrets, shard.

The output is a sequence of deterministic shards on disk that downstream
phases (frequency counting, BPE training) can iterate efficiently.
"""

from __future__ import annotations

import gzip
import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import orjson
import regex as re

from ._accel_loader import USE_RUST, accel
from .pua import find_pua_codepoints

# ---------------------------------------------------------------------------
# Secret scrubbing
# ---------------------------------------------------------------------------

# Each pattern is conservative — false positives drop a file, which is fine
# at corpus scale. False negatives are far more dangerous (secret in vocab).
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("openai_api_key", re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("anthropic_api_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{50,}")),
    ("github_pat", re.compile(r"ghp_[A-Za-z0-9]{36}")),
    ("github_oauth", re.compile(r"gho_[A-Za-z0-9]{36}")),
    ("github_app", re.compile(r"(ghu|ghs)_[A-Za-z0-9]{36}")),
    ("google_api", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("slack_token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("private_key_pem", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
)


def has_secret(text: str) -> str | None:
    """Return the name of the first matching secret pattern, or None."""
    if USE_RUST:
        return accel.has_secret(text)
    for name, pat in SECRET_PATTERNS:
        if pat.search(text):
            return name
    return None


# ---------------------------------------------------------------------------
# License filter
# ---------------------------------------------------------------------------

# SPDX header detector. Matches lines like:
#   # SPDX-License-Identifier: MIT
#   // SPDX-License-Identifier: Apache-2.0
# Operates on the first 4 KiB of each file so we don't scan large blobs.
_SPDX_REGEX = re.compile(
    r"SPDX-License-Identifier\s*:\s*([A-Za-z0-9.\-+ ]+)",
    re.IGNORECASE,
)
# Heuristic: explicit "All rights reserved" / "Proprietary" / "Confidential"
# in the file head. We refuse files matching these unless an SPDX header
# explicitly grants a permissive license.
_PROPRIETARY_REGEX = re.compile(
    r"\b(All Rights Reserved|Proprietary and Confidential|UNLICENSED|License: Proprietary)\b",
    re.IGNORECASE,
)
_LICENSE_HEAD_BYTES = 4096


def detect_license(text: str) -> str | None:
    """Best-effort license detection from a file's head.

    Returns the SPDX identifier if found, otherwise None. Does NOT make a
    keep/drop decision — that's `is_license_allowed`'s job.
    """
    head = text[:_LICENSE_HEAD_BYTES]
    m = _SPDX_REGEX.search(head)
    if m:
        return m.group(1).strip()
    return None


def is_license_allowed(text: str, allowlist: Iterable[str]) -> bool:
    """Decide whether a file's license header (if any) permits inclusion.

    Logic:
      1. If an SPDX header is present and matches the allowlist → allow.
      2. If an SPDX header is present and does NOT match → reject.
      3. If no SPDX header but a 'proprietary' marker is in the head → reject.
      4. Otherwise (no headers, no markers) → allow. The corpus owner is
         responsible for not feeding obviously copyrighted material; the
         filter is a safety net, not a legal review.
    """
    spdx = detect_license(text)
    if spdx is not None:
        allow_set = {entry.strip().lower() for entry in allowlist}
        return spdx.lower() in allow_set

    head = text[:_LICENSE_HEAD_BYTES]
    return not _PROPRIETARY_REGEX.search(head)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusRecord:
    """One file's content + metadata."""

    path: str  # path relative to the corpus root
    text: str
    sha256: str

    def to_json(self) -> bytes:
        return orjson.dumps({"path": self.path, "text": self.text, "sha256": self.sha256})

    @classmethod
    def from_json(cls, line: bytes) -> CorpusRecord:
        d = orjson.loads(line)
        return cls(path=d["path"], text=d["text"], sha256=d["sha256"])


@dataclass(frozen=True)
class IngestStats:
    """Aggregated stats from one ingest pass."""

    files_seen: int
    files_kept: int
    files_dropped_dedup: int
    files_dropped_secret: int
    files_dropped_license: int
    files_dropped_decode: int
    files_dropped_size: int
    bytes_kept: int
    pua_codepoints_in_corpus: frozenset[int]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_corpus_files(
    corpus_dir: Path,
    extensions: Iterable[str],
    max_bytes: int = 5_000_000,
) -> Iterator[Path]:
    """Yield candidate files under `corpus_dir`, deterministically ordered.

    Sorted by relative path so iteration order is reproducible across runs.
    """
    ext_set = {e.lower() for e in extensions}
    candidates = [p for p in corpus_dir.rglob("*") if p.is_file() and p.suffix.lower() in ext_set]
    candidates.sort(key=lambda p: str(p.relative_to(corpus_dir)).replace("\\", "/"))
    for p in candidates:
        try:
            if p.stat().st_size > max_bytes:
                continue
        except OSError:
            continue
        yield p


def ingest_corpus(
    corpus_dir: Path,
    out_dir: Path,
    extensions: Iterable[str],
    shard_size_bytes: int = 64 * 1024 * 1024,
    enable_secret_scrub: bool = True,
    enable_license_filter: bool = False,
    license_allowlist: Iterable[str] = (),
    max_file_bytes: int = 5_000_000,
) -> IngestStats:
    """Read corpus files, dedupe + scrub, write line-delimited gzipped shards.

    Output layout:
        out_dir/shards/shard_00000.jsonl.gz
        out_dir/shards/shard_00001.jsonl.gz
        ...

    Each line of each shard is a CorpusRecord.to_json() blob.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(exist_ok=True)

    seen_hashes: set[str] = set()
    files_seen = files_kept = 0
    drop_dedup = drop_secret = drop_license = drop_decode = drop_size = 0
    bytes_kept = 0
    pua_codepoints: set[int] = set()
    license_allowlist_t = tuple(license_allowlist)

    shard_idx = 0
    shard_path = shards_dir / f"shard_{shard_idx:05d}.jsonl.gz"
    shard_fh: gzip.GzipFile | None = gzip.open(shard_path, "wb")  # noqa: SIM115 (rolling handle, closed in finally)
    bytes_in_shard = 0

    try:
        for path in iter_corpus_files(corpus_dir, extensions, max_bytes=max_file_bytes):
            files_seen += 1
            try:
                text = path.read_text(encoding="utf-8", errors="strict")
            except (UnicodeDecodeError, OSError):
                drop_decode += 1
                continue

            if not text:
                drop_size += 1
                continue

            sha = _hash_text(text)
            if sha in seen_hashes:
                drop_dedup += 1
                continue

            if enable_secret_scrub and has_secret(text):
                drop_secret += 1
                continue

            if enable_license_filter and not is_license_allowed(text, license_allowlist_t):
                drop_license += 1
                continue

            seen_hashes.add(sha)
            pua_codepoints.update(find_pua_codepoints(text))

            rec = CorpusRecord(
                path=str(path.relative_to(corpus_dir)).replace("\\", "/"),
                text=text,
                sha256=sha,
            )
            line = rec.to_json() + b"\n"

            assert shard_fh is not None
            if bytes_in_shard + len(line) > shard_size_bytes and bytes_in_shard > 0:
                shard_fh.close()
                shard_idx += 1
                shard_path = shards_dir / f"shard_{shard_idx:05d}.jsonl.gz"
                shard_fh = gzip.open(shard_path, "wb")  # noqa: SIM115
                bytes_in_shard = 0

            shard_fh.write(line)
            bytes_in_shard += len(line)
            bytes_kept += len(text.encode("utf-8"))
            files_kept += 1
    finally:
        if shard_fh is not None:
            shard_fh.close()

    return IngestStats(
        files_seen=files_seen,
        files_kept=files_kept,
        files_dropped_dedup=drop_dedup,
        files_dropped_secret=drop_secret,
        files_dropped_license=drop_license,
        files_dropped_decode=drop_decode,
        files_dropped_size=drop_size,
        bytes_kept=bytes_kept,
        pua_codepoints_in_corpus=frozenset(pua_codepoints),
    )


def iter_shards(shards_dir: Path) -> Iterator[Path]:
    """Yield shard paths in deterministic order."""
    shards = sorted(shards_dir.glob("shard_*.jsonl.gz"))
    yield from shards


def read_shard(shard_path: Path) -> Iterator[CorpusRecord]:
    """Stream records from one shard."""
    with gzip.open(shard_path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield CorpusRecord.from_json(line)


def iter_shard_texts(shards_dir: Path) -> Iterator[str]:
    """Stream text payloads from all shards in order."""
    for shard in iter_shards(shards_dir):
        for rec in read_shard(shard):
            yield rec.text


__all__ = [
    "SECRET_PATTERNS",
    "CorpusRecord",
    "IngestStats",
    "detect_license",
    "has_secret",
    "ingest_corpus",
    "is_license_allowed",
    "iter_corpus_files",
    "iter_shard_texts",
    "iter_shards",
    "read_shard",
]
