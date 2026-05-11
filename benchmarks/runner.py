"""Benchmark runner — produces a reproducible JSON + Markdown report
comparing CUTE against any available baseline tokenizer on a held-out
corpus.

Metrics per tokenizer:
    - mean / p50 / p95 / p99 token count per file
    - bytes per token (overall)
    - encode latency: p50 / p95 / p99 microseconds per file
    - decode latency: same
    - peak RSS delta during encoding pass

Usage:
    python -m benchmarks.runner \\
        --tokenizer ./output_v6 \\
        --holdout ./holdout/python \\
        --output reports/v6.{json,md} \\
        --max-files 1500
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .baselines import Baseline, load_all


@dataclass
class TokStats:
    name: str
    note: str
    vocab_size: int
    files_evaluated: int
    # Token-count distribution per file
    mean_tokens: float
    p50_tokens: float
    p95_tokens: float
    p99_tokens: float
    bytes_per_token: float
    # Latency distribution per file (microseconds per encode/decode call)
    encode_p50_us: float
    encode_p95_us: float
    encode_p99_us: float
    decode_p50_us: float
    decode_p99_us: float
    # Memory delta during the full encode pass (MB peak − MB baseline)
    rss_delta_mb: float
    # Roundtrip integrity
    roundtrip_ok: int
    roundtrip_failed: int


@dataclass
class BenchReport:
    cute_dir: str
    holdout_dir: str
    max_files: int
    tokenizers: list[TokStats] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=dict)
    timestamp_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))
    return float(s[k])


def _peak_rss_mb() -> float:
    """Cross-platform peak RSS in MB. Falls back to 0 on unsupported platforms."""
    try:
        import psutil  # type: ignore[import-not-found]
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except (ImportError, AttributeError):
            return 0.0


def _evaluate_one(b: Baseline, files: list[Path]) -> TokStats:
    """Run one baseline over all files. Returns full stats."""
    counts: list[int] = []
    encode_times_us: list[float] = []
    decode_times_us: list[float] = []
    total_bytes = 0
    total_tokens = 0
    n_evaluated = 0
    n_roundtrip_ok = 0
    n_roundtrip_failed = 0

    gc.collect()
    rss_start = _peak_rss_mb()
    rss_peak = rss_start

    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        # Encode timing
        t0 = time.perf_counter_ns()
        ids = b.encode(text)
        encode_times_us.append((time.perf_counter_ns() - t0) / 1000.0)

        # Decode timing
        t0 = time.perf_counter_ns()
        decoded = b.decode(ids)
        decode_times_us.append((time.perf_counter_ns() - t0) / 1000.0)

        if decoded == text:
            n_roundtrip_ok += 1
        else:
            n_roundtrip_failed += 1

        counts.append(len(ids))
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(ids)
        n_evaluated += 1

        rss_peak = max(rss_peak, _peak_rss_mb())

    return TokStats(
        name=b.name,
        note=b.note,
        vocab_size=b.vocab_size,
        files_evaluated=n_evaluated,
        mean_tokens=(statistics.mean(counts) if counts else 0.0),
        p50_tokens=_percentile([float(c) for c in counts], 50),
        p95_tokens=_percentile([float(c) for c in counts], 95),
        p99_tokens=_percentile([float(c) for c in counts], 99),
        bytes_per_token=(total_bytes / max(1, total_tokens)),
        encode_p50_us=_percentile(encode_times_us, 50),
        encode_p95_us=_percentile(encode_times_us, 95),
        encode_p99_us=_percentile(encode_times_us, 99),
        decode_p50_us=_percentile(decode_times_us, 50),
        decode_p99_us=_percentile(decode_times_us, 99),
        rss_delta_mb=max(0.0, rss_peak - rss_start),
        roundtrip_ok=n_roundtrip_ok,
        roundtrip_failed=n_roundtrip_failed,
    )


def _gather_files(holdout: Path, max_files: int) -> list[Path]:
    extensions = (".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp",
                  ".rs", ".go", ".rb", ".php", ".cs", ".swift", ".kt")
    out: list[Path] = []
    for p in sorted(holdout.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in extensions:
            continue
        out.append(p)
        if len(out) >= max_files:
            break
    return out


def run(
    cute_dir: Path,
    holdout: Path,
    *,
    max_files: int = 1500,
    json_out: Path | None = None,
    md_out: Path | None = None,
) -> BenchReport:
    files = _gather_files(holdout, max_files)
    if not files:
        raise RuntimeError(f"no files found under {holdout}")

    print(f"loading baselines (cute_dir={cute_dir})", flush=True)
    baselines = load_all(cute_dir)
    if not baselines:
        raise RuntimeError("no baselines loaded; check tokenizer dir + dependencies")

    versions = _collect_versions()

    from datetime import datetime, timezone
    report = BenchReport(
        cute_dir=str(cute_dir),
        holdout_dir=str(holdout),
        max_files=len(files),
        versions=versions,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )

    print(f"\nevaluating {len(files):,} files\n", flush=True)
    for b in baselines:
        print(f"  -> {b.name} ...", flush=True)
        stats = _evaluate_one(b, files)
        report.tokenizers.append(stats)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nJSON -> {json_out}")

    if md_out is not None:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(_render_markdown(report), encoding="utf-8")
        print(f"Markdown -> {md_out}")

    return report


def _collect_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for mod_name in ("cute_tokenizer", "tokenizers", "transformers", "tiktoken", "regex"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[mod_name] = "n/a"
    import sys
    versions["python"] = sys.version.split()[0]
    return versions


def _render_markdown(report: BenchReport) -> str:
    """Render a clean, copy-paste-friendly Markdown report."""
    lines: list[str] = []
    lines.append("# CUTE benchmark report")
    lines.append("")
    lines.append(f"- **CUTE tokenizer**: `{report.cute_dir}`")
    lines.append(f"- **Holdout**: `{report.holdout_dir}`")
    lines.append(f"- **Files evaluated**: {report.max_files:,}")
    lines.append(f"- **Timestamp (UTC)**: {report.timestamp_utc}")
    lines.append("")
    lines.append("Library versions:")
    for k, v in sorted(report.versions.items()):
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")

    cute = next((t for t in report.tokenizers if t.name == "CUTE"), None)

    lines.append("## Compression")
    lines.append("")
    lines.append(
        "| Tokenizer | mean | p50 | p95 | p99 | bytes/tok | vocab | vs CUTE | roundtrip |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for t in report.tokenizers:
        ratio = (
            f"{(t.mean_tokens / cute.mean_tokens):.3f}×"
            if cute and cute.mean_tokens
            else "—"
        )
        rt = (
            f"{t.roundtrip_ok}/{t.roundtrip_ok + t.roundtrip_failed}"
            if (t.roundtrip_ok + t.roundtrip_failed) > 0
            else "—"
        )
        lines.append(
            f"| **{t.name}** | {t.mean_tokens:,.1f} | {t.p50_tokens:,.0f} | "
            f"{t.p95_tokens:,.0f} | {t.p99_tokens:,.0f} | {t.bytes_per_token:.2f} | "
            f"{t.vocab_size:,} | {ratio} | {rt} |"
        )
    lines.append("")
    lines.append(
        "_Lower mean tokens = better compression. Higher bytes/token = better. "
        "Ratio < 1.0× means the baseline produces fewer tokens than CUTE._"
    )
    lines.append("")

    lines.append("## Latency (microseconds per encode/decode)")
    lines.append("")
    lines.append(
        "| Tokenizer | encode p50 | encode p95 | encode p99 | decode p50 | decode p99 | RSS Δ (MB) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for t in report.tokenizers:
        lines.append(
            f"| **{t.name}** | {t.encode_p50_us:,.0f} | {t.encode_p95_us:,.0f} | "
            f"{t.encode_p99_us:,.0f} | {t.decode_p50_us:,.0f} | {t.decode_p99_us:,.0f} | "
            f"{t.rss_delta_mb:,.1f} |"
        )
    lines.append("")

    if cute is not None:
        lines.append("## Headlines")
        lines.append("")
        for t in report.tokenizers:
            if t.name == "CUTE" or not t.mean_tokens:
                continue
            ratio = cute.mean_tokens / t.mean_tokens
            if ratio < 1.0:
                pct = (1.0 - ratio) * 100
                lines.append(
                    f"- **CUTE is {pct:.1f}% more compact than {t.name}** "
                    f"({cute.mean_tokens:,.0f} vs {t.mean_tokens:,.0f} mean tokens)."
                )
            else:
                pct = (ratio - 1.0) * 100
                lines.append(
                    f"- {t.name} is {pct:.1f}% more compact than CUTE "
                    f"({t.mean_tokens:,.0f} vs {cute.mean_tokens:,.0f} mean tokens)."
                )
        lines.append("")

    return "\n".join(lines) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="CUTE tokenizer dir")
    ap.add_argument("--holdout", required=True, help="Held-out corpus dir")
    ap.add_argument("--max-files", type=int, default=1500)
    ap.add_argument("--output", default="reports/latest", help="Output prefix (writes .json + .md)")
    args = ap.parse_args(argv)

    out = Path(args.output)
    json_out = out.with_suffix(".json")
    md_out = out.with_suffix(".md")

    run(
        Path(args.tokenizer),
        Path(args.holdout),
        max_files=args.max_files,
        json_out=json_out,
        md_out=md_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
