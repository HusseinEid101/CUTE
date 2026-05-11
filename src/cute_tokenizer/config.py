"""Configuration for the CUTE tokenizer build pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _load_toml(path: Path) -> dict[str, Any]:
    """Load TOML file, handling both stdlib tomllib (3.11+) and tomli."""
    try:
        import tomllib  # Python 3.11+

        with open(path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        try:
            import tomli

            with open(path, "rb") as f:
                return tomli.load(f)
        except ImportError as e:
            raise ImportError(
                "Install 'tomli' for TOML support on Python <3.11: pip install tomli"
            ) from e


DEFAULT_SPECIAL_TOKENS: tuple[str, ...] = (
    # Pipe-style markers only. We deliberately exclude `<s>`, `</s>`, `<unk>`,
    # `<pad>` because they collide with natural text in code (e.g. Python
    # docstrings use `<unk>` as a placeholder, NLP code prints `<s>`/`</s>`),
    # which would cause silent roundtrip loss when those substrings appear
    # in a file. Industry practice (cl100k, gpt2, starcoder2) follows the
    # same pipe-style convention.
    "<|endoftext|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|file_sep|>",
    "<|repo_name|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|im_sep|>",
)

DEFAULT_CODE_EXTENSIONS: tuple[str, ...] = (
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".rs",
    ".go",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".sh",
    ".sql",
    ".html",
    ".css",
    ".scss",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
)


@dataclass(frozen=True)
class CUTEConfig:
    """All knobs for a CUTE build, in one place.

    Frozen so hashing/comparison is well-defined and the manifest serializer
    can dump a stable representation.
    """

    vocab_size: int = 120_000
    pua_budget: int = 50_000
    coverage_target: float = 0.90  # used only by deprecated select_by_coverage path
    max_token_len: int = 50
    boost_weight: float = 0.3
    min_bpe_budget: int = 50_000
    min_frequency: int = 2
    seed: int = 42
    allow_supplementary_pua: bool = False
    use_savings_selection: bool = True
    strict_pua_atomicity: bool = True
    # When True, do not use BMP PUA codepoints (U+E000-U+F8FF) for assignments.
    # BMP PUAs occasionally appear in real source text (Asian fonts, Unicode
    # mapping tables in TS/JS) which causes decode-time collisions: the user's
    # literal PUA char gets reverse-substituted into the mapped word.
    # Setting True forces all assignments to supplementary planes (4-byte UTF-8)
    # which are virtually never used in real text. Production-recommended.
    pua_skip_bmp: bool = True
    extensions: tuple[str, ...] = DEFAULT_CODE_EXTENSIONS
    special_tokens: tuple[str, ...] = DEFAULT_SPECIAL_TOKENS
    workers: int = 0  # 0 means os.cpu_count()
    shard_size_bytes: int = 64 * 1024 * 1024  # 64 MiB per shard
    license_allowlist: tuple[str, ...] = (
        "MIT",
        "Apache-2.0",
        "BSD-3-Clause",
        "BSD-2-Clause",
        "ISC",
        "Apache 2.0",
        "Apache License 2.0",
    )
    enable_secret_scrub: bool = True
    enable_license_filter: bool = False  # off by default; opt-in

    def __post_init__(self) -> None:
        if not 0.0 < self.coverage_target < 1.0:
            raise ValueError(f"coverage_target must be in (0,1), got {self.coverage_target}")
        if self.vocab_size < 1024:
            raise ValueError(f"vocab_size too small: {self.vocab_size}")
        if self.max_token_len < 1:
            raise ValueError(f"max_token_len must be positive: {self.max_token_len}")
        if self.pua_budget < 0:
            raise ValueError(f"pua_budget must be non-negative: {self.pua_budget}")
        if self.min_bpe_budget < 256:
            raise ValueError(
                f"min_bpe_budget must be ≥ 256 (byte alphabet), got {self.min_bpe_budget}"
            )
        # Vocab math: byte alphabet (256) + special_tokens + pua_budget + min_bpe_budget ≤ vocab_size
        floor = 256 + len(self.special_tokens) + self.pua_budget + self.min_bpe_budget
        if floor > self.vocab_size:
            raise ValueError(
                f"vocab_size={self.vocab_size} too small for "
                f"pua_budget={self.pua_budget} + min_bpe_budget={self.min_bpe_budget} "
                f"+ specials={len(self.special_tokens)} + bytes=256 (need ≥ {floor})"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_toml(cls, path: Path) -> CUTEConfig:
        """Load config from TOML file (e.g., configs/default.toml).

        Falls back to defaults for any missing keys.
        """
        data = _load_toml(path)
        # Map TOML keys to dataclass fields
        kwargs: dict[str, Any] = {}

        # Simple scalar fields
        for key in [
            "vocab_size",
            "pua_budget",
            "coverage_target",
            "max_token_len",
            "boost_weight",
            "min_bpe_budget",
            "min_frequency",
            "seed",
            "workers",
            "shard_size_bytes",
            "enable_secret_scrub",
            "enable_license_filter",
            "allow_supplementary_pua",
            "use_savings_selection",
            "strict_pua_atomicity",
            "pua_skip_bmp",
        ]:
            if key in data:
                kwargs[key] = data[key]

        # Tuple fields
        if "extensions" in data:
            kwargs["extensions"] = tuple(data["extensions"])
        if "special_tokens" in data:
            kwargs["special_tokens"] = tuple(data["special_tokens"])
        if "license_allowlist" in data:
            kwargs["license_allowlist"] = tuple(data["license_allowlist"])

        return cls(**kwargs)


__all__ = ["DEFAULT_CODE_EXTENSIONS", "DEFAULT_SPECIAL_TOKENS", "CUTEConfig"]
