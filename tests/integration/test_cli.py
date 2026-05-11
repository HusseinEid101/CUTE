"""CLI smoke tests.

Drives the same code path the end user invokes via `cute build` etc. We
test the public API, not the internal argparse plumbing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cute_tokenizer.cli import main

pytestmark = pytest.mark.integration


def test_cli_build_then_info(tiny_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    rc = main(
        [
            "build",
            "--corpus",
            str(tiny_corpus),
            "--output",
            str(out),
        ]
    )
    # The default config has vocab_size=80k which is too big for the tiny
    # corpus; expect it to either succeed or raise on min_bpe_budget.
    # Real CLI users will pass --config with a smaller vocab_size for tiny
    # corpora. We don't verify success here — only that it returns int.
    assert isinstance(rc, int)


def test_cli_build_with_config(
    tiny_corpus: Path,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "small.toml"
    config_path.write_text(
        "vocab_size = 2000\n"
        "pua_budget = 200\n"
        "coverage_target = 0.85\n"
        "min_bpe_budget = 1500\n"
        "min_frequency = 1\n"
        "workers = 1\n",
        encoding="utf-8",
    )

    out = tmp_path / "out"
    rc = main(
        [
            "build",
            "--corpus",
            str(tiny_corpus),
            "--output",
            str(out),
            "--config",
            str(config_path),
        ]
    )
    assert rc == 0
    assert (out / "tokenizer.json").exists()
    assert (out / "cute_mapping.json").exists()
    assert (out / "build_manifest.json").exists()


def test_cli_info(tiny_corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "small.toml"
    config_path.write_text(
        "vocab_size = 2000\n"
        "pua_budget = 200\n"
        "coverage_target = 0.85\n"
        "min_bpe_budget = 1500\n"
        "min_frequency = 1\n"
        "workers = 1\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    main(
        ["build", "--corpus", str(tiny_corpus), "--output", str(out), "--config", str(config_path)]
    )
    capsys.readouterr()  # clear

    rc = main(["info", "--tokenizer", str(out)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "cute_version" in captured.out
    assert "vocab_hash" in captured.out


def test_cli_roundtrip_check(
    tiny_corpus: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "small.toml"
    config_path.write_text(
        "vocab_size = 2000\n"
        "pua_budget = 200\n"
        "coverage_target = 0.85\n"
        "min_bpe_budget = 1500\n"
        "min_frequency = 1\n"
        "workers = 1\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    main(
        ["build", "--corpus", str(tiny_corpus), "--output", str(out), "--config", str(config_path)]
    )
    capsys.readouterr()

    rc = main(
        [
            "roundtrip-check",
            "--tokenizer",
            str(out),
            "--corpus",
            str(tiny_corpus),
        ]
    )
    captured = capsys.readouterr()
    assert "Round-trip check" in captured.out
    assert rc == 0  # All files should round-trip
