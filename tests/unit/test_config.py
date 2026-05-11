"""Unit tests for cute_tokenizer.config."""

from __future__ import annotations

import pytest

from cute_tokenizer.config import CUTEConfig


class TestCUTEConfig:
    def test_defaults(self) -> None:
        c = CUTEConfig()
        assert c.vocab_size == 120_000
        assert c.pua_budget == 50_000
        assert c.min_bpe_budget == 50_000
        assert c.use_savings_selection is True
        assert c.strict_pua_atomicity is True
        assert 0.0 < c.coverage_target < 1.0

    def test_invalid_coverage(self) -> None:
        with pytest.raises(ValueError, match="coverage_target"):
            CUTEConfig(coverage_target=1.5)
        with pytest.raises(ValueError, match="coverage_target"):
            CUTEConfig(coverage_target=0.0)
        with pytest.raises(ValueError, match="coverage_target"):
            CUTEConfig(coverage_target=-0.1)

    def test_invalid_vocab_size(self) -> None:
        with pytest.raises(ValueError, match="vocab_size"):
            CUTEConfig(vocab_size=100)

    def test_invalid_max_token_len(self) -> None:
        with pytest.raises(ValueError, match="max_token_len"):
            CUTEConfig(max_token_len=0)

    def test_vocab_size_floor_enforced(self) -> None:
        """vocab_size must accommodate byte alphabet + specials + pua + bpe budgets."""
        with pytest.raises(ValueError, match="too small"):
            CUTEConfig(vocab_size=2_000, pua_budget=50_000, min_bpe_budget=50_000)

    def test_negative_pua_budget(self) -> None:
        with pytest.raises(ValueError, match="pua_budget"):
            CUTEConfig(pua_budget=-1)

    def test_to_dict_round_trip(self) -> None:
        c = CUTEConfig(
            vocab_size=10_000,
            pua_budget=2_000,
            min_bpe_budget=1_000,
            coverage_target=0.85,
        )
        d = c.to_dict()
        # Reconstruct via tuple-cast for nested tuple fields
        d["extensions"] = tuple(d["extensions"])
        d["special_tokens"] = tuple(d["special_tokens"])
        d["license_allowlist"] = tuple(d["license_allowlist"])
        c2 = CUTEConfig(**d)
        assert c == c2

    def test_frozen(self) -> None:
        c = CUTEConfig()
        with pytest.raises((AttributeError, TypeError)):
            c.vocab_size = 50_000  # type: ignore[misc]
