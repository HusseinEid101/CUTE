"""Rust acceleration loader + Python fallback gate.

Importing this module exposes:

- ``accel``: the loaded Rust extension module, or ``None`` if unavailable.
- ``USE_RUST``: bool. ``True`` iff the Rust path should be used.
- ``prepare_mapping(pua_mapping)``: returns a cached ``PreparedMapping``
  for the given ``PUAMapping`` instance, building it lazily on first use.

The Rust path is the default. To force the pure-Python path (for parity
testing or debugging), set ``CUTE_USE_PYTHON_PRETOKENIZER=1`` before
importing :mod:`cute_tokenizer`.
"""

from __future__ import annotations

import contextlib
import os
import weakref
from typing import Any

_FORCE_PY = os.environ.get("CUTE_USE_PYTHON_PRETOKENIZER", "") not in (
    "",
    "0",
    "false",
    "False",
)

try:  # pragma: no cover - import-time path differs per environment
    from cute_tokenizer import _accel as _accel_module  # type: ignore[attr-defined]

    _ACCEL_AVAILABLE = True
except ImportError:
    _accel_module = None  # type: ignore[assignment]
    _ACCEL_AVAILABLE = False

accel: Any = _accel_module
USE_RUST: bool = _ACCEL_AVAILABLE and not _FORCE_PY

# Cache: keyed by id(PUAMapping). Frozen dataclasses support weakref.finalize,
# so we drop the entry when the mapping object is collected.
_prepared_cache: dict[int, Any] = {}


def _release(key: int) -> None:
    _prepared_cache.pop(key, None)


def prepare_mapping(mapping: Any) -> Any:
    """Return a cached ``PreparedMapping`` for ``mapping``.

    Builds the prepared form on first call per mapping object; subsequent
    calls with the same object return the cached handle. The cache uses
    ``id(mapping)`` as key and a weakref finalizer to drop entries when
    the mapping is garbage collected.

    Raises ``RuntimeError`` if the Rust extension is not available.
    """
    if not _ACCEL_AVAILABLE:
        raise RuntimeError("cute_tokenizer._accel is not available")
    key = id(mapping)
    cached = _prepared_cache.get(key)
    if cached is not None:
        return cached
    prepared = accel.PreparedMapping(mapping.word_to_pua, mapping.pua_to_word)
    _prepared_cache[key] = prepared
    # Mapping objects without `__weakref__` (very rare) silently skip
    # the finalizer; the entry then lives until process exit, which is
    # fine because the prepared mapping is small and the dict caps at
    # one entry per distinct PUAMapping instance.
    with contextlib.suppress(TypeError):
        weakref.finalize(mapping, _release, key)
    return prepared


def build_tag() -> str:
    """Return the Rust extension build tag (or 'python-fallback')."""
    if USE_RUST:
        return getattr(accel, "__build_tag__", "rust-unknown")
    return "python-fallback"


__all__ = ["USE_RUST", "accel", "build_tag", "prepare_mapping"]
