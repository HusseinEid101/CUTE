"""PUA-aware decoding.

After the underlying ByteLevel BPE decoder reconstructs a string, we must
substitute every PUA character back to its original word. This is a single
linear scan with a dict lookup per character — O(n).
"""

from __future__ import annotations

from ._accel_loader import USE_RUST, accel, prepare_mapping
from .pua import PUAMapping, is_pua_char


def reverse_pua_batch(texts: list[str], mapping: PUAMapping) -> list[str]:
    """Batched reverse PUA substitution. One FFI hop, Rayon-parallel."""
    if USE_RUST and hasattr(accel, "reverse_pua_batch"):
        return list(accel.reverse_pua_batch(texts, prepare_mapping(mapping)))
    return [reverse_pua_substitute(t, mapping) for t in texts]


def reverse_pua_substitute(text: str, mapping: PUAMapping) -> str:
    """Replace every PUA character in `text` with its original mapped word.

    Characters not in the mapping are passed through unchanged. This is
    safe even if the input contains PUA chars that weren't in the mapping
    (they survive the round-trip as themselves).
    """
    if USE_RUST:
        return accel.reverse_pua_substitute(text, prepare_mapping(mapping))

    pua_to_word = mapping.pua_to_word
    if not pua_to_word:
        return text

    # Fast path: if no PUA chars present, return as-is.
    if not any(is_pua_char(c) for c in text):
        return text

    out: list[str] = []
    for ch in text:
        if is_pua_char(ch):
            out.append(pua_to_word.get(ch, ch))
        else:
            out.append(ch)
    return "".join(out)


__all__ = ["reverse_pua_substitute"]
