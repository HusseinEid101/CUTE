"""Token regex and identifier-splitting helpers.

Uses the third-party `regex` module (NOT stdlib `re`) so that Unicode property
classes like `\\p{Emoji_Presentation}` work. This is a hard requirement for
the pure-Python fallback. The Rust extension uses the `regex` crate's own
Unicode tables; the two are kept in parity by `tests/property/test_python_rust_parity.py`.
"""

from __future__ import annotations

from collections.abc import Iterator

import regex as re

from ._accel_loader import USE_RUST, accel

# The token regex covers, in priority order:
#   1. Emoji sequences (incl. ZWJ, VS16, keycap, emoji modifiers)
#   2. Word tokens (letters, digits, underscores, internal hyphens)
#   3. Multi-char operators (==, !=, <=, >=, +=, ->, &&, ||, :=, etc.)
#   4. Single non-space punctuation
#
# Note: we deliberately do NOT match \s+ here. Whitespace is preserved via
# gap-fill in the pre-tokenizer (fix #6 from the build plan), avoiding the
# double-counting bug present in the v2.1 draft.
TOKEN_REGEX = re.compile(
    r"""
    (?:
        [\p{Emoji_Presentation}\p{Extended_Pictographic}]
        (?:‍[\p{Emoji_Presentation}\p{Extended_Pictographic}])*
        [️⃣\p{Emoji_Modifier}]*
    )+                                                  # emoji sequence
    | [\p{L}\p{N}_](?:[\p{L}\p{N}_\-]*[\p{L}\p{N}_])?   # word / identifier
    | (?:!=|==|<=|>=|<<|>>|\+=|-=|\*=|/=|%=|&&|\|\||->|=>|::|:=|\.\.\.|\.\.)
    | [^\s\w]                                            # single punctuation
    """,
    re.VERBOSE | re.UNICODE,
)

# A strict ASCII identifier matcher; we only sub-split on ASCII identifiers
# because non-ASCII identifiers don't have well-defined camelCase semantics.
IDENTIFIER_REGEX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Sub-part splitter for camelCase / PascalCase / SCREAMING_CASE / snake_case / digits.
_SUBPART_REGEX = re.compile(
    r"[A-Z]+(?=[A-Z][a-z])"  # acronym before camel: HTTPRequest -> HTTP
    r"|[A-Z]?[a-z]+"  # Capitalized or lowercase run
    r"|[A-Z]+"  # all-caps run
    r"|[0-9]+"  # digits
)


def is_identifier(s: str) -> bool:
    """True iff `s` matches the conservative ASCII identifier shape."""
    if USE_RUST:
        return accel.is_identifier(s)
    return bool(IDENTIFIER_REGEX.match(s))


def split_identifier(ident: str) -> list[str]:
    """Split camelCase / PascalCase / snake_case / SCREAMING_CASE into pieces.

    Property: ``''.join(split_identifier(x))`` reconstructs `x` minus underscores.

    Examples
    --------
    >>> split_identifier("myVar")
    ['my', 'Var']
    >>> split_identifier("HTTPRequestParser")
    ['HTTP', 'Request', 'Parser']
    >>> split_identifier("MAX_BUFFER_SIZE")
    ['MAX', 'BUFFER', 'SIZE']
    >>> split_identifier("get_user_id_42")
    ['get', 'user', 'id', '42']
    >>> split_identifier("")
    []
    """
    if USE_RUST:
        return accel.split_identifier(ident)
    if not ident:
        return []
    parts: list[str] = []
    for chunk in ident.split("_"):
        if not chunk:
            continue
        sub = _SUBPART_REGEX.findall(chunk)
        if sub:
            parts.extend(sub)
        else:
            parts.append(chunk)
    return parts


def iter_tokens(text: str) -> Iterator[tuple[str, int, int]]:
    """Yield (token, start, end) for each non-whitespace token in `text`.

    Whitespace between matches is intentionally NOT yielded; consumers are
    responsible for gap-filling if they need round-trip preservation.
    """
    for m in TOKEN_REGEX.finditer(text):
        yield m.group(), m.start(), m.end()


__all__ = [
    "IDENTIFIER_REGEX",
    "TOKEN_REGEX",
    "is_identifier",
    "iter_tokens",
    "split_identifier",
]
