"""Shared fixtures for the CUTE test suite."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from cute_tokenizer.pua import PUAMapping, assign_pua_mapping

# ---------------------------------------------------------------------------
# Tiny corpus fixture (Python + JS + TS files with code-like content)
# ---------------------------------------------------------------------------

_CORPUS_FILES: dict[str, str] = {
    "math.py": (
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def multiply(x, y):\n"
        "    return x * y\n\n"
        "class Calculator:\n"
        "    def __init__(self):\n"
        "        self.history = []\n\n"
        "    def calculate(self, op, a, b):\n"
        "        if op == 'add':\n"
        "            return add(a, b)\n"
        "        elif op == 'mul':\n"
        "            return multiply(a, b)\n"
        "        return None\n"
    ),
    "user.py": (
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class User:\n"
        "    user_id: int\n"
        "    user_name: str\n"
        "    is_active: bool = True\n\n"
        "def get_user_by_id(user_id):\n"
        "    return User(user_id=user_id, user_name='alice')\n"
    ),
    "utils.js": (
        "function calculateTotal(items) {\n"
        "    return items.reduce((sum, item) => sum + item.price, 0);\n"
        "}\n\n"
        "const formatCurrency = (amount) => `$${amount.toFixed(2)}`;\n\n"
        "export { calculateTotal, formatCurrency };\n"
    ),
    "service.ts": (
        "interface UserService {\n"
        "    getUser(id: number): Promise<User>;\n"
        "    updateUser(id: number, data: Partial<User>): Promise<void>;\n"
        "}\n\n"
        "class UserServiceImpl implements UserService {\n"
        "    async getUser(id: number): Promise<User> {\n"
        "        return fetch(`/users/${id}`).then(r => r.json());\n"
        "    }\n"
        "    async updateUser(id: number, data: Partial<User>): Promise<void> {\n"
        "        await fetch(`/users/${id}`, { method: 'PUT', body: JSON.stringify(data) });\n"
        "    }\n"
        "}\n"
    ),
    "emoji.md": ("# Hello 🌍\n\nThis is a test with emoji: 🚀 ✨ 🐭\nAnd ZWJ: 👨‍👩‍👧‍👦\n"),
}


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for rel, content in _CORPUS_FILES.items():
        (corpus / rel).write_text(content, encoding="utf-8")
    return corpus


@pytest.fixture
def tiny_corpus_factory(tmp_path_factory: pytest.TempPathFactory):
    """Factory for parameterized corpus tests."""

    def _make(name: str = "corpus") -> Path:
        d = tmp_path_factory.mktemp(name)
        for rel, content in _CORPUS_FILES.items():
            (d / rel).write_text(content, encoding="utf-8")
        return d

    return _make


# ---------------------------------------------------------------------------
# Synthetic mappings
# ---------------------------------------------------------------------------


@pytest.fixture
def small_mapping() -> PUAMapping:
    """Mapping over a hand-picked vocabulary used in unit tests."""
    tokens = [
        "def",
        "return",
        "self",
        "class",
        "import",
        "from",
        "function",
        "const",
        "let",
        "if",
        "else",
        "for",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        ",",
        ":",
        ";",
        "=",
        "user",
        "id",
        "name",
        "data",
        "calculate",
        "Total",
    ]
    return assign_pua_mapping(tokens)


@pytest.fixture
def empty_mapping() -> PUAMapping:
    return assign_pua_mapping([])


# ---------------------------------------------------------------------------
# Frequency counter fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_freq() -> Counter[str]:
    """Realistic-looking frequency distribution."""
    return Counter(
        {
            "def": 120,
            "return": 100,
            "self": 80,
            "class": 60,
            "import": 50,
            "from": 40,
            "(": 200,
            ")": 200,
            ":": 80,
            "=": 70,
            ",": 90,
            "user_id": 30,
            "user_name": 25,
            "is_active": 20,
            "rare_token": 1,
            "another_rare": 1,
        }
    )
