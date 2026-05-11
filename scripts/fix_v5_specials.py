#!/usr/bin/env python3
"""Patch v5's tokenizer.json to remove the special-token flag from
`<s>`, `</s>`, `<unk>`, `<pad>` (which collided with natural code text).

Removes the entries entirely from `added_tokens` so they're no longer
auto-matched. The trained merges (which never relied on them as boundaries)
are unaffected.

Usage: python scripts/fix_v5_specials.py <tokenizer_dir>
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


PROBLEM_TOKENS = {"<s>", "</s>", "<unk>", "<pad>"}


def main(tokenizer_dir: str) -> int:
    path = Path(tokenizer_dir) / "tokenizer.json"
    if not path.exists():
        print(f"not found: {path}", file=sys.stderr)
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))

    before = len(data.get("added_tokens", []))
    # Drop the offending entries entirely. Their IDs become "vacant" but
    # other tokens keep their IDs since added_tokens are appended at the
    # end of the vocab space.
    kept = [t for t in data.get("added_tokens", []) if t.get("content") not in PROBLEM_TOKENS]
    removed = before - len(kept)
    data["added_tokens"] = kept

    # Backup
    backup = path.with_suffix(".json.bak")
    if not backup.exists():
        shutil.copy(path, backup)
        print(f"backed up original -> {backup}")

    # Atomic write
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

    print(f"removed {removed} special-token entries (of {before} total): {sorted(PROBLEM_TOKENS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
