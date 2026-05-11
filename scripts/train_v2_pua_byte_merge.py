"""Retrain BPE with PUA chars in `initial_alphabet`, no AddedToken safety net.

This is the load-bearing experiment from the v2 PRD: with PUA chars
treated as ordinary symbols by the BPE trainer, merges like
``Ġ + PUA("def")`` (whitespace + ``def`` PUA → single token) become
learnable. Today's CUTE registers PUA chars as ``AddedToken``s after
training, which makes them match atomically *before* any BPE merge runs
— eliminating exactly these whitespace+PUA merges that account for
substantial code-corpus compression.

Skips ingest (uses pre-built shards from ``C:/Users/husse/Downloads/CUTE/output/shards``).
Skips frequency + selection (uses existing PUA mapping from ``model/cute_mapping.json``).

Usage::

    python scripts/train_v2_pua_byte_merge.py \
        --shards C:/Users/husse/Downloads/CUTE/output/shards \
        --mapping model/cute_mapping.json \
        --output model_v2_pua_merge \
        --vocab-size 200000
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tokenizers import Tokenizer, decoders, models, trainers
from tokenizers.pre_tokenizers import ByteLevel

from cute_tokenizer.corpus import iter_shard_texts
from cute_tokenizer.pretokenizer import pretokenize_to_string
from cute_tokenizer.pua import PUAMapping
from cute_tokenizer.trainer import load_mapping


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shards",
        default=r"C:\Users\husse\Downloads\CUTE\output\shards",
        help="dir with shard_*.jsonl.gz",
    )
    ap.add_argument("--mapping", default="model/cute_mapping.json")
    ap.add_argument("--output", default="model_v2_pua_merge")
    ap.add_argument("--vocab-size", type=int, default=200_000)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument(
        "--register-added-tokens",
        action="store_true",
        help="Register PUA chars as AddedTokens after training (defeats the experiment).",
    )
    args = ap.parse_args()

    shards = Path(args.shards)
    if not shards.exists():
        print(f"ERROR: shards dir not found: {shards}", file=sys.stderr)
        return 1

    print(f"Loading PUA mapping from {args.mapping} ...")
    mapping: PUAMapping = load_mapping(Path(args.mapping))
    print(f"  {mapping.size:,} PUA entries")

    # The KEY change: PUA chars go into initial_alphabet so BPE treats
    # them as starting symbols rather than 4-byte UTF-8 sequences. This
    # lets BPE learn merges like " " + PUA("def") → single token.
    pua_alphabet = list(mapping.pua_chars)
    byte_alphabet = list(ByteLevel.alphabet())
    full_alphabet = byte_alphabet + pua_alphabet
    print(f"Initial alphabet: {len(byte_alphabet)} byte + {len(pua_alphabet)} PUA = {len(full_alphabet)}")

    # Vocab budget: vocab_size = byte_alphabet + pua_alphabet + special_tokens + bpe_merges
    special_tokens = [
        "<|endoftext|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|file_sep|>",
        "<|repo_name|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|im_sep|>",
    ]
    bpe_merge_budget = args.vocab_size - len(special_tokens) - len(full_alphabet)
    print(
        f"Vocab budget: {args.vocab_size} = {len(special_tokens)} special "
        f"+ {len(full_alphabet)} alphabet + {bpe_merge_budget} BPE merges"
    )

    if bpe_merge_budget < 50_000:
        print(
            f"ERROR: BPE merge budget too small ({bpe_merge_budget} < 50,000). "
            f"Increase --vocab-size.",
            file=sys.stderr,
        )
        return 1

    print("\nBuilding tokenizer ...")
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True, trim_offsets=True)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=full_alphabet,
        min_frequency=args.min_frequency,
        show_progress=True,
    )

    # PUA-substituted training stream — same as the existing trainer.
    def substituted_stream():
        n = 0
        for text in iter_shard_texts(shards):
            yield pretokenize_to_string(text, mapping)
            n += 1
            if n % 5000 == 0:
                print(f"  fed {n:,} texts", flush=True)

    print("\nTraining BPE on PUA-substituted stream ...")
    t0 = time.perf_counter()
    tok.train_from_iterator(substituted_stream(), trainer=trainer)
    train_time = time.perf_counter() - t0
    print(f"BPE training: {train_time:.1f}s")

    # The CRITICAL difference: NOT registering PUA chars as AddedTokens.
    # The original trainer adds them after training as a safety net:
    #     tokenizer.add_tokens([AddedToken(ch) for ch in mapping.pua_chars
    #                            if ch not in existing_vocab])
    # We deliberately skip this so BPE merges can fire across PUA boundaries.
    if args.register_added_tokens:
        from tokenizers import AddedToken
        existing_vocab = tok.get_vocab()
        pua_added = [
            AddedToken(ch, single_word=False, lstrip=False, rstrip=False,
                       normalized=False, special=False)
            for ch in mapping.pua_chars if ch not in existing_vocab
        ]
        if pua_added:
            tok.add_tokens(pua_added)
            print(f"Registered {len(pua_added)} PUA chars as AddedTokens (safety net).")
    else:
        # Verify how many PUA chars made it into the BPE vocab on their own.
        existing_vocab = tok.get_vocab()
        in_vocab = sum(1 for ch in mapping.pua_chars if ch in existing_vocab)
        print(f"PUA chars in BPE vocab: {in_vocab:,}/{mapping.size:,}")
        missing = [ch for ch in mapping.pua_chars if ch not in existing_vocab]
        if missing:
            print(
                f"  WARNING: {len(missing)} PUA chars not in vocab. "
                f"Inputs containing them will fall back to byte-level encoding."
            )

    # Save
    out_dir = Path(args.output)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    tok.save(str(out_dir / "tokenizer.json"))
    shutil.copyfile(args.mapping, out_dir / "cute_mapping.json")
    if Path("model/tokenizer_config.json").is_file():
        shutil.copyfile("model/tokenizer_config.json", out_dir / "tokenizer_config.json")

    manifest = {
        "experiment": "pua_byte_merge_v2",
        "vocab_size": args.vocab_size,
        "initial_alphabet_size": len(full_alphabet),
        "pua_chars_in_initial_alphabet": len(pua_alphabet),
        "register_added_tokens": args.register_added_tokens,
        "train_time_seconds": train_time,
        "final_vocab_size": len(tok.get_vocab()),
    }
    (out_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved to {out_dir} (vocab size: {len(tok.get_vocab()):,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
