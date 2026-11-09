#!/usr/bin/env python3
"""Check draft model vocabulary coverage (Chinese token presence).

Usage:
    python3 check_vocab.py /path/to/checkpoint/model.safetensors
    python3 check_vocab.py /path/to/checkpoint/model.safetensors --tokenizer /path/to/model
"""
import sys
from pathlib import Path

from safetensors import safe_open


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <model.safetensors> [--tokenizer <model_path>]")
        sys.exit(1)

    ckpt = Path(sys.argv[1])
    tokenizer_path = None
    if "--tokenizer" in sys.argv:
        idx = sys.argv.index("--tokenizer")
        tokenizer_path = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None

    with safe_open(str(ckpt), framework="pt") as f:
        all_keys = sorted(f.keys())
        print("=== vocab-related weights:")
        for k in all_keys:
            if any(x in k for x in ("d2t", "t2d", "lm_head", "embed_tokens")):
                shape = f.get_slice(k).get_shape()
                print(f"  {k}: {shape}")

        # d2t = draft token id -> target token id mapping
        d2t = None
        for prefix in ("", "model."):
            key = f"{prefix}d2t"
            if key in f.keys():
                d2t = f.get_tensor(key)
                print(f"\ndraft vocab size: {len(d2t)}")
                print(f"d2t valid mappings: {(d2t != 0).sum().item()}")
                break

        if d2t is not None:
            # Build a set of target token IDs covered by the draft vocab
            covered = set(d2t[d2t != 0].tolist())
            print(f"unique target tokens covered: {len(covered)}")

            if tokenizer_path:
                check_chinese_coverage(covered, tokenizer_path)
            else:
                print("\n(pass --tokenizer <model_path> to check Chinese token coverage)")
            return

        # No d2t found -- check lm_head to determine vocab mode
        for prefix in ("", "model."):
            key = f"{prefix}lm_head.weight"
            if key in f.keys():
                shape = f.get_slice(key).get_shape()
                print(f"\nlm_head shape: {shape}")
                if shape[0] >= 152064:
                    print("-> FULL vocab mode (no pruning)")
                elif shape[0] <= 40000:
                    print(f"-> PRUNED to {shape[0]} tokens")
                    if tokenizer_path:
                        check_chinese_coverage_from_lmhead(
                            f, key, shape[0], tokenizer_path
                        )
                break


def check_chinese_coverage(covered_target_ids, tokenizer_path):
    """Check whether common Chinese tokens are in the draft vocabulary."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    words = ["我们", "时候", "因为", "所以", "可以", "知道", "这个",
             "什么", "就是", "没有", "这样", "如果", "但是", "现在"]
    print(f"\n=== Chinese token coverage (target token in draft vocab?):")
    missing = 0
    for w in words:
        tids = tok.encode(w, add_special_tokens=False)
        hits = [t in covered_target_ids for t in tids]
        status = "OK" if all(hits) else "MISSING"
        if not all(hits):
            missing += 1
        print(f"  {w}: {tids} -> {status}")

    if missing > 0:
        print(f"\n!! {missing}/{len(words)} Chinese words MISSING from draft vocab")
        print("!! This explains low Chinese acceptance rates.")
    else:
        print(f"\nAll {len(words)} Chinese words covered.")


def check_chinese_coverage_from_lmhead(f, key, vocab_size, tokenizer_path):
    """When lm_head is pruned, check coverage via embed_tokens shape."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # In pruned mode, draft token 0..N-1 maps to target via d2t.
    # Without d2t we can't check exact coverage, but we can report sizes.
    print(f"\nDraft vocab: {vocab_size} / target vocab: 152064")
    print(f"Coverage ratio: {vocab_size / 152064 * 100:.1f}%")
    print("\n(pass --tokenizer to check specific Chinese tokens)")


if __name__ == "__main__":
    main()
