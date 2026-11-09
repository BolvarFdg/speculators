#!/usr/bin/env python3
"""Check draft model vocabulary coverage (Chinese token presence).

Correctly decodes d2t as OFFSETS (not absolute IDs): the serving side
computes `absolute_id = arange(len(d2t)) + d2t`, so we must do the same
here. The previous version read d2t values directly as token IDs, which
produced false "MISSING" results for every Chinese token.

Usage:
    python3 check_vocab.py <model.safetensors> --tokenizer <model_path>
"""
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <model.safetensors> "
              "[--tokenizer <model_path>]")
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

        d2t = None
        for prefix in ("", "model."):
            key = f"{prefix}d2t"
            if key in f.keys():
                d2t = f.get_tensor(key)
                break

        if d2t is None:
            # No d2t -- full vocab mode, no pruning
            for prefix in ("", "model."):
                key = f"{prefix}lm_head.weight"
                if key in f.keys():
                    shape = f.get_slice(key).get_shape()
                    print(f"\nlm_head shape: {shape}")
                    if shape[0] >= 152064:
                        print("-> FULL vocab mode (no pruning, no d2t needed)")
                        if tokenizer_path:
                            print("-> All tokens covered by definition")
                    break
            return

        # d2t stores OFFSETS: target_id = draft_idx + d2t[draft_idx]
        # (vocab_mapping.py:98, speculator.py:94-96)
        draft_idx = torch.arange(len(d2t))
        absolute_ids = (draft_idx + d2t).tolist()

        print(f"\ndraft vocab size: {len(d2t)}")
        print(f"d2t format: offsets (target_id = index + d2t[index])")
        print(f"zero offsets (identity-mapped head): {(d2t == 0).sum().item()}")
        print(f"unique target token IDs covered: {len(set(absolute_ids))}")

        # Sanity: absolute IDs should all be in [0, target_vocab_size)
        max_id = max(absolute_ids)
        min_id = min(absolute_ids)
        print(f"target id range: [{min_id}, {max_id}]")

        if tokenizer_path:
            check_chinese(set(absolute_ids), tokenizer_path)
        else:
            print("\n(pass --tokenizer <model_path> to check Chinese coverage)")


def check_chinese(covered_target_ids, tokenizer_path):
    """Check whether common Chinese tokens are in the draft vocabulary."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    words = ["的", "了", "是", "我", "你", "他", "这", "那", "就", "也",
             "我们", "时候", "因为", "所以", "可以", "知道", "这个",
             "什么", "就是", "没有", "这样", "如果", "但是", "现在"]
    print(f"\n=== Chinese token coverage ({len(words)} words):")
    missing = 0
    for w in words:
        tids = tok.encode(w, add_special_tokens=False)
        hits = [t in covered_target_ids for t in tids]
        status = "OK" if all(hits) else "MISSING"
        if not all(hits):
            missing += 1
        print(f"  {w} (ids={tids}): {status}")

    if missing > 0:
        print(f"\n!! {missing}/{len(words)} Chinese words MISSING from draft vocab")
    else:
        print(f"\nAll {len(words)} Chinese words covered -- vocabulary is healthy.")
        print("-> Chinese acceptance issue is NOT a vocabulary problem.")
        print("-> Investigate: Markov head quality, serving-side mapping,")
        print("   or training data distribution.")


if __name__ == "__main__":
    main()
