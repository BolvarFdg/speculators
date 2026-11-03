#!/usr/bin/env python3
"""Patch a trained DSpark draft's config.json to satisfy vLLM's Qwen3-Omni
contract (_validate_qwen3_omni_dspark).

Some contract fields are pure metadata that a checkpoint trained before the
export fixes landed may be missing; the weights are unaffected. This script
patches the safe metadata fields and verifies (without touching) the fields
that must match how the draft was actually trained.

Usage:
    python3 fix_draft_config.py <draft_checkpoint_dir> \
        --target-hidden 2048 --num-speculative-tokens 7

    <draft_checkpoint_dir> is e.g. $OUTPUT_DIR/checkpoints/0
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROPED_KEYS = ("mrope_section", "mrope_interleaved", "type")


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="draft checkpoint dir, e.g. .../checkpoints/0")
    parser.add_argument("--target-hidden", type=int, required=True,
                        help="thinker text hidden size (2048 for Qwen3-Omni-30B-A3B)")
    parser.add_argument("--num-speculative-tokens", type=int, default=7,
                        help="serving num_speculative_tokens; must equal block_size")
    parser.add_argument("--vocab-size", type=int, default=152064,
                        help="target (thinker text) vocab size")
    args = parser.parse_args()

    cfg_path = Path(args.checkpoint) / "config.json"
    if not cfg_path.is_file():
        _fail(f"{cfg_path} not found")
    cfg = json.loads(cfg_path.read_text())

    # ---- fields safe to patch (pure serving-side metadata) ----
    patched = []
    if cfg.get("architectures") != ["Qwen3OmniDSparkModel"]:
        cfg["architectures"] = ["Qwen3OmniDSparkModel"]
        patched.append("architectures")
    if cfg.get("use_aux_hidden_state") is not True:
        cfg["use_aux_hidden_state"] = True
        patched.append("use_aux_hidden_state")
    if cfg.get("target_hidden_size") != args.target_hidden:
        cfg["target_hidden_size"] = args.target_hidden
        patched.append("target_hidden_size")

    # Draft must use logical 1-D RoPE: strip MRoPE channels inherited from the
    # thinker text config.
    tlc = cfg.get("transformer_layer_config") or {}
    for key in ("rope_parameters", "rope_scaling"):
        rope = tlc.get(key)
        if isinstance(rope, dict):
            for rk in ROPED_KEYS:
                if rk in rope:
                    del rope[rk]
                    patched.append(f"transformer_layer_config.{key}.{rk}")

    # ---- fields that must match training; verify only, never rewrite ----
    if cfg.get("sample_from_anchor") is not True:
        _fail("sample_from_anchor is not True -- the draft was trained with "
              "the 1+N fill-in layout, which the Qwen3-Omni contract forbids. "
              "Retrain with sample_from_anchor=True (the DSpark default).")
    if cfg.get("markov_head_type") != "vanilla":
        _fail("markov_head_type is not 'vanilla' -- the Qwen3-Omni contract "
              "requires vanilla; retrain or relax the serving-side check.")
    if cfg.get("block_size") != args.num_speculative_tokens:
        _fail(f"block_size={cfg.get('block_size')} != "
              f"num_speculative_tokens={args.num_speculative_tokens}; serve "
              "with a matching num_speculative_tokens or retrain.")
    if not cfg.get("aux_hidden_state_layer_ids"):
        _fail("aux_hidden_state_layer_ids missing")
    if not isinstance(cfg.get("markov_rank"), int) or cfg["markov_rank"] <= 0:
        _fail("markov_rank must be a positive integer")
    mask = cfg.get("mask_token_id")
    if not isinstance(mask, int) or not 0 <= mask < args.vocab_size:
        _fail(f"mask_token_id={mask} outside [0, {args.vocab_size})")
    dv = cfg.get("draft_vocab_size")
    if not isinstance(dv, int) or not 0 < dv <= args.vocab_size:
        _fail(f"draft_vocab_size={dv} outside (0, {args.vocab_size}]")

    backup = cfg_path.with_suffix(".json.bak")
    if not backup.exists():
        shutil.copy2(cfg_path, backup)
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    print(f"patched fields: {patched or '(none needed)'}")
    print(f"backup: {backup}")
    print("config now satisfies the vLLM Qwen3-Omni DSpark contract; "
          "weights untouched.")


if __name__ == "__main__":
    main()
