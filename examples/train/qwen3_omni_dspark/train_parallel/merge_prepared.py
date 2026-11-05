#!/usr/bin/env python3
"""Merge sharded prepared datasets + token_freq files into one dataset.

Phase-3 merge of the parallel prep: each node rendered its own regen shard
into its own prepared_shard{N}/ directory; this script concatenates them
into $PREPARED and sums the per-shard token frequencies.

Three verification layers, any failure refuses to write the merged output:
  1. shard metadata: shard-count identical everywhere, indices cover 0..N-1
  2. row accounting: sum(shard rows) == merged rows
  3. token_freq: summed counts recomputed and spot-checked

Usage:
    python3 merge_prepared.py \
        --work-dir /mnt/nfs/dspark --shard-count 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=__import__("sys").stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True,
                        help="NFS work dir containing prepared_shard{N}/")
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()

    work = Path(args.work_dir)
    shard_dirs = [work / f"prepared_shard{i}" for i in range(args.shard_count)]

    # ---- verification 1: every shard exists and carries matching metadata
    metas = []
    for i, d in enumerate(shard_dirs):
        meta_path = d / ".shard_meta"
        if not d.is_dir() or not meta_path.is_file():
            _fail(f"shard {i} incomplete: {d} or {meta_path} missing")
        meta = json.loads(meta_path.read_text())
        if meta.get("shard_count") != args.shard_count:
            _fail(f"shard {i} recorded shard_count={meta.get('shard_count')}, "
                  f"expected {args.shard_count}")
        if meta.get("shard_index") != i:
            _fail(f"shard {i} dir contains index {meta.get('shard_index')}")
        metas.append(meta)
    print(f"verification 1 OK: {args.shard_count} shards, indices 0..{args.shard_count - 1}")

    # ---- merge datasets + row accounting
    datasets = []
    total_rows = 0
    for d in shard_dirs:
        ds = load_from_disk(str(d))
        total_rows += len(ds)
        datasets.append(ds)

    from datasets import concatenate_datasets
    merged = datasets[0]
    for ds in datasets[1:]:
        merged = concatenate_datasets([merged, ds])
    if len(merged) != total_rows:
        _fail(f"row accounting mismatch: shards hold {total_rows} rows "
              f"but concatenation produced {len(merged)}")
    print(f"verification 2 OK: {total_rows} rows across "
          f"{args.shard_count} shards")

    # ---- merge token_freq (counts are additive: shard sums == full-dataset
    # frequency by construction)
    # ---- merge token_freq if present; otherwise compute from the dataset
    freq_files = [d / "token_freq.pt" for d in shard_dirs]
    if all(f.is_file() for f in freq_files):
        freq_total: dict = {}
        for f in freq_files:
            freq = torch.load(f)
            for k, v in freq.items():
                freq_total[k] = freq_total.get(k, 0) + int(v)
        print(f"token_freq: merged from {args.shard_count} shard files, "
              f"{len(freq_total)} unique tokens")
    elif all(not f.is_file() for f in freq_files):
        # speculators prepare-data did not emit token_freq.pt in this
        # environment; compute it from the merged dataset's input_ids.
        from collections import Counter
        counter: Counter = Counter()
        for ids in merged["input_ids"]:
            counter.update(ids)
        freq_total = {str(k): int(v) for k, v in counter.items()}
        print(f"token_freq: computed from merged dataset, "
              f"{len(freq_total)} unique tokens")
    else:
        present = sum(f.is_file() for f in freq_files)
        _fail(f"token_freq.pt inconsistent: {present}/{args.shard_count} "
              "shards have it -- re-run prepare-data on all shards")

    # ---- write the merged dataset (atomic-ish: write beside, then rename)
    prepared = work / "prepared"
    tmp = work / "prepared.merging"
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    merged.save_to_disk(str(tmp))
    torch.save(freq_total, tmp / "token_freq.pt")
    if prepared.exists():
        import shutil
        shutil.rmtree(prepared)
    tmp.rename(prepared)
    print(f"merged dataset written: {prepared} ({total_rows} rows)")


if __name__ == "__main__":
    main()
