#!/usr/bin/env python3
"""Merge sharded prepared datasets into one dataset (multi-node phase 3).

Each node rendered its own regen shard into its own prepared_shard{N}/
directory; this script concatenates them into $PREPARED.

Unlike the dspark merge there is NO token_freq.pt handling: DFlash2 trains
on the full verifier vocabulary, so no token frequency distribution is
needed for vocab pruning.

Two verification layers, any failure refuses to write the merged output:
  1. shard metadata: shard-count identical everywhere, indices cover 0..N-1
  2. row accounting: every shard non-empty, sum(shard rows) == merged rows

Usage:
    python3 merge_prepared.py \
        --work-dir /mnt/nfs/dflash2 --shard-count 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
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
    print(f"verification 1 OK: {args.shard_count} shards, indices 0..{args.shard_count - 1}")

    # ---- merge datasets + row accounting
    datasets = []
    total_rows = 0
    for i, d in enumerate(shard_dirs):
        ds = load_from_disk(str(d))
        if len(ds) == 0:
            _fail(f"shard {i} rendered 0 rows into {d} -- its upstream "
                  f"{work / f'asr_onpolicy.shard{i}.jsonl'} was probably "
                  "empty; re-run that node's phases 2-3")
        total_rows += len(ds)
        datasets.append(ds)

    merged = datasets[0]
    for ds in datasets[1:]:
        merged = concatenate_datasets([merged, ds])
    if len(merged) != total_rows:
        _fail(f"row accounting mismatch: shards hold {total_rows} rows "
              f"but concatenation produced {len(merged)}")
    print(f"verification 2 OK: {total_rows} rows across "
          f"{args.shard_count} shards")

    # ---- write the merged dataset (atomic-ish: write beside, then rename)
    prepared = work / "prepared"
    tmp = work / "prepared.merging"
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    merged.save_to_disk(str(tmp))
    if prepared.exists():
        import shutil
        shutil.rmtree(prepared)
    tmp.rename(prepared)
    print(f"merged dataset written: {prepared} ({total_rows} rows)")


if __name__ == "__main__":
    main()
