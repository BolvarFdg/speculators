#!/usr/bin/env python3
"""Prepare Qwen3-Omni ASR audio data for DSpark draft training.

Two subcommands:

convert
    Convert raw ASR JSONL rows into speculators ``conversations`` rows,
    replacing ``<audio>`` placeholders with audio content parts. Two input
    layouts are accepted:
      * GSPO-style: response in a top-level ``ground_truth`` field (the
        assistant turn is empty).
      * CPO-style: chosen response in the assistant turn's ``content`` (a
        ``rejected_response`` may also be present; it is ignored -- the draft
        trains on on-policy regenerated responses only).
    Either way the extracted response is a placeholder: ``regen`` replaces it
    with the final verifier's own output.

regen
    Regenerate assistant responses against a live vLLM server while
    PRESERVING the conversations structure (unlike ``speculators
    regenerate-responses``, which emits token-only rows and drops the audio
    references needed for multimodal hidden-state extraction).

Examples:

    # Step 0 (offline): raw -> conversations (GSPO and CPO files can be
    # concatenated first, or converted in sequence with --resume to append)
    python examples/train/qwen3_omni_dspark/qwen3_omni_asr_prepare_data.py \
        convert --input raw_asr.jsonl --output asr_conversations.jsonl

    # Step 1 (with a plain `vllm serve $MODEL` running): on-policy responses
    python examples/train/qwen3_omni_dspark/qwen3_omni_asr_prepare_data.py \
        regen \
        --input asr_conversations.jsonl --output asr_onpolicy.jsonl \
        --endpoint http://localhost:8000/v1 --concurrency 64
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

from openai import OpenAI

AUDIO_PLACEHOLDER = re.compile(r"(<audio>)")


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _adapt_turn(turn: dict) -> dict:
    """Map a conversations turn onto the vLLM chat-completions format."""
    if isinstance(turn["value"], str):
        return {"role": turn["from"], "content": turn["value"]}
    parts = []
    for part in turn["value"]:
        if part.get("type") == "audio":
            url = "file://" + quote(str(Path(part["path"]).resolve()))
            parts.append({"type": "audio_url", "audio_url": {"url": url}})
        else:
            parts.append({"type": "text", "text": part.get("text", "")})
    return {"role": turn["from"], "content": parts}


def cmd_convert(args: argparse.Namespace) -> None:
    raw_rows, converted, empty_gt, fallback_asst = 0, 0, 0, 0
    missing_audio = []
    out_path = Path(args.output)
    if out_path.exists() and not args.resume:
        _fail(f"{out_path} exists; pass --resume or remove it")

    mode = "a" if args.resume and out_path.exists() else "w"
    with open(args.input) as f, out_path.open(mode) as out:
        for line in f:
            if not line.strip():
                continue
            raw_rows += 1
            row = json.loads(line)
            asst_msg = next(
                (m for m in row["messages"] if m["role"] == "assistant"), None
            )
            # GSPO rows carry ground_truth; CPO rows carry the chosen
            # response in the assistant turn. Either is a placeholder that
            # regen replaces on-policy. rejected_response is deliberately
            # ignored (the final model was trained to avoid it).
            gt = (row.get("ground_truth") or "").strip()
            if not gt and asst_msg is not None:
                gt = (asst_msg.get("content") or "").strip()
                if gt:
                    fallback_asst += 1
            if not gt:
                empty_gt += 1
                continue
            user_msg = next(
                (m for m in row["messages"] if m["role"] == "user"), None
            )
            audios = row.get("audios") or []
            if user_msg is None or not audios:
                continue

            user_parts, audio_idx = [], 0
            for seg in AUDIO_PLACEHOLDER.split(user_msg.get("content") or ""):
                if seg == "<audio>":
                    user_parts.append(
                        {"type": "audio", "path": audios[audio_idx]}
                    )
                    audio_idx += 1
                elif seg:
                    user_parts.append({"type": "text", "text": seg})

            convs = []
            sys_msg = next(
                (m for m in row["messages"] if m["role"] == "system"), None
            )
            if sys_msg and sys_msg.get("content"):
                convs.append({"from": "system", "value": sys_msg["content"]})
            convs.append({"from": "human", "value": user_parts})
            # Placeholder response: the regen step replaces it on-policy.
            convs.append({"from": "gpt", "value": gt})

            for part in user_parts:
                if part["type"] == "audio" and not Path(part["path"]).exists():
                    missing_audio.append(part["path"])
            out.write(json.dumps({"conversations": convs}, ensure_ascii=False))
            out.write("\n")
            converted += 1

    print(f"rows: {raw_rows}  converted: {converted}  "
          f"skipped(no response): {empty_gt}  "
          f"response_from_assistant_turn: {fallback_asst}")
    if missing_audio:
        print(f"missing audio files: {len(missing_audio)}")
        for p in missing_audio[:10]:
            print("  ", p)
        if not args.allow_missing_audio:
            _fail("audio files not found locally -- sync them first, or pass "
                  "--allow-missing-audio to drop rows lazily at regen time")


def _regen_one(client: OpenAI, model: str, row: dict,
               max_tokens: int, temperature: float) -> str:
    convs = row["conversations"]
    prompt_msgs = [_adapt_turn(t) for t in convs[:-1]]
    prompt_msgs = [m for m in prompt_msgs if m["role"] != "assistant"]
    resp = client.chat.completions.create(
        model=model, messages=prompt_msgs, max_tokens=max_tokens,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def cmd_regen(args: argparse.Namespace) -> None:
    in_path, out_path = Path(args.input), Path(args.output)
    start_at = 0
    if args.resume and out_path.exists():
        with out_path.open() as f:
            start_at = sum(1 for _ in f)
        print(f"resume: skipping first {start_at} completed rows")

    rows = []
    with in_path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    todo = list(enumerate(rows))[start_at:][: args.limit or None]

    client = OpenAI(base_url=args.endpoint, api_key="EMPTY", timeout=600.0,
                    max_retries=3)
    model = args.model or client.models.list().data[0].id
    print(f"model={model} rows_to_regenerate={len(todo)} "
          f"concurrency={args.concurrency}")

    def work(item: tuple[int, dict]) -> tuple[int, dict]:
        idx, row = item
        text = _regen_one(client, model, row, args.max_tokens, args.temperature)
        row["conversations"][-1]["value"] = text or row["conversations"][-1]["value"]
        return idx, row

    results, failures = {}, 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for idx, row in pool.map(work, todo):
            if not (row["conversations"][-1]["value"] or "").strip():
                failures += 1
            results[idx] = row

    with out_path.open("a" if start_at else "w") as out:
        for idx in sorted(results):
            out.write(json.dumps(results[idx], ensure_ascii=False))
            out.write("\n")
    print(f"regenerated: {len(results)}  empty_responses: {failures} "
          f"-> {out_path}")
    if failures:
        _fail(f"{failures} rows came back empty; re-run with --resume")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_conv = sub.add_parser("convert", help="raw ASR JSONL -> conversations")
    p_conv.add_argument("--input", required=True)
    p_conv.add_argument("--output", required=True)
    p_conv.add_argument("--resume", action="store_true")
    p_conv.add_argument("--allow-missing-audio", action="store_true")
    p_conv.set_defaults(func=cmd_convert)

    p_regen = sub.add_parser(
        "regen", help="on-policy responses, conversations preserved")
    p_regen.add_argument("--input", required=True)
    p_regen.add_argument("--output", required=True)
    p_regen.add_argument("--endpoint", default="http://localhost:8000/v1")
    p_regen.add_argument("--model", default=None)
    p_regen.add_argument("--limit", type=int, default=None)
    p_regen.add_argument("--concurrency", type=int, default=64)
    p_regen.add_argument("--max-tokens", type=int, default=2048)
    p_regen.add_argument("--temperature", type=float, default=0.0)
    p_regen.add_argument("--resume", action="store_true")
    p_regen.set_defaults(func=cmd_regen)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
