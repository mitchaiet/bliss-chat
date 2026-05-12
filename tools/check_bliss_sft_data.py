#!/usr/bin/env python3
"""Validate Bliss SFT JSONL with nanochat's tokenizer/rendering path."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--nanochat", default=os.environ.get("NANOCHAT_DIR", "~/nanochat"))
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--plain", action="store_true", help="validate plain text rendering")
    ap.add_argument("--plain-style", default="user_assistant", choices=["user_assistant", "qa", "qa_nospace"], help="plain rendering style")
    ap.add_argument("--system-prefix", default="", help="optional plain-text prefix before each conversation")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.nanochat).expanduser()))
    from nanochat.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    bos = tokenizer.get_bos_token_id()

    def render_plain(messages: list[dict[str, str]]) -> tuple[list[int], list[int]]:
        ids = [bos]
        mask = [0]

        def add(text: str, bit: int) -> None:
            toks = tokenizer.encode(text)
            ids.extend(toks)
            mask.extend([bit] * len(toks))

        if args.system_prefix:
            prefix = args.system_prefix if args.system_prefix.endswith("\n") else args.system_prefix + "\n"
            add(prefix, 0)

        for i, message in enumerate(messages):
            if message["role"] == "user":
                if args.plain_style == "qa":
                    add(f"Q: {message['content'].strip()}\nA: ", 0)
                elif args.plain_style == "qa_nospace":
                    add(f"Q: {message['content'].strip()}\nA:", 0)
                else:
                    add(f"User: {message['content'].strip()}\nAssistant: ", 0)
            else:
                add(message["content"].strip(), 1)
                add("\n", 1)
        return ids[:args.max_tokens], mask[:args.max_tokens]
    lengths = []
    masked = []
    too_long = 0
    missing_assistant_end = 0
    path = Path(args.jsonl)
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if args.limit and i > args.limit:
                break
            messages = json.loads(line)
            if args.plain:
                ids, mask = render_plain(messages)
            else:
                ids, mask = tokenizer.render_conversation({"messages": messages}, max_tokens=args.max_tokens)
            m = sum(mask)
            lengths.append(len(ids))
            masked.append(m)
            if len(ids) >= args.max_tokens:
                too_long += 1
            if m == 0:
                raise SystemExit(f"{path}:{i}: all labels ignored")
            if not args.plain and assistant_end not in [tok for tok, bit in zip(ids, mask) if bit]:
                missing_assistant_end += 1
            if messages[-1]["role"] != "assistant":
                raise SystemExit(f"{path}:{i}: last role is not assistant")
    if not lengths:
        raise SystemExit(f"{path}: no examples")
    print(f"[check] rows={len(lengths)} max_len={max(lengths)} avg_len={sum(lengths)/len(lengths):.1f}")
    print(f"[check] avg_supervised={sum(masked)/len(masked):.1f} too_long={too_long} missing_assistant_end={missing_assistant_end}")
    if too_long:
        raise SystemExit("[check] some examples reached max token cap")
    if not args.plain and missing_assistant_end:
        raise SystemExit("[check] some examples lost assistant_end supervision")


if __name__ == "__main__":
    main()
