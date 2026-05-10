#!/usr/bin/env python3
"""
Export a nanochat tiktoken tokenizer (~/.cache/nanochat/tokenizer/tokenizer.pkl)
into a custom binary format (.nct) consumable by nc_tokenizer.c.

NCT1 format (little-endian):
  Header:
    char[8] magic                "NCT1\0\0\0\0"
    u32     version              1
    u32     vocab_size           total token count incl. specials
    u32     n_regular            count of regular (mergeable_ranks) tokens
    u32     n_specials           count of special tokens
    (rest zero to 64 bytes)

  Regular tokens (in id order, 0..n_regular-1), packed:
    u32  id
    u32  nbytes
    u8[] bytes

  Special tokens (each):
    u32  id
    u32  nbytes
    u8[] name  (UTF-8)

  EOF marker:
    u32  0xFFFFFFFF

Notes:
  - id values are the token IDs assigned by tiktoken.
  - For each unique byte 0..255 there is a single-byte token; the C side uses
    those for initial encoding of arbitrary bytes.
"""
import argparse
import os
import pickle
import struct
import sys
from pathlib import Path

NCT_MAGIC = b"NCT1\x00\x00\x00\x00"
HEADER_SIZE = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to tokenizer.pkl")
    ap.add_argument("--out", required=True, help="output .nct path")
    args = ap.parse_args()

    src = Path(os.path.expanduser(args.src))
    if not src.exists():
        sys.exit(f"missing {src}")
    with src.open("rb") as f:
        enc = pickle.load(f)

    # tiktoken.Encoding internal attrs
    mergeable = enc._mergeable_ranks  # dict[bytes, int]
    specials = enc._special_tokens    # dict[str, int]

    n_regular = len(mergeable)
    n_specials = len(specials)
    vocab_size = enc.n_vocab
    print(f"[nct] regular={n_regular} specials={n_specials} vocab={vocab_size}", file=sys.stderr)

    out = Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        # Header
        head = NCT_MAGIC + struct.pack("<IIII", 1, vocab_size, n_regular, n_specials)
        f.write(head)
        f.write(b"\x00" * (HEADER_SIZE - len(head)))

        # Regular tokens, sorted by id
        items = sorted(mergeable.items(), key=lambda kv: kv[1])
        for tok_bytes, tok_id in items:
            f.write(struct.pack("<II", tok_id, len(tok_bytes)))
            f.write(tok_bytes)

        # Special tokens
        for name, tok_id in sorted(specials.items(), key=lambda kv: kv[1]):
            name_b = name.encode("utf-8")
            f.write(struct.pack("<II", tok_id, len(name_b)))
            f.write(name_b)

        # EOF marker
        f.write(struct.pack("<I", 0xFFFFFFFF))

    sz = out.stat().st_size
    print(f"[nct] wrote {out} ({sz} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
