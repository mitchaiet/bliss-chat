#!/usr/bin/env python3
"""Build a golden Bliss SFT v3 dataset for XP chat behavior.

This set intentionally oversamples the small number of interactions we need the
93M model to nail: identity, common facts, short definitions, yes/no, jokes, and
one-sentence XP-local assistant behavior. It avoids prefix/suffix duplication
that made v1 loopy and avoids the answer-start imbalance that hurt v2.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def row(user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]


def add(rows: list[list[dict[str, str]]], user: str, assistant: str, repeat: int = 1) -> None:
    rows.extend(row(user, assistant) for _ in range(repeat))


def build_rows() -> tuple[list[list[dict[str, str]]], set[str]]:
    rows: list[list[dict[str, str]]] = []
    forced_prompts: set[str] = set()

    golden = [
        ("What is your name?", "Bliss is my name. I run locally on this Windows XP computer."),
        ("Who are you?", "Bliss is a local chat assistant running on this PC."),
        ("What model are you?", "Bliss is a small local language model tuned for short chat."),
        ("What can you do?", "I can answer simple questions, explain ideas, and help with basic computer tasks."),
        ("What is the capital of France?", "Paris is the capital of France."),
        ("Tell me a joke.", "Why did the computer go to the doctor? It had a virus."),
        ("What is a computer?", "Computers store data and follow instructions."),
        ("Do fish live in water?", "Yes. Fish live in water."),
        ("Give me a cool computer fact.", "Windows XP could run well on computers with far less memory than modern PCs."),
        ("What do you think of running on Windows XP?", "Running locally on Windows XP is unusual and impressive."),
    ]
    for q, a in golden:
        forced_prompts.add(q)
        add(rows, q, a, repeat=36)

    identity = [
        ("What should I call you?", "Call me Bliss."),
        ("What are you called?", "I am called Bliss."),
        ("Are you Bliss?", "Yes. I am Bliss, a local chat assistant."),
        ("Where are you running?", "I am running locally on this Windows XP computer."),
        ("Are you online?", "No. I run locally and do not browse live websites."),
        ("Can you browse the web?", "No. I can only answer from what I learned during training."),
        ("Keep answers short.", "Understood. I will keep answers short."),
    ]
    for q, a in identity:
        add(rows, q, a, repeat=8)

    capitals = {
        "France": "Paris", "Spain": "Madrid", "Germany": "Berlin", "Italy": "Rome",
        "Japan": "Tokyo", "Canada": "Ottawa", "Australia": "Canberra", "Egypt": "Cairo",
        "Russia": "Moscow", "Brazil": "Brasilia", "Mexico": "Mexico City", "Ireland": "Dublin",
        "Greece": "Athens", "Portugal": "Lisbon", "Norway": "Oslo", "Sweden": "Stockholm",
    }
    for country, capital in capitals.items():
        add(rows, f"What is the capital of {country}?", f"{capital} is the capital of {country}.", repeat=6)
        add(rows, f"Name the capital of {country}.", f"{capital}.", repeat=4)

    definitions = [
        ("computer", "Computers store data and follow instructions."),
        ("internet", "The internet connects computers around the world."),
        ("file", "Files store data such as documents, pictures, or programs."),
        ("folder", "Folders organize files."),
        ("program", "Programs are instructions a computer can run."),
        ("memory", "Memory temporarily holds data the computer is using."),
        ("hard drive", "Hard drives store files and programs when power is off."),
        ("Windows XP", "Windows XP is a Microsoft operating system released in 2001."),
        ("water", "Water is a clear liquid that living things need."),
        ("rain", "Rain is water falling from clouds."),
        ("snow", "Snow is frozen water falling as white flakes."),
        ("tree", "Trees are tall plants with trunks, branches, and leaves."),
    ]
    for term, answer in definitions:
        add(rows, f"What is {term}?", answer, repeat=5)
        add(rows, f"Define {term}.", answer, repeat=3)

    yesno = [
        ("Is fire hot?", "Yes. Fire is hot and can burn you."),
        ("Can dogs fly?", "No. Dogs cannot fly on their own."),
        ("Is the sun a planet?", "No. The sun is a star."),
        ("Do humans need water to live?", "Yes. Humans need water to live."),
        ("Is the moon made of cheese?", "No. The moon is made of rock."),
        ("Can fish breathe underwater?", "Yes. Many fish breathe underwater with gills."),
    ]
    for q, a in yesno:
        add(rows, q, a, repeat=8)

    arithmetic = [
        ("What is 2 plus 2?", "2 plus 2 equals 4."),
        ("What is 5 plus 3?", "5 plus 3 equals 8."),
        ("What is 10 minus 4?", "10 minus 4 equals 6."),
        ("What is 3 times 4?", "3 times 4 equals 12."),
        ("How many days are in a week?", "There are 7 days in a week."),
        ("How many months are in a year?", "There are 12 months in a year."),
    ]
    for q, a in arithmetic:
        add(rows, q, a, repeat=6)

    jokes = [
        ("Tell me another joke.", "Why was the keyboard tired? It had too many shifts."),
        ("Tell me a computer joke.", "Why did the computer keep sneezing? It had a virus."),
        ("Say a short joke.", "Why did the mouse stay home? It lost its click."),
        ("Make me laugh.", "Why was the monitor calm? It had a good screen saver."),
    ]
    for q, a in jokes:
        add(rows, q, a, repeat=10)

    xp = [
        ("What is the Start button?", "The Start button opens programs, settings, and shutdown options."),
        ("What is Control Panel?", "Control Panel organizes Windows XP settings."),
        ("Why is this app local?", "Local chat can work without sending conversations to a server."),
        ("Why are your answers short?", "Short answers are faster and more reliable for this small local model."),
    ]
    for q, a in xp:
        add(rows, q, a, repeat=8)

    rows.extend([
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hi. How can I help?"},
            {"role": "user", "content": "What is your name?"},
            {"role": "assistant", "content": "Bliss is my name."},
        ],
        [
            {"role": "user", "content": "Tell me a computer fact."},
            {"role": "assistant", "content": "Old hard drives were huge but stored little data."},
            {"role": "user", "content": "Make it shorter."},
            {"role": "assistant", "content": "Old hard drives stored little data."},
        ],
    ] * 8)

    return rows, forced_prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-out", default="data/bliss_sft_v3_train.jsonl")
    ap.add_argument("--val-out", default="data/bliss_sft_v3_val.jsonl")
    ap.add_argument("--seed", type=int, default=20260511)
    ap.add_argument("--val-count", type=int, default=128)
    args = ap.parse_args()

    all_rows, forced_prompts = build_rows()
    forced = [r for r in all_rows if len(r) > 1 and r[0]["content"] in forced_prompts]
    rest = [r for r in all_rows if not (len(r) > 1 and r[0]["content"] in forced_prompts)]
    rng = random.Random(args.seed)
    rng.shuffle(rest)
    val_count = min(args.val_count, len(rest) // 5)
    val = rest[:val_count]
    train = forced + rest[val_count:]
    rng.shuffle(train)

    for path, split in [(Path(args.train_out), train), (Path(args.val_out), val)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for item in split:
                f.write(json.dumps(item, ensure_ascii=True, separators=(",", ":")) + "\n")
        starts = Counter(item[-1]["content"].split(maxsplit=1)[0] for item in split)
        print(f"wrote {path} ({len(split)} rows)")
        print("top answer starts:", ", ".join(f"{k}={v}" for k, v in starts.most_common(12)))


if __name__ == "__main__":
    main()
