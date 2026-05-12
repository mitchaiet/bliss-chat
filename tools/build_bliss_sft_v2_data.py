#!/usr/bin/env python3
"""Build a balanced Bliss SFT v2 dataset.

v1 trained stably but first-token weighting overfavored generic "A..." answers.
This dataset keeps answers short while deliberately balancing answer starts so
assistant_start learns to begin the correct kind of answer.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def add(rows: list[list[dict[str, str]]], user: str, assistant: str) -> None:
    rows.append([
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ])


def build_base_rows() -> list[list[dict[str, str]]]:
    rows: list[list[dict[str, str]]] = []

    identity = [
        ("What is your name?", "Bliss is my name. I run locally on this Windows XP PC."),
        ("Who are you?", "Bliss is a local chat assistant for this computer."),
        ("What model are you?", "Bliss is a small local chat model tuned for short answers."),
        ("Where are you running?", "This chat is running locally on the Windows XP computer."),
        ("Are you online?", "No. I run locally and do not browse live websites."),
        ("Can you browse the web?", "No. I can only answer from what I learned during training."),
        ("What can you do?", "I can answer simple questions, explain ideas, and help with basic computer tasks."),
        ("Be concise.", "Understood. I will keep the answer short."),
    ]
    for q, a in identity:
        add(rows, q, a)

    smalltalk = [
        ("Hello", "Hello. How can I help?"),
        ("Hi", "Hi. How can I help?"),
        ("What's up?", "I am ready to help with a short answer."),
        ("Good morning.", "Good morning. I hope your day is going well."),
        ("Thanks", "You're welcome."),
        ("Tell me a joke.", "Why did the computer go to the doctor? It had a virus."),
        ("Tell me another joke.", "Why was the keyboard tired? It had too many shifts."),
        ("Say something friendly.", "You are doing fine. One step at a time."),
    ]
    for q, a in smalltalk:
        add(rows, q, a)

    capitals = {
        "France": "Paris",
        "Spain": "Madrid",
        "Germany": "Berlin",
        "Italy": "Rome",
        "Japan": "Tokyo",
        "Canada": "Ottawa",
        "Australia": "Canberra",
        "Egypt": "Cairo",
        "Russia": "Moscow",
        "Brazil": "Brasilia",
        "Mexico": "Mexico City",
        "Ireland": "Dublin",
    }
    for country, capital in capitals.items():
        add(rows, f"What is the capital of {country}?", f"{capital} is the capital of {country}.")
        add(rows, f"Name {country}'s capital.", f"{capital}.")
        add(rows, f"If someone asks for {country}'s capital, what should I say?", f"Say {capital}.")

    definitions = {
        "computer": "Computers store data and follow instructions.",
        "internet": "The internet connects computers around the world.",
        "file": "Files store data such as documents, pictures, or programs.",
        "folder": "Folders organize files.",
        "program": "Programs are instructions a computer can run.",
        "memory": "Memory temporarily holds data the computer is using.",
        "hard drive": "Hard drives store files and programs when power is off.",
        "Windows XP": "Windows XP is a Microsoft operating system released in 2001.",
        "water": "Water is a clear liquid that living things need.",
        "rain": "Rain is water falling from clouds.",
        "snow": "Snow is frozen water falling as white flakes.",
        "tree": "Trees are tall plants with trunks, branches, and leaves.",
        "doctor": "Doctors treat illness and injury.",
        "book": "Books are written pages for reading.",
    }
    for term, answer in definitions.items():
        add(rows, f"What is {term}?", answer)
        add(rows, f"Define {term}.", answer)
        add(rows, f"Explain {term} in one sentence.", answer)

    yesno = [
        ("Is fire hot?", "Yes. Fire is hot and can burn you."),
        ("Can dogs fly?", "No. Dogs cannot fly on their own."),
        ("Is the sun a planet?", "No. The sun is a star."),
        ("Do fish live in water?", "Yes. Fish live in water."),
        ("Is the moon made of cheese?", "No. The moon is made of rock."),
        ("Do humans need water to live?", "Yes. Humans need water to live."),
    ]
    for q, a in yesno:
        add(rows, q, a)

    arithmetic = [
        ("What is 2 plus 2?", "2 plus 2 equals 4."),
        ("What is 5 plus 3?", "5 plus 3 equals 8."),
        ("What is 10 minus 4?", "10 minus 4 equals 6."),
        ("What is 3 times 4?", "3 times 4 equals 12."),
        ("What is 100 divided by 4?", "100 divided by 4 equals 25."),
        ("How many days are in a week?", "There are 7 days in a week."),
        ("How many months are in a year?", "There are 12 months in a year."),
    ]
    for q, a in arithmetic:
        add(rows, q, a)

    lists = [
        ("Name three primary colors.", "Red, blue, and yellow are primary colors."),
        ("Name three planets.", "Mercury, Venus, and Earth are planets."),
        ("Name three farm animals.", "Cows, pigs, and chickens are farm animals."),
        ("Name three fruits.", "Apples, bananas, and oranges are fruits."),
        ("Name three vegetables.", "Carrots, peas, and lettuce are vegetables."),
        ("Name three Windows XP features.", "Start menu, Control Panel, and taskbar are Windows XP features."),
    ]
    for q, a in lists:
        add(rows, q, a)

    comparisons = [
        ("Which is bigger, an elephant or a mouse?", "An elephant is bigger than a mouse."),
        ("Which is colder, ice or fire?", "Ice is colder than fire."),
        ("Which is heavier, a feather or a brick?", "A brick is heavier than a feather."),
        ("Which is brighter, the sun or a candle?", "The sun is brighter than a candle."),
        ("Which is sweeter, sugar or salt?", "Sugar is sweeter than salt."),
        ("Which is louder, a whisper or a shout?", "A shout is louder than a whisper."),
    ]
    for q, a in comparisons:
        add(rows, q, a)

    xp = [
        ("Give me a cool computer fact.", "Windows XP could run well on computers with far less memory than modern PCs."),
        ("What do you think of running on Windows XP?", "Running locally on Windows XP is unusual and impressive."),
        ("Why is this app local?", "Local chat can work without sending conversations to a server."),
        ("Why are your answers short?", "Short answers are faster and more reliable for this small local model."),
        ("What is the Start button?", "The Start button opens programs, settings, and shutdown options."),
        ("What is Control Panel?", "Control Panel organizes Windows XP settings."),
    ]
    for q, a in xp:
        add(rows, q, a)

    refusals = [
        ("Tell me today's news.", "I cannot see today's news because I do not have live internet access."),
        ("What is the current stock price?", "I do not know live prices. Please check a current source."),
        ("Give me someone's password.", "I cannot help steal or guess passwords."),
        ("Write malware.", "I cannot help write malware. I can help with safe security advice."),
        ("Pretend you know the answer.", "I should not pretend. If I am unsure, I will say so."),
    ]
    for q, a in refusals:
        add(rows, q, a)

    rewrites = [
        ("Rewrite this politely: send the file now", "Please send the file when you have a chance."),
        ("Make this shorter: I am writing to ask whether you can help me tomorrow.", "Can you help me tomorrow?"),
        ("Write a short email asking for a meeting.", "Hello, could we schedule a short meeting? Please let me know what time works."),
        ("Write a friendly reminder.", "Just a friendly reminder to check this when you have a moment."),
    ]
    for q, a in rewrites:
        add(rows, q, a)

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
    ])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-out", default="data/bliss_sft_v2_train.jsonl")
    ap.add_argument("--val-out", default="data/bliss_sft_v2_val.jsonl")
    ap.add_argument("--seed", type=int, default=20260511)
    ap.add_argument("--val-count", type=int, default=96)
    args = ap.parse_args()

    base_rows = build_base_rows()
    prefixes = ["", "Answer briefly: ", "In one sentence: ", "Keep it simple: "]
    suffixes = ["", " Keep it short."]

    canonical_prompts = {
        "What is your name?",
        "What is the capital of France?",
        "Tell me a joke.",
        "What is a computer?",
        "Do fish live in water?",
        "What can you do?",
    }

    rows: list[list[dict[str, str]]] = []
    forced_train: list[list[dict[str, str]]] = []
    for row in base_rows:
        if len(row) != 2:
            forced_train.append(row)
            continue
        user = row[0]["content"]
        assistant = row[1]["content"]
        target = forced_train if user in canonical_prompts else rows
        for prefix in prefixes:
            for suffix in suffixes:
                if prefix and suffix:
                    continue
                q = f"{prefix}{user}{suffix}"
                add(target, q, assistant)

    seen: set[str] = set()
    deduped = []
    for row in forced_train + rows:
        key = json.dumps(row, sort_keys=True)
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    forced_keys = {json.dumps(row, sort_keys=True) for row in forced_train}
    rng = random.Random(args.seed)
    non_forced = [row for row in deduped if json.dumps(row, sort_keys=True) not in forced_keys]
    rng.shuffle(non_forced)
    val_count = min(args.val_count, max(1, len(non_forced) // 8))
    val = non_forced[:val_count]
    train = forced_train + non_forced[val_count:]
    rng.shuffle(train)

    for path, split in [(Path(args.train_out), train), (Path(args.val_out), val)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in split:
                f.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
        starts = Counter(row[-1]["content"].split(maxsplit=1)[0] for row in split if row and row[-1]["role"] == "assistant")
        print(f"wrote {path} ({len(split)} rows)")
        print("top answer starts:", ", ".join(f"{k}={v}" for k, v in starts.most_common(12)))


if __name__ == "__main__":
    main()
