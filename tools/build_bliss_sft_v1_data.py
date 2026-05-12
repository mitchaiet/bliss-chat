#!/usr/bin/env python3
"""Build a tiny-model-friendly Bliss SFT v1 JSONL dataset.

Each line is a nanochat CustomJSON conversation: a JSON array of alternating
user/assistant messages. Answers are intentionally short for a 110M model on XP.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def add(rows: list[list[dict[str, str]]], user: str, assistant: str) -> None:
    rows.append([
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ])


def build_rows() -> list[list[dict[str, str]]]:
    rows: list[list[dict[str, str]]] = []

    identity = [
        ("What is your name?", "I'm Bliss, a local chat assistant running on this Windows XP computer."),
        ("Who are you?", "I'm Bliss. I answer short questions locally on this PC."),
        ("Are you online?", "No. I run locally here, so I may not know recent facts."),
        ("What can you do?", "I can answer simple questions, explain ideas, draft short text, and help with basic computer tasks."),
        ("What model are you?", "I'm a small Bliss language model tuned for short local chat."),
        ("Where are you running?", "I'm running locally on this Windows XP machine."),
        ("Can you browse the web?", "No. I can only answer from what I learned during training."),
        ("Be concise.", "Understood. I will keep the answer short."),
    ]
    for q, a in identity:
        add(rows, q, a)

    smalltalk = [
        ("Hello", "Hello. What would you like to talk about?"),
        ("Hi", "Hi. How can I help?"),
        ("What's up?", "I'm ready to help with a short answer."),
        ("Good morning.", "Good morning. I hope your day is going well."),
        ("Thanks", "You're welcome."),
        ("Tell me a joke.", "Why did the computer go to the doctor? It had a virus."),
        ("Tell me another joke.", "Why was the keyboard tired? It had too many shifts."),
        ("Say something friendly.", "You are doing fine. One step at a time."),
    ]
    for q, a in smalltalk:
        add(rows, q, a)

    facts = {
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
    for country, capital in facts.items():
        add(rows, f"What is the capital of {country}?", f"The capital of {country} is {capital}.")
        add(rows, f"Name {country}'s capital.", f"{country}'s capital is {capital}.")

    definitions = {
        "computer": "A computer is a machine that stores data and follows instructions.",
        "internet": "The internet is a global network that connects computers.",
        "file": "A file is stored data, such as a document, picture, or program.",
        "folder": "A folder is a place to organize files.",
        "program": "A program is a set of instructions a computer can run.",
        "memory": "Memory temporarily holds data that a computer is using.",
        "hard drive": "A hard drive stores files and programs even when power is off.",
        "Windows XP": "Windows XP is a Microsoft operating system released in 2001.",
        "water": "Water is a clear liquid that people, animals, and plants need to live.",
        "rain": "Rain is water that falls from clouds.",
        "snow": "Snow is frozen water that falls as soft white flakes.",
        "tree": "A tree is a tall plant with a trunk, branches, and leaves.",
        "doctor": "A doctor is a person trained to treat illness and injury.",
        "book": "A book is a set of written pages for reading.",
    }
    for term, answer in definitions.items():
        add(rows, f"What is {term}?", answer)
        add(rows, f"Define {term}.", answer)

    lists = [
        ("Name three primary colors.", "Red, blue, and yellow are primary colors."),
        ("Name three planets.", "Mercury, Venus, and Earth are planets."),
        ("Name three farm animals.", "Cows, pigs, and chickens are farm animals."),
        ("Name three fruits.", "Apples, bananas, and oranges are fruits."),
        ("Name three vegetables.", "Carrots, peas, and lettuce are vegetables."),
        ("Name three days of the week.", "Monday, Tuesday, and Wednesday are days of the week."),
        ("Name three months.", "January, February, and March are months."),
        ("Name three Windows XP features.", "The Start menu, Control Panel, and taskbar are Windows XP features."),
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
        ("What is 7 plus 5?", "7 plus 5 equals 12."),
        ("What is half of 10?", "Half of 10 is 5."),
        ("How many days are in a week?", "There are 7 days in a week."),
        ("How many months are in a year?", "There are 12 months in a year."),
        ("How many seconds are in a minute?", "There are 60 seconds in a minute."),
    ]
    for q, a in arithmetic:
        add(rows, q, a)

    xp = [
        ("Give me a cool computer fact.", "Windows XP could run well on computers with far less memory than modern PCs."),
        ("What do you think of running on Windows XP?", "It is unusual and impressive. I am a small local model built to fit this machine."),
        ("Why is this app local?", "A local app can work without sending your chats to a server."),
        ("Why are your answers short?", "Short answers are faster and more reliable for this small local model."),
        ("What is the Start button?", "The Start button opens the main menu for programs, settings, and shutdown."),
        ("What is Control Panel?", "Control Panel is where Windows XP settings are organized."),
    ]
    for q, a in xp:
        add(rows, q, a)

    refusals = [
        ("Tell me today's news.", "I cannot see today's news because I do not have live internet access."),
        ("What is the current stock price?", "I do not know live prices. Please check a current source."),
        ("Give me someone's password.", "I cannot help with stealing or guessing passwords."),
        ("Write malware.", "I cannot help write malware. I can help with safe computer security advice."),
        ("Pretend you know the answer.", "I should not pretend. If I am unsure, I will say so."),
    ]
    for q, a in refusals:
        add(rows, q, a)

    rewrites = [
        ("Rewrite this politely: send the file now", "Please send the file when you have a chance."),
        ("Make this shorter: I am writing to ask whether you can help me tomorrow.", "Can you help me tomorrow?"),
        ("Write a short email asking for a meeting.", "Hello, could we schedule a short meeting? Please let me know what time works for you."),
        ("Write a friendly reminder.", "Just a friendly reminder to check this when you have a moment."),
    ]
    for q, a in rewrites:
        add(rows, q, a)

    # Expand with stable paraphrases that teach complete short answers.
    more: list[tuple[str, str]] = []
    for term, answer in definitions.items():
        more.append((f"Explain {term} in one sentence.", answer))
    for country, capital in facts.items():
        more.append((f"If someone asks for the capital of {country}, what should I say?", f"Say that the capital of {country} is {capital}."))
    for q, a in arithmetic:
        more.append((f"Answer briefly: {q}", a))
    for q, a in lists + comparisons + yesno + xp:
        more.append((f"Answer briefly: {q}", a))
    for q, a in more:
        add(rows, q, a)

    # A few two-turn conversations, still short.
    rows.extend([
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hi. How can I help?"},
            {"role": "user", "content": "What is your name?"},
            {"role": "assistant", "content": "I'm Bliss, a local chat assistant."},
        ],
        [
            {"role": "user", "content": "Tell me a computer fact."},
            {"role": "assistant", "content": "The first hard drives were much larger and held far less data than modern drives."},
            {"role": "user", "content": "Make it shorter."},
            {"role": "assistant", "content": "Old hard drives were huge but stored little data."},
        ],
    ])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-out", default="data/bliss_sft_v1_train.jsonl")
    ap.add_argument("--val-out", default="data/bliss_sft_v1_val.jsonl")
    ap.add_argument("--seed", type=int, default=20260511)
    ap.add_argument("--val-count", type=int, default=48)
    args = ap.parse_args()

    base_rows = build_rows()
    prefixes = [
        "",
        "Please answer briefly: ",
        "Give a short answer: ",
        "Quick question: ",
        "In one sentence: ",
        "Answer in one sentence: ",
        "Keep it simple: ",
        "For a Windows XP chat app, answer: ",
        "Local assistant question: ",
        "Tell me briefly: ",
    ]
    suffixes = ["", " Keep it short.", " Answer plainly.", " Use one or two sentences."]
    rows = []
    for row in base_rows:
        if len(row) != 2:
            rows.append(row)
            continue
        user = row[0]["content"]
        assistant = row[1]["content"]
        for prefix in prefixes:
            for suffix in suffixes:
                q = f"{prefix}{user}{suffix}"
                if len(q) <= 240:
                    add(rows, q, assistant)
    seen: set[str] = set()
    deduped = []
    for row in rows:
        key = json.dumps(row, sort_keys=True)
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    rng = random.Random(args.seed)
    rng.shuffle(deduped)
    val_count = min(args.val_count, max(1, len(deduped) // 10))
    val = deduped[:val_count]
    train = deduped[val_count:]

    for path, split in [(Path(args.train_out), train), (Path(args.val_out), val)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in split:
                f.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
        print(f"wrote {path} ({len(split)} rows)")


if __name__ == "__main__":
    main()
