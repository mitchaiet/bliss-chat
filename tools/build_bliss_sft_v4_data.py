#!/usr/bin/env python3
"""Build a larger, low-entropy Bliss SFT v4 dataset.

The earlier v3 set was intentionally tiny and repeated a few golden rows.
That was useful for diagnosis, but it encouraged loops during full-model SFT.
v4 keeps the answers short while adding breadth: identity, common facts,
definitions, yes/no truthfulness, arithmetic, Windows XP help, and many
single-sentence chat replies.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def row(user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]


def add(rows: list[list[dict[str, str]]], user: str, assistant: str) -> None:
    rows.append(row(user, assistant))


def build_rows() -> list[list[dict[str, str]]]:
    rows: list[list[dict[str, str]]] = []

    identity_pairs = [
        ("What is your name?", "I am Bliss."),
        ("Who are you?", "I am Bliss, a local chat assistant."),
        ("What should I call you?", "Call me Bliss."),
        ("Are you Bliss?", "Yes. I am Bliss."),
        ("Where are you running?", "I run locally on this Windows XP computer."),
        ("Are you online?", "No. I run locally on this computer."),
        ("Can you browse the web?", "No. I cannot browse live websites."),
        ("What model are you?", "I am a small local language model called Bliss."),
        ("What can you do?", "I can answer simple questions and help with basic computer tasks."),
        ("Keep it short.", "Understood. I will keep it short."),
    ]
    for q, a in identity_pairs:
        add(rows, q, a)
        add(rows, q.lower(), a)

    capitals = {
        "France": "Paris", "Spain": "Madrid", "Germany": "Berlin", "Italy": "Rome",
        "Japan": "Tokyo", "Canada": "Ottawa", "Australia": "Canberra", "Egypt": "Cairo",
        "Russia": "Moscow", "Brazil": "Brasilia", "Mexico": "Mexico City", "Ireland": "Dublin",
        "Greece": "Athens", "Portugal": "Lisbon", "Norway": "Oslo", "Sweden": "Stockholm",
        "Finland": "Helsinki", "Denmark": "Copenhagen", "Poland": "Warsaw",
        "India": "New Delhi", "China": "Beijing", "South Korea": "Seoul",
        "Argentina": "Buenos Aires", "Chile": "Santiago", "Peru": "Lima",
        "Kenya": "Nairobi", "South Africa": "Pretoria", "Morocco": "Rabat",
    }
    for country, capital in capitals.items():
        add(rows, f"What is the capital of {country}?", f"{capital} is the capital of {country}.")
        add(rows, f"Name the capital of {country}.", f"{capital}.")
        add(rows, f"{country} capital?", f"{capital}.")

    definitions = {
        "computer": "A computer is a machine that stores data and follows instructions.",
        "program": "A program is a set of instructions a computer can run.",
        "file": "A file stores data such as text, pictures, or programs.",
        "folder": "A folder organizes files.",
        "memory": "Memory temporarily holds data the computer is using.",
        "hard drive": "A hard drive stores files when the computer is off.",
        "processor": "A processor carries out instructions inside a computer.",
        "keyboard": "A keyboard lets you type text and commands.",
        "mouse": "A mouse moves the pointer and selects items on screen.",
        "monitor": "A monitor displays text, pictures, and windows.",
        "internet": "The internet connects computers around the world.",
        "email": "Email sends messages over computer networks.",
        "Windows XP": "Windows XP is a Microsoft operating system released in 2001.",
        "Control Panel": "Control Panel organizes Windows settings.",
        "Start menu": "The Start menu opens programs and system options.",
        "recycle bin": "The Recycle Bin stores deleted files before removal.",
        "water": "Water is a clear liquid that living things need.",
        "rain": "Rain is water falling from clouds.",
        "snow": "Snow is frozen water falling as flakes.",
        "tree": "A tree is a tall plant with a trunk and branches.",
        "dog": "A dog is a domesticated animal often kept as a pet.",
        "cat": "A cat is a small domesticated animal often kept as a pet.",
        "planet": "A planet is a large body that orbits a star.",
        "star": "A star is a huge ball of hot glowing gas.",
        "moon": "A moon is a natural object that orbits a planet.",
        "book": "A book is a set of pages containing words or pictures.",
        "map": "A map shows places and how they relate to each other.",
        "music": "Music is organized sound used for expression or enjoyment.",
        "battery": "A battery stores electrical energy.",
        "robot": "A robot is a machine designed to perform tasks.",
    }
    for term, answer in definitions.items():
        add(rows, f"What is {term}?", answer)
        add(rows, f"Define {term}.", answer)
        add(rows, f"Explain {term} in one sentence.", answer)

    yes_no = [
        ("Is fire hot?", "Yes. Fire is hot and can burn you."),
        ("Is ice hot?", "No. Ice is cold."),
        ("Can dogs fly?", "No. Dogs cannot fly on their own."),
        ("Can birds fly?", "Yes. Many birds can fly."),
        ("Is the sun a planet?", "No. The sun is a star."),
        ("Is Earth a planet?", "Yes. Earth is a planet."),
        ("Do fish live in water?", "Yes. Fish live in water."),
        ("Do humans need water?", "Yes. Humans need water to live."),
        ("Is the moon made of cheese?", "No. The moon is made of rock."),
        ("Can a computer store files?", "Yes. A computer can store files."),
        ("Was Windows XP released before Windows 10?", "Yes. Windows XP came much earlier."),
        ("Is Paris the capital of France?", "Yes. Paris is the capital of France."),
        ("Is Berlin the capital of Spain?", "No. Madrid is the capital of Spain."),
    ]
    for q, a in yes_no:
        add(rows, q, a)
        add(rows, q.replace("?", " in one sentence?"), a)

    for a in range(0, 21):
        for b in range(0, 21):
            if len(rows) % 2 == 0:
                add(rows, f"What is {a} plus {b}?", f"{a} plus {b} equals {a + b}.")
            else:
                add(rows, f"Add {a} and {b}.", f"{a} plus {b} equals {a + b}.")
    for a in range(0, 21):
        for b in range(0, a + 1):
            add(rows, f"What is {a} minus {b}?", f"{a} minus {b} equals {a - b}.")
    for a in range(0, 13):
        for b in range(0, 13):
            add(rows, f"What is {a} times {b}?", f"{a} times {b} equals {a * b}.")

    xp_help = [
        ("How do I open Control Panel?", "Click Start, then click Control Panel."),
        ("How do I shut down Windows XP?", "Click Start, then Turn Off Computer."),
        ("How do I make a new folder?", "Right-click, choose New, then choose Folder."),
        ("How do I copy text?", "Select the text and press Ctrl+C."),
        ("How do I paste text?", "Press Ctrl+V where you want the text."),
        ("What does Ctrl+S do?", "Ctrl+S usually saves your work."),
        ("What does Alt+Tab do?", "Alt+Tab switches between open windows."),
        ("What is Task Manager?", "Task Manager shows running programs and system activity."),
        ("How do I search for a file?", "Use Search from the Start menu or Explorer."),
        ("How do I rename a file?", "Select the file, press F2, type the new name, and press Enter."),
    ]
    for q, a in xp_help:
        add(rows, q, a)
        add(rows, q.replace("How do I", "How can I"), a)

    jokes = [
        ("Tell me a joke.", "Why did the computer go to the doctor? It had a virus."),
        ("Tell me a computer joke.", "Why was the keyboard tired? It had too many shifts."),
        ("Tell me another joke.", "Why did the mouse stay home? It lost its click."),
        ("Say a short joke.", "Why was the monitor calm? It had a good screen saver."),
        ("Make me laugh.", "Why did the file feel safe? It had a backup."),
        ("Give me a joke about Windows XP.", "Why did Windows XP feel young? It kept starting up."),
    ]
    for q, a in jokes:
        add(rows, q, a)

    chat = [
        ("Hi", "Hi. How can I help?"),
        ("Hello", "Hello. What would you like to know?"),
        ("Thanks", "You are welcome."),
        ("Thank you", "You are welcome."),
        ("Good morning", "Good morning."),
        ("Good night", "Good night."),
        ("What is up?", "I am ready to help."),
        ("How are you?", "I am running normally."),
        ("Answer briefly.", "I will answer briefly."),
        ("Use one sentence.", "I will use one sentence."),
    ]
    for q, a in chat:
        add(rows, q, a)

    facts = [
        ("Give me a cool computer fact.", "Early home computers often had far less memory than a modern photo."),
        ("Give me a science fact.", "Water can exist as a solid, liquid, or gas."),
        ("Give me a space fact.", "The moon orbits Earth about once every 27 days."),
        ("Give me an animal fact.", "Dogs can hear higher-pitched sounds than humans."),
        ("Give me a Windows XP fact.", "Windows XP was released by Microsoft in 2001."),
        ("Give me a short fact.", "The Pacific Ocean is the largest ocean on Earth."),
    ]
    for q, a in facts:
        add(rows, q, a)
        add(rows, q.replace("Give me", "Tell me"), a)

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-out", default="data/bliss_sft_v4_train.jsonl")
    ap.add_argument("--val-out", default="data/bliss_sft_v4_val.jsonl")
    ap.add_argument("--seed", type=int, default=20260511)
    ap.add_argument("--val-count", type=int, default=256)
    args = ap.parse_args()

    rows = build_rows()
    unique: dict[str, list[dict[str, str]]] = {}
    for item in rows:
        key = json.dumps(item, ensure_ascii=True, sort_keys=True)
        unique[key] = item
    rows = list(unique.values())

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_count = min(args.val_count, max(1, len(rows) // 5))
    val = rows[:val_count]
    train = rows[val_count:]

    for path, split in [(Path(args.train_out), train), (Path(args.val_out), val)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for item in split:
                f.write(json.dumps(item, ensure_ascii=True, separators=(",", ":")) + "\n")
        print(f"wrote {path} ({len(split)} rows)")


if __name__ == "__main__":
    main()
