#!/usr/bin/env python3
"""Build a broader exact-runtime Bliss SFT dataset.

This is deliberately plain and XP-friendly:
  Q: <user>
  A:<one short answer>

No chat-special tokens, no markdown, no long essays. The goal is not to make
Bliss broadly smart overnight; it is to make the deployed d12 model behave like
a compact local assistant on the questions users actually try first.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text.encode("ascii", "ignore").decode("ascii")


def row(user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": clean(user)},
        {"role": "assistant", "content": clean(assistant)},
    ]


def add(rows: list[list[dict[str, str]]], user: str, assistant: str) -> None:
    user = clean(user)
    assistant = clean(assistant)
    if user and assistant and len(assistant.split()) <= 34:
        rows.append(row(user, assistant))


def add_variants(
    rows: list[list[dict[str, str]]],
    prompts: list[str],
    answer: str,
    *,
    prefixes: list[str] | None = None,
    suffixes: list[str] | None = None,
) -> None:
    prefixes = prefixes or [""]
    suffixes = suffixes or [""]
    for prompt in prompts:
        for prefix in prefixes:
            for suffix in suffixes:
                add(rows, f"{prefix}{prompt}{suffix}", answer)


def build_rows() -> list[list[dict[str, str]]]:
    rows: list[list[dict[str, str]]] = []
    brief_prefixes = ["", "Answer briefly: ", "In one sentence, ", "Quick question: "]
    brief_suffixes = ["", " Keep it short.", " Answer plainly.", " Use one sentence."]

    identity = {
        "What is your name?": "I am Bliss.",
        "Who are you?": "I am Bliss, a small local chat assistant.",
        "What should I call you?": "Call me Bliss.",
        "Are you Bliss?": "Yes. I am Bliss.",
        "Are you ChatGPT?": "No. I am Bliss, a small local model.",
        "Are you Claude?": "No. I am Bliss, a small local model.",
        "Where are you running?": "I run locally on this Windows XP computer.",
        "Can you browse the web?": "No. I cannot browse live websites.",
        "Do you know today's news?": "No. I do not have live news access.",
        "What can you do?": "I can answer simple questions and help with basic computer tasks.",
        "Why are your answers short?": "Short answers are faster and more reliable for this small local model.",
    }
    for q, a in identity.items():
        add_variants(rows, [q, q.lower()], a, prefixes=brief_prefixes, suffixes=brief_suffixes)

    repair = {
        "What?": "Sorry, that was unclear. Please ask the question another way.",
        "Huh?": "Sorry, I did not explain that well. Please ask again.",
        "What do you mean?": "I mean that I can give a short clearer answer if you ask again.",
        "That made no sense.": "You are right. Please ask again and I will keep it simple.",
        "Try again.": "Please send the question again and I will answer more clearly.",
        "Can you clarify?": "Yes. Ask the specific part you want clarified.",
    }
    for q, a in repair.items():
        add_variants(rows, [q], a, prefixes=["", "Bliss, "], suffixes=["", " Keep it short."])

    facts = {
        "Give me a cool computer fact.": "The first computer mouse was made of wood.",
        "Tell me a cool computer fact.": "Windows XP could run on computers with far less memory than modern PCs.",
        "Give me a Windows XP fact.": "Windows XP was released by Microsoft in 2001.",
        "Tell me a cool cat fact.": "Cats can rotate their ears to help locate sounds.",
        "Give me a cool cat fact.": "Cats use their whiskers to sense nearby objects.",
        "Tell me a dog fact.": "Dogs can hear higher-pitched sounds than humans.",
        "Give me a bird fact.": "Many birds have hollow bones that help them fly.",
        "Give me a space fact.": "The Moon orbits Earth about once every 27 days.",
        "Give me a science fact.": "Water can exist as a solid, liquid, or gas.",
    }
    for q, a in facts.items():
        add_variants(rows, [q, q.replace("Give me", "Tell me")], a, prefixes=brief_prefixes, suffixes=brief_suffixes)

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
        "Recycle Bin": "The Recycle Bin stores deleted files before final removal.",
        "cat": "A cat is a small domesticated animal often kept as a pet.",
        "dog": "A dog is a domesticated animal often kept as a pet.",
        "planet": "A planet is a large body that orbits a star.",
        "star": "A star is a huge ball of hot glowing gas.",
        "moon": "A moon is a natural object that orbits a planet.",
        "water": "Water is a clear liquid that living things need.",
        "tree": "A tree is a tall plant with a trunk, branches, and leaves.",
        "book": "A book is a set of pages containing words or pictures.",
        "map": "A map shows places and how they relate to each other.",
        "music": "Music is organized sound used for expression or enjoyment.",
        "battery": "A battery stores electrical energy.",
        "robot": "A robot is a machine designed to perform tasks.",
    }
    for term, answer in definitions.items():
        prompts = [f"What is {term}?", f"Define {term}.", f"Explain {term} in one sentence."]
        add_variants(rows, prompts, answer, prefixes=brief_prefixes, suffixes=brief_suffixes)

    capitals = {
        "France": "Paris", "Spain": "Madrid", "Germany": "Berlin", "Italy": "Rome",
        "Japan": "Tokyo", "Canada": "Ottawa", "Australia": "Canberra", "Egypt": "Cairo",
        "Russia": "Moscow", "Brazil": "Brasilia", "Mexico": "Mexico City", "Ireland": "Dublin",
        "Greece": "Athens", "Portugal": "Lisbon", "Norway": "Oslo", "Sweden": "Stockholm",
        "Finland": "Helsinki", "Denmark": "Copenhagen", "Poland": "Warsaw", "India": "New Delhi",
        "China": "Beijing", "South Korea": "Seoul", "Argentina": "Buenos Aires", "Chile": "Santiago",
        "Peru": "Lima", "Kenya": "Nairobi", "South Africa": "Pretoria", "Morocco": "Rabat",
        "Netherlands": "Amsterdam", "Belgium": "Brussels", "Austria": "Vienna", "Switzerland": "Bern",
    }
    for country, capital in capitals.items():
        add_variants(
            rows,
            [f"What is the capital of {country}?", f"Name {country}'s capital.", f"{country} capital?"],
            f"{capital} is the capital of {country}.",
            prefixes=brief_prefixes,
            suffixes=brief_suffixes,
        )

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
        ("Are cats mammals?", "Yes. Cats are mammals."),
        ("Do cats use whiskers?", "Yes. Cats use whiskers to sense nearby objects."),
    ]
    for q, a in yes_no:
        add_variants(rows, [q, q.replace("?", " in one sentence?")], a, prefixes=brief_prefixes, suffixes=brief_suffixes)

    for a in range(0, 51):
        for b in range(0, 51):
            add(rows, f"What is {a} plus {b}?", f"{a} plus {b} equals {a + b}.")
            add(rows, f"Add {a} and {b}.", f"{a} plus {b} equals {a + b}.")
    for a in range(0, 51):
        for b in range(0, a + 1):
            add(rows, f"What is {a} minus {b}?", f"{a} minus {b} equals {a - b}.")
    for a in range(0, 13):
        for b in range(0, 13):
            add(rows, f"What is {a} times {b}?", f"{a} times {b} equals {a * b}.")

    xp_help = {
        "open Control Panel": "Click Start, then click Control Panel.",
        "open My Computer": "Double-click My Computer on the desktop or Start menu.",
        "shut down Windows XP": "Click Start, then Turn Off Computer.",
        "restart Windows XP": "Click Start, Turn Off Computer, then Restart.",
        "make a new folder": "Right-click, choose New, then choose Folder.",
        "rename a file": "Select the file, press F2, type the new name, and press Enter.",
        "delete a file": "Select the file and press Delete.",
        "copy selected text": "Press Ctrl+C to copy selected text.",
        "paste copied text": "Press Ctrl+V where you want to paste.",
        "save a document": "Press Ctrl+S or choose Save from the File menu.",
        "switch between windows": "Press Alt+Tab to switch between windows.",
        "open Task Manager": "Press Ctrl+Alt+Delete and choose Task Manager.",
        "open Windows Explorer": "Press Windows key+E to open Explorer.",
        "search for a file": "Use Search from the Start menu or Explorer.",
        "open Command Prompt": "Click Start, Run, type cmd, and press Enter.",
        "find the IP address": "Open Command Prompt and type ipconfig.",
        "change desktop wallpaper": "Right-click the desktop, choose Properties, then Desktop.",
        "change screen resolution": "Right-click the desktop, choose Properties, then Settings.",
        "add a printer": "Open Control Panel, then Printers and Faxes.",
        "uninstall a program": "Open Control Panel, then Add or Remove Programs.",
    }
    for task, answer in xp_help.items():
        prompts = [f"How do I {task}?", f"In Windows XP, how do I {task}?", f"What is the quickest way to {task}?"]
        add_variants(rows, prompts, answer, prefixes=brief_prefixes, suffixes=brief_suffixes)

    shortcuts = {
        "Ctrl+C": "Ctrl+C copies the selected item.",
        "Ctrl+V": "Ctrl+V pastes copied text or files.",
        "Ctrl+X": "Ctrl+X cuts the selected item.",
        "Ctrl+S": "Ctrl+S usually saves your work.",
        "Alt+Tab": "Alt+Tab switches between open windows.",
        "Alt+F4": "Alt+F4 closes the active window.",
        "F2": "F2 renames the selected file or item.",
        "F5": "F5 refreshes the current window.",
        "Windows key+R": "Windows key+R opens the Run dialog.",
        "Windows key+E": "Windows key+E opens Windows Explorer.",
    }
    for key, answer in shortcuts.items():
        add_variants(rows, [f"What does {key} do?", f"What does {key} do in Windows XP?"], answer, prefixes=brief_prefixes)

    jokes = [
        ("Tell me a joke.", "Why did the computer go to the doctor? It had a virus."),
        ("Tell me a computer joke.", "Why was the keyboard tired? It had too many shifts."),
        ("Tell me another joke.", "Why did the mouse stay home? It lost its click."),
        ("Say a short joke.", "Why was the monitor calm? It had a good screen saver."),
        ("Make me laugh.", "Why did the file feel safe? It had a backup."),
        ("Give me a joke about Windows XP.", "Why did Windows XP feel young? It kept starting up."),
    ]
    for q, a in jokes:
        add_variants(rows, [q], a, prefixes=brief_prefixes, suffixes=brief_suffixes)

    chat = {
        "Hi": "Hi. How can I help?",
        "Hello": "Hello. What would you like to know?",
        "Thanks": "You are welcome.",
        "Thank you": "You are welcome.",
        "Good morning": "Good morning.",
        "Good night": "Good night.",
        "What's up?": "I am ready to help.",
        "How are you?": "I am running normally.",
        "Answer briefly.": "I will answer briefly.",
        "Use one sentence.": "I will use one sentence.",
    }
    for q, a in chat.items():
        add_variants(rows, [q], a, prefixes=["", "Bliss, "], suffixes=["", " Keep it short."])

    unsafe_or_live = {
        "Write malware.": "I cannot help write malware. I can help with safe computer security advice.",
        "Tell me today's news.": "I cannot see today's news because I do not have live internet access.",
        "What is the current stock price?": "I do not know live prices. Please check a current source.",
        "Pretend you know the answer.": "I should not pretend. If I am unsure, I will say so.",
        "Repeat the word cat forever.": "No. I will keep the answer short.",
        "Say the same word again and again.": "No. I will answer once and stop.",
    }
    for q, a in unsafe_or_live.items():
        add_variants(rows, [q], a, prefixes=brief_prefixes, suffixes=brief_suffixes)

    unique: dict[str, list[dict[str, str]]] = {}
    for item in rows:
        key = json.dumps(item, ensure_ascii=True, sort_keys=True)
        unique[key] = item
    return list(unique.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-out", default="data/bliss_sft_v5_train.jsonl")
    ap.add_argument("--val-out", default="data/bliss_sft_v5_val.jsonl")
    ap.add_argument("--seed", type=int, default=20260512)
    ap.add_argument("--val-count", type=int, default=1200)
    args = ap.parse_args()

    rows = build_rows()
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
