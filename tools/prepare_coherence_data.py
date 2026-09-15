#!/usr/bin/env python3
"""Prepare bounded, reproducible SmolLM2 SFT data without reading any benchmarks.

Only reads a pinned public Smol-SmolTalk snapshot and tokenizer files. Public
answers remain intact; a long dialogue may contribute a complete prefix ending
at an assistant turn. New procedural records have independently partitioned
entity pools, arithmetic operands, and prompt families for training/validation.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
from pathlib import Path

DATA_REVISION = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"
MODEL_REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"
DATASET_ID = "HuggingFaceTB/smol-smoltalk"
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
SYSTEM = (
    "You are Bliss, a helpful local assistant running on Windows XP. "
    "Give a clear, concise answer, usually in one to three sentences. "
    "Use the conversation and supplied notes when relevant. "
    "If the supplied information does not answer the question, say so."
)
# Keep conversational examples dominant. Constraints must not dominate just
# because their answers tend to be short. All source names are from the card.
SOURCE_WEIGHTS = {
    "smol-magpie-ultra-short": 0.65,
    "everyday-conversations": 0.10,
    "smol-summarize-20k": 0.10,
    "smollm-rewrite-30k": 0.10,
    "smol-contraints": 0.05,
}
POOL = {
    "train": {
        "names": ["Amara", "Belen", "Cato", "Dara", "Emil", "Fara", "Galen", "Hana", "Idris", "Jori", "Kavi", "Leona", "Marek", "Nadia", "Oren", "Petra", "Quin", "Ravi", "Sana", "Tavi"],
        "places": ["amber desk", "birch shelf", "copper door", "dune room", "elm counter", "fern gate", "granite table", "hazel cabinet", "indigo box", "jade bench"],
        "items": ["pencils", "buttons", "bottles", "postcards", "ribbons", "marbles", "notebooks", "envelopes", "shells", "badges"],
        "numbers": list(range(4, 100, 2)),
    },
    "val": {
        "names": ["Avel", "Brina", "Cyril", "Demi", "Elian", "Fenna", "Gita", "Hollis", "Iona", "Jules", "Kira", "Lior"],
        "places": ["lilac drawer", "maple cupboard", "navy tray", "opal stand", "pearl alcove", "quartz basket", "russet closet", "silver hatch"],
        "items": ["stamps", "pebbles", "spools", "folders", "clips", "beads", "cards", "tiles"],
        "numbers": list(range(5, 100, 2)),
    },
}


def digest(value) -> str:
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def signature(messages) -> str:
    return digest([(m["role"], normalize(m["content"])) for m in messages])


def prompt_signature(messages) -> str:
    # Group entire first user questions across public train/test. This can reject
    # legitimate shared greetings; conservative removal is preferable here.
    return digest(next(normalize(m["content"]) for m in messages if m["role"] == "user"))


class Renderer:
    def __init__(self, tokenizer_dir: Path, max_length: int):
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
        cfg = json.loads((tokenizer_dir / "tokenizer_config.json").read_text())
        loader = PreTrainedTokenizerFast if cfg.get("tokenizer_class") == "TokenizersBackend" else AutoTokenizer
        compat = {"extra_special_tokens": {}} if cfg.get("tokenizer_class") == "TokenizersBackend" else {}
        self.tokenizer = loader.from_pretrained(str(tokenizer_dir),
            local_files_only=True, trust_remote_code=False, **compat)
        if not self.tokenizer.chat_template:
            raise ValueError("Expected a model-native chat template")
        self.max_length = max_length

    def ids(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def render(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=False)

    def record(self, messages, source, split, family, **metadata):
        token_count = len(self.ids(self.render(messages)))
        if token_count > self.max_length:
            return None
        assistant_tokens = sum(len(self.ids(m["content"])) + 1 for m in messages if m["role"] == "assistant")
        if not assistant_tokens:
            return None
        return {
            "id": signature(messages), "messages": messages,
            "source": source, "split": split, "family": family,
            "token_count": token_count, "assistant_token_count": assistant_tokens,
            **metadata,
        }


def public_candidates(data_dir, split, renderer, max_answer_tokens, seed, rejects):
    import pyarrow.parquet as pq
    candidates = collections.defaultdict(list)
    seen = set()
    for path in sorted(data_dir.glob(f"{split}-*.parquet")):
        row_index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                index = row_index
                row_index += 1
                source = row.get("source")
                if source not in SOURCE_WEIGHTS:
                    continue
                original = row["messages"]
                if any(m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str) for m in original):
                    rejects["public_invalid_roles_or_content"] += 1
                    continue
                # Preserve task-specific system instructions, otherwise use the
                # same explicit system as deployment, avoiding implicit SmolLM identity.
                if original and original[0]["role"] == "system":
                    messages = [dict(original[0])]
                    original = original[1:]
                else:
                    messages = [{"role": "system", "content": SYSTEM}]
                kept = 0
                for i in range(0, len(original) - 1, 2):
                    user, answer = original[i:i + 2]
                    if user["role"] != "user" or answer["role"] != "assistant":
                        break
                    text = answer["content"].strip()
                    if len(text) < 3 or len(text) > 2000 or len(user["content"]) > 5000:
                        break
                    if any(marker in user["content"] + text for marker in ("<|im_start|>", "<|im_end|>")):
                        break
                    # Public answers are never shortened or ASCII-stripped.
                    n = len(renderer.ids(text))
                    if n > max_answer_tokens:
                        break
                    grams = [tuple(normalize(text).split()[j:j + 4]) for j in range(max(0, len(text.split()) - 3))]
                    if grams and collections.Counter(grams).most_common(1)[0][1] >= 3:
                        break
                    proposed = messages + [dict(user), dict(answer)]
                    if len(renderer.ids(renderer.render(proposed))) > renderer.max_length:
                        break
                    messages = proposed
                    kept += 1
                    if kept == 4:
                        break
                if not kept:
                    rejects["public_no_complete_short_prefix"] += 1
                    continue
                key = signature(messages)
                if key in seen:
                    rejects["public_duplicate_within_split"] += 1
                    continue
                seen.add(key)
                rec = renderer.record(messages, f"{DATASET_ID}/{source}", split, "public_retention", public_file=path.name, public_row=index, public_revision=DATA_REVISION, original_assistant_turns=sum(m["role"] == "assistant" for m in original), selected_assistant_turns=kept)
                candidates[source].append(rec)
    # Hash order is reproducible and independent of source file ordering.
    for source, rows in candidates.items():
        rows.sort(key=lambda r: digest([seed, split, r["id"]]))
    return candidates


def select_public(candidates, requested, exclude_prompts, rejects):
    selected = []
    for source, weight in SOURCE_WEIGHTS.items():
        quota = round(requested * weight)
        for row in candidates.get(source, []):
            if prompt_signature(row["messages"]) in exclude_prompts:
                rejects["public_cross_split_prompt_group"] += 1
                continue
            selected.append(row)
            quota -= 1
            if quota == 0:
                break
        if quota:
            rejects[f"public_unfilled_quota:{source}"] += quota
    if not selected:
        raise ValueError("No suitable public examples found")
    return selected


def procedural_example(split, index, rng):
    p = POOL[split]
    name, other, third = rng.sample(p["names"], 3)
    place, second, destination = rng.sample(p["places"], 3)
    item, other_item = rng.sample(p["items"], 2)
    a, b, c = rng.sample(p["numbers"], 3)
    val = split == "val"
    family = index % 8
    messages = [{"role": "system", "content": SYSTEM}]
    def turn(user, answer):
        messages.extend([{"role": "user", "content": user}, {"role": "assistant", "content": answer}])
    if family == 0:
        if val:
            turn(f"Please keep this detail in mind: {name} will collect the {item} at the {place}.", "I will use that detail in this conversation.")
            turn(f"Separately, {other} is taking {other_item} to the {second}.", f"Understood; {other}'s destination is the {second}.")
            turn(f"Where is {name}'s collection point?", f"{name} will collect the {item} at the {place}.")
        else:
            turn(f"{name}'s {item} are at the {place}. Remember that for this chat.", f"I will remember that {name}'s {item} are at the {place} during this chat.")
            turn(f"{other} has {b} {other_item} at the {second}.", f"Got it; {other} has {b} {other_item} there.")
            turn(f"Where are {name}'s {item}?", f"{name}'s {item} are at the {place}.")
        category = "memory_distractor"
    elif family == 1:
        if val:
            turn(f"The meeting with {name} is planned for the {place}.", f"The meeting with {name} is planned for the {place}.")
            turn(f"Change of plan: it will be at the {second}, replacing the earlier location.", f"The updated meeting location is the {second}.")
            turn("Which location should I use now?", f"Use the {second}.")
        else:
            turn(f"The delivery for {name} goes to the {place}.", f"The delivery for {name} goes to the {place}.")
            turn(f"Correction: send it to the {second} instead.", f"I will use the updated destination, the {second}.")
            turn(f"Where should {name}'s delivery go?", f"{name}'s delivery should go to the {second}.")
        category = "memory_correction"
    elif family == 2:
        messages[0]["content"] += f"\nNotes supplied for this conversation: {name} keeps {item} at the {place}. {other} keeps {other_item} at the {second}."
        turn(f"Locate the {item} that belong to {name}." if val else f"Where does {name} keep the {item}?", f"{name} keeps the {item} at the {place}.")
        category = "notes_selective"
    elif family == 3:
        if val:
            turn(f"Reference note: {name} stores {item} at the {place}. Using only this note, identify {name}'s preferred drink.", f"The note does not give {name}'s preferred drink.")
        else:
            turn(f"Context: {name} left {a} {item} at the {place}. Based only on this context, what time were they left there?", "The context does not say what time they were left there.")
        category = "grounded_unknown"
    elif family == 4:
        if val:
            turn(f"{name} has two packs containing {a} and {b} {item}. What is the combined count?", f"There are {a + b} {item} in total: {a} + {b} = {a + b}.")
        else:
            turn(f"A box holds {a} {item}, and I add {b} more {item}. How many {item} are in the box now?", f"The box now holds {a + b} {item}, since {a} + {b} = {a + b}.")
        category = "reasoning_addition"
    elif family == 5:
        high, low = max(a, b), min(a, b)
        if val:
            turn(f"An inventory starts at {high} {item} and ends at {low}. How many {item} were removed?", f"{high - low} {item} were removed, because {high} - {low} = {high - low}.")
        else:
            turn(f"{name} has {high} {item} and gives {low} to {other}. How many {item} does {name} have left?", f"{name} has {high - low} {item} left, since {high} - {low} = {high - low}.")
        category = "reasoning_subtraction"
    elif family == 6:
        if val:
            turn(f"A queue contains just {name}, {other}, and {third}, in that order from front to back. Identify the person between the other two.", f"{other} is between {name} and {third}.")
        else:
            turn(f"Three people stand in a line: {name} first, {other} second, and {third} third. Who is in the middle?", f"{other} is in the middle.")
        category = "reasoning_order"
    else:
        if val:
            turn(f"For this exercise, every parcel from {name} is routed to the {place}. This parcel comes from {name}. State its destination.", f"Its destination is the {place}, because all parcels from {name} go there.")
        else:
            turn(f"Rule: all {item} go to the {place}. {name} is carrying {item}. Where should {name} put them?", f"{name} should put the {item} at the {place}, following the rule.")
        category = "reasoning_rule"
    return messages, category, f"{category}:{split}_wording"


def make_procedural(split, target_tokens, renderer, seed, excluded, rejects):
    rows = []
    count = 0
    rng = random.Random(seed + (100003 if split == "val" else 0))
    seen = set(excluded)
    for index in range(200000):
        if count >= target_tokens:
            break
        messages, category, family = procedural_example(split, index, rng)
        row = renderer.record(messages, "bliss-procedural-v1", split, family, category=category, provenance="deterministic facts and arithmetic; no teacher or benchmark inputs")
        if row is None:
            rejects["procedural_too_long"] += 1
            continue
        if row["id"] in seen:
            rejects["procedural_duplicate"] += 1
            continue
        seen.add(row["id"])
        rows.append(row)
        count += row["token_count"]
    if count < target_tokens:
        raise RuntimeError("Procedural unique-example budget exhausted")
    return rows


def stats(rows):
    total_tokens = sum(r["token_count"] for r in rows)
    procedural_tokens = sum(r["token_count"] for r in rows if r["source"] == "bliss-procedural-v1")
    return {
        "rows": len(rows), "rendered_tokens": total_tokens,
        "assistant_tokens_estimate_including_eos": sum(r["assistant_token_count"] for r in rows),
        "procedural_token_fraction": procedural_tokens / total_tokens,
        "sources": dict(collections.Counter(r["source"] for r in rows)),
        "categories": dict(collections.Counter(r.get("category", "public_retention") for r in rows)),
        "max_rendered_tokens": max(r["token_count"] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    hub = Path.home() / ".cache/huggingface/hub"
    ap.add_argument("--public-data", type=Path, default=hub / f"datasets--HuggingFaceTB--smol-smoltalk/snapshots/{DATA_REVISION}/data")
    ap.add_argument("--tokenizer", type=Path, default=hub / f"models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/{MODEL_REVISION}")
    ap.add_argument("--tokenizer-id", help="Publisher ID for provenance when using another tokenizer")
    ap.add_argument("--tokenizer-revision", help="Pinned publisher revision for the selected tokenizer")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--public-train-rows", type=int, default=16000)
    ap.add_argument("--public-val-rows", type=int, default=800)
    ap.add_argument("--procedural-token-fraction", type=float, default=0.25)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-answer-tokens", type=int, default=192)
    ap.add_argument("--seed", type=int, default=20260915)
    args = ap.parse_args()
    if not (0.0 < args.procedural_token_fraction < 0.5):
        ap.error("Procedural token fraction must be between zero and 0.5")
    if args.public_train_rows < 20 or args.public_val_rows < 20:
        ap.error("Public row targets must each be at least 20")
    if args.out.exists() and any(args.out.iterdir()):
        ap.error("Output directory must be absent or empty; refusing to overwrite a recipe")
    for split in ("train", "test"):
        if not list(args.public_data.glob(f"{split}-*.parquet")):
            ap.error(f"Missing pinned public {split} parquet files in {args.public_data}")
    renderer = Renderer(args.tokenizer, args.max_length)
    rejects = collections.Counter()
    print("Selecting held-out public test records...", flush=True)
    val_candidates = public_candidates(args.public_data, "test", renderer, args.max_answer_tokens, args.seed, rejects)
    public_val = select_public(val_candidates, args.public_val_rows, set(), rejects)
    val_groups = {prompt_signature(r["messages"]) for rows in val_candidates.values() for r in rows}
    del val_candidates
    print("Selecting public training records...", flush=True)
    train_candidates = public_candidates(args.public_data, "train", renderer, args.max_answer_tokens, args.seed, rejects)
    public_train = select_public(train_candidates, args.public_train_rows, val_groups, rejects)
    del train_candidates
    # Fail if broad conversations cannot remain dominant rather than quietly
    # replacing them with a mass of formatting/summarization tasks.
    if len(public_train) < args.public_train_rows * 0.70:
        raise RuntimeError("Too few public examples for source quotas; inspect eligibility counts before changing the recipe")
    output = {}
    for split, public in (("train", public_train), ("val", public_val)):
        for r in public:
            r["split"] = split
        base_tokens = sum(r["token_count"] for r in public)
        target = math.ceil(base_tokens * args.procedural_token_fraction / (1 - args.procedural_token_fraction))
        excluded = {r["id"] for rows in output.values() for r in rows}
        generated = make_procedural(split, target, renderer, args.seed, excluded, rejects)
        rows = public + generated
        random.Random(args.seed + (1 if split == "val" else 0)).shuffle(rows)
        output[split] = rows
    train_ids = {r["id"] for r in output["train"]}
    val_ids = {r["id"] for r in output["val"]}
    if len(train_ids) != len(output["train"]) or len(val_ids) != len(output["val"]) or train_ids & val_ids:
        raise AssertionError("Conversation duplicates within or across final splits")
    procedural_train_families = {r["family"] for r in output["train"] if r["source"] == "bliss-procedural-v1"}
    procedural_val_families = {r["family"] for r in output["val"] if r["source"] == "bliss-procedural-v1"}
    assert not procedural_train_families & procedural_val_families
    for key in POOL["train"]:
        assert not set(POOL["train"][key]) & set(POOL["val"][key])
    args.out.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, rows in output.items():
        path = args.out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        files[path.name] = file_digest(path)
    inputs = {}
    for p in sorted(args.public_data.glob("*.parquet")):
        inputs[p.name] = {"sha256": file_digest(p), "bytes": p.stat().st_size}
    manifest = {
        "recipe": "bliss-coherence-v1", "seed": args.seed,
        "generator_sha256": file_digest(Path(__file__)),
        "dataset": {"id": DATASET_ID, "revision": DATA_REVISION, "license": "Apache-2.0 as published by HuggingFaceTB", "card": f"https://huggingface.co/datasets/{DATASET_ID}", "input_files": inputs},
        "tokenizer": {"id": args.tokenizer_id or (MODEL_ID if args.tokenizer.name == MODEL_REVISION else "local"), "revision": args.tokenizer_revision or (MODEL_REVISION if args.tokenizer.name == MODEL_REVISION else None), "tokenizer_json_sha256": file_digest(args.tokenizer / "tokenizer.json"), "tokenizer_config_sha256": file_digest(args.tokenizer / "tokenizer_config.json"), "chat_template_sha256": digest(renderer.tokenizer.chat_template)},
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "system_message": SYSTEM, "source_weights_by_public_rows": SOURCE_WEIGHTS,
        "splits": {k: stats(v) for k, v in output.items()}, "output_sha256": files,
        "rejections_or_unfilled_quotas": dict(rejects),
        "split_checks": {"exact_conversation_overlap": 0, "public_first_user_overlap": 0, "procedural_entity_pool_overlap": 0, "procedural_prompt_family_overlap": 0},
        "limitations": ["Public test split is held out from this adaptation; upstream model training exposure cannot be excluded.", "Procedural validation uses new entities and phrasings of the same task types; it is not an independent general-reasoning benchmark.", "No benchmark files or existing Bliss curated data were read.", "Tokens count the pinned native chat template; assistant-token counts are estimates for reporting, not training labels.", "Public prefixes retain complete original assistant responses; rare factual errors in public data remain possible.", "No tuning gain is assumed; the unchanged instruct baseline must remain eligible for selection."],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "splits": manifest["splits"], "sha256": files}, indent=2), flush=True)


if __name__ == "__main__":
    main()
