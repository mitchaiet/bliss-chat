#!/usr/bin/env python3
"""Patch nanochat's chat_sft.py for Bliss SFT runs.

The patch is intentionally small and reversible:
- if BLISS_SFT_JSONL is set, train/eval only on that CustomJSON data;
- abort on non-finite/high loss;
- manually clip gradients without torch's inf-times-zero trap.
- optionally upweight first assistant tokens and assistant_end.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        if new in text:
            return text
        raise SystemExit(f"patch anchor not found:\n{old[:400]}")
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nanochat", default="~/nanochat")
    args = ap.parse_args()

    path = Path(args.nanochat).expanduser() / "scripts" / "chat_sft.py"
    text = path.read_text()
    backup = path.with_suffix(path.suffix + ".blissbak")
    if not backup.exists():
        backup.write_text(text)
    else:
        # Always reapply from the original file. The training scripts call this
        # helper on every run, so patching the already-patched file would stack
        # duplicate imports and gradient clipping blocks.
        text = backup.read_text()

    text = replace_once(text, "import gc\nimport argparse\n", "import gc\nimport argparse\nimport math\n")
    text = replace_once(text, "import torch\n", "import torch\nimport torch.nn.functional as F\n")

    text = replace_once(
        text,
        "args = parser.parse_args()\nuser_config = vars(args).copy()\n",
        "args = parser.parse_args()\n"
        "BLISS_GRAD_CLIP = float(os.environ.get(\"BLISS_GRAD_CLIP\", \"0\"))\n"
        "BLISS_ABORT_LOSS = float(os.environ.get(\"BLISS_ABORT_LOSS\", \"0\"))\n"
        "BLISS_FIRST_TOKEN_WEIGHT = float(os.environ.get(\"BLISS_FIRST_TOKEN_WEIGHT\", \"1\"))\n"
        "BLISS_END_TOKEN_WEIGHT = float(os.environ.get(\"BLISS_END_TOKEN_WEIGHT\", \"1\"))\n"
        "BLISS_PLAIN_CHAT = os.environ.get(\"BLISS_PLAIN_CHAT\", \"0\") == \"1\"\n"
        "BLISS_PLAIN_STYLE = os.environ.get(\"BLISS_PLAIN_STYLE\", \"user_assistant\")\n"
        "BLISS_SYSTEM_PREFIX = os.environ.get(\"BLISS_SYSTEM_PREFIX\", \"\")\n"
        "BLISS_TRAIN_SCOPE = os.environ.get(\"BLISS_TRAIN_SCOPE\", \"full\")\n"
        "user_config = vars(args).copy()\n"
        "user_config.update({\"bliss_sft_jsonl\": os.environ.get(\"BLISS_SFT_JSONL\", \"\"), \"bliss_grad_clip\": BLISS_GRAD_CLIP, \"bliss_abort_loss\": BLISS_ABORT_LOSS, \"bliss_first_token_weight\": BLISS_FIRST_TOKEN_WEIGHT, \"bliss_end_token_weight\": BLISS_END_TOKEN_WEIGHT, \"bliss_plain_chat\": BLISS_PLAIN_CHAT, \"bliss_plain_style\": BLISS_PLAIN_STYLE, \"bliss_system_prefix\": BLISS_SYSTEM_PREFIX, \"bliss_train_scope\": BLISS_TRAIN_SCOPE})\n",
    )

    text = replace_once(
        text,
        "# Inherit training hyperparameters from pretrained checkpoint (None = inherit, explicit value = override)\n",
        "if BLISS_TRAIN_SCOPE != \"full\":\n"
        "    allowed = {\n"
        "        \"lm_head\": lambda n: n == \"lm_head.weight\",\n"
        "        \"head_value\": lambda n: n == \"lm_head.weight\" or n.startswith(\"value_embeds.\"),\n"
        "    }\n"
        "    if BLISS_TRAIN_SCOPE not in allowed:\n"
        "        raise ValueError(f\"unknown BLISS_TRAIN_SCOPE={BLISS_TRAIN_SCOPE}\")\n"
        "    kept = 0\n"
        "    total = 0\n"
        "    for name, param in model.named_parameters():\n"
        "        keep = allowed[BLISS_TRAIN_SCOPE](name)\n"
        "        param.requires_grad_(keep)\n"
        "        total += param.numel()\n"
        "        if keep:\n"
        "            kept += param.numel()\n"
        "    print0(f\"Bliss train scope: {BLISS_TRAIN_SCOPE} trainable={kept:,}/{total:,} params\")\n\n"
        "# Inherit training hyperparameters from pretrained checkpoint (None = inherit, explicit value = override)\n",
    )

    start = text.index("# SFT data mixture and DataLoader")
    end = text.index("# DataLoader is defined here", start)
    data_block = '''# SFT data mixture and DataLoader
bliss_sft_jsonl = os.environ.get("BLISS_SFT_JSONL")
bliss_val_jsonl = os.environ.get("BLISS_VAL_JSONL", bliss_sft_jsonl or "")
if bliss_sft_jsonl:
    train_dataset = TaskMixture([CustomJSON(filepath=bliss_sft_jsonl)])
    val_dataset = TaskMixture([CustomJSON(filepath=bliss_val_jsonl)])
    print0(f"Training mixture: Bliss CustomJSON train={bliss_sft_jsonl} val={bliss_val_jsonl} rows={len(train_dataset):,}")
else:
    identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
    train_tasks = [
        SmolTalk(split="train"), # 460K rows of general conversations
        CustomJSON(filepath=identity_conversations_filepath), # 1000 rows of synthetic identity conversations
        CustomJSON(filepath=identity_conversations_filepath), # 2 epochs of these
        *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)], # 100K rows per epoch
        *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)], # 8K rows per epoch
        SimpleSpelling(size=200000, split="train"), # 200K rows of Simple Spelling (e.g. spell the word 'apple')
        SpellingBee(size=80000, split="train"), # 80K rows of Spelling Bee (e.g. how many 'r' are in 'strawberry'?)
    ]
    train_dataset = TaskMixture(train_tasks)
    print0(f"Training mixture: {len(train_dataset):,} rows (MMLU x{args.mmlu_epochs}, GSM8K x{args.gsm8k_epochs})")
    val_dataset = TaskMixture([
        SmolTalk(split="test"), # 24K rows in test set
        MMLU(subset="all", split="test", stop=5200), # 14K rows in test set, use only 5.2K to match the train ratios
        GSM8K(subset="main", split="test", stop=420), # 1.32K rows in test set, use only 420 to match the train ratios
    ]) # total: 24K + 5.2K + 0.42K ~= 29.6K rows
'''
    text = text[:start] + data_block + text[end:]

    text = replace_once(
        text,
        "token_bytes = get_token_bytes(device=device)\n\n# Initialize the Optimizer",
        "token_bytes = get_token_bytes(device=device)\n"
        "assistant_start_token = tokenizer.encode_special(\"<|assistant_start|>\")\n"
        "assistant_end_token = tokenizer.encode_special(\"<|assistant_end|>\")\n"
        "plain_assistant_marker = \"A:\" if BLISS_PLAIN_STYLE == \"qa_nospace\" else (\"A: \" if BLISS_PLAIN_STYLE == \"qa\" else \"Assistant: \")\n"
        "plain_assistant_token_ids = tokenizer.encode(plain_assistant_marker)\n"
        "print0(f\"Bliss loss weights: first={BLISS_FIRST_TOKEN_WEIGHT:g} assistant_end={BLISS_END_TOKEN_WEIGHT:g} plain={BLISS_PLAIN_CHAT} style={BLISS_PLAIN_STYLE}\")\n\n"
        "# Initialize the Optimizer",
    )

    text = replace_once(
        text,
        "optimizer = model.setup_optimizer(unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr, matrix_lr=args.matrix_lr, weight_decay=0.0)\n\n# Optionally warm-start optimizer from pretrained checkpoint (momentum buffers etc.)",
        "optimizer = model.setup_optimizer(unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr, matrix_lr=args.matrix_lr, weight_decay=0.0)\n"
        "if BLISS_TRAIN_SCOPE != \"full\":\n"
        "    kept_groups = []\n"
        "    kept_params = 0\n"
        "    for group in optimizer.param_groups:\n"
        "        group[\"params\"] = [p for p in group[\"params\"] if p.requires_grad]\n"
        "        kept_params += len(group[\"params\"])\n"
        "        if group[\"params\"]:\n"
        "            kept_groups.append(group)\n"
        "    optimizer.param_groups[:] = kept_groups\n"
        "    print0(f\"Bliss optimizer scope: groups={len(optimizer.param_groups)} params={kept_params}\")\n\n"
        "# Optionally warm-start optimizer from pretrained checkpoint (momentum buffers etc.)",
    )

    text = replace_once(
        text,
        "train_loader = sft_data_generator_bos_bestfit(\"train\")\n",
        "def bliss_render_conversation(conversation):\n"
        "    if not BLISS_PLAIN_CHAT:\n"
        "        return tokenizer.render_conversation(conversation)\n"
        "    ids = [tokenizer.get_bos_token_id()]\n"
        "    mask = [0]\n"
        "    def add(text, bit):\n"
        "        toks = tokenizer.encode(text)\n"
        "        ids.extend(toks)\n"
        "        mask.extend([bit] * len(toks))\n"
        "    if BLISS_SYSTEM_PREFIX:\n"
        "        prefix = BLISS_SYSTEM_PREFIX if BLISS_SYSTEM_PREFIX.endswith(\"\\n\") else BLISS_SYSTEM_PREFIX + \"\\n\"\n"
        "        add(prefix, 0)\n"
        "    def content_text(value):\n"
        "        if isinstance(value, str):\n"
        "            return value.strip()\n"
        "        if isinstance(value, list):\n"
        "            parts = []\n"
        "            for item in value:\n"
        "                if isinstance(item, str):\n"
        "                    parts.append(item)\n"
        "                elif isinstance(item, dict):\n"
        "                    for key in (\"text\", \"content\", \"value\"):\n"
        "                        if isinstance(item.get(key), str):\n"
        "                            parts.append(item[key])\n"
        "                            break\n"
        "            return \" \".join(parts).strip()\n"
        "        return str(value).strip()\n"
        "    for message in conversation[\"messages\"]:\n"
        "        if message[\"role\"] == \"system\":\n"
        "            content = content_text(message.get(\"content\", \"\"))\n"
        "            if content:\n"
        "                add(f\"{content}\\n\", 0)\n"
        "        elif message[\"role\"] == \"user\":\n"
        "            content = content_text(message.get(\"content\", \"\"))\n"
        "            if BLISS_PLAIN_STYLE == \"qa\":\n"
        "                add(f\"Q: {content}\\nA: \", 0)\n"
        "            elif BLISS_PLAIN_STYLE == \"qa_nospace\":\n"
        "                add(f\"Q: {content}\\nA:\", 0)\n"
        "            else:\n"
        "                add(f\"User: {content}\\nAssistant: \", 0)\n"
        "        elif message[\"role\"] == \"assistant\":\n"
        "            add(content_text(message.get(\"content\", \"\")), 1)\n"
        "            add(\"\\n\", 1)\n"
        "        else:\n"
        "            content = content_text(message.get(\"content\", \"\"))\n"
        "            if content:\n"
        "                add(f\"{content}\\n\", 0)\n"
        "    return ids, mask\n\n"
        "train_loader = sft_data_generator_bos_bestfit(\"train\")\n",
    )

    text = replace_once(
        text,
        "            ids, mask = tokenizer.render_conversation(conversation)\n",
        "            ids, mask = bliss_render_conversation(conversation)\n",
    )

    text = replace_once(
        text,
        "        loss = model(x, y)\n        train_loss = loss.detach() # for logging\n",
        "        if BLISS_FIRST_TOKEN_WEIGHT != 1.0 or BLISS_END_TOKEN_WEIGHT != 1.0:\n"
        "            logits = model(x)\n"
        "            token_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1, reduction=\"none\").view_as(y)\n"
        "            weights = (y != -1).float()\n"
        "            if BLISS_FIRST_TOKEN_WEIGHT != 1.0:\n"
        "                if BLISS_PLAIN_CHAT:\n"
        "                    first_mask = torch.zeros_like(weights, dtype=torch.bool)\n"
        "                    marker_len = len(plain_assistant_token_ids)\n"
        "                    if marker_len > 0 and x.size(1) >= marker_len:\n"
        "                        candidate = torch.ones((x.size(0), x.size(1) - marker_len + 1), device=x.device, dtype=torch.bool)\n"
        "                        for marker_offset, marker_token in enumerate(plain_assistant_token_ids):\n"
        "                            candidate &= x[:, marker_offset:marker_offset + candidate.size(1)] == marker_token\n"
        "                        first_mask[:, marker_len - 1:] = candidate\n"
        "                    weights[first_mask & (y != -1)] *= BLISS_FIRST_TOKEN_WEIGHT\n"
        "                else:\n"
        "                    weights[(x == assistant_start_token) & (y != -1)] *= BLISS_FIRST_TOKEN_WEIGHT\n"
        "            if BLISS_END_TOKEN_WEIGHT != 1.0:\n"
        "                if BLISS_PLAIN_CHAT:\n"
        "                    newline_ids = tokenizer.encode(\"\\n\")\n"
        "                    if newline_ids:\n"
        "                        weights[y == newline_ids[-1]] *= BLISS_END_TOKEN_WEIGHT\n"
        "                else:\n"
        "                    weights[y == assistant_end_token] *= BLISS_END_TOKEN_WEIGHT\n"
        "            loss = (token_loss * weights).sum() / weights.sum().clamp_min(1.0)\n"
        "        else:\n"
        "            loss = model(x, y)\n"
        "        if not torch.isfinite(loss).all():\n"
        "            print0(f\"BLISS_ABORT non-finite loss at step {step}: {loss}\")\n"
        "            raise SystemExit(20)\n"
        "        if BLISS_ABORT_LOSS > 0 and float(loss.detach().item()) > BLISS_ABORT_LOSS:\n"
        "            print0(f\"BLISS_ABORT loss {float(loss.detach().item()):.6f} > {BLISS_ABORT_LOSS:.6f} at step {step}\")\n"
        "            raise SystemExit(21)\n"
        "        train_loss = loss.detach() # for logging\n",
    )

    text = replace_once(
        text,
        "    if scaler is not None:\n        scaler.unscale_(optimizer)\n",
        "    if BLISS_GRAD_CLIP > 0:\n"
        "        grad_norm_sq = torch.zeros((), device=device)\n"
        "        bad_grad = False\n"
        "        for p in model.parameters():\n"
        "            if p.grad is None:\n"
        "                continue\n"
        "            if not torch.isfinite(p.grad).all():\n"
        "                bad_grad = True\n"
        "                break\n"
        "            grad_norm_sq += p.grad.detach().float().pow(2).sum()\n"
        "        if ddp:\n"
        "            dist.all_reduce(grad_norm_sq, op=dist.ReduceOp.SUM)\n"
        "        grad_norm = torch.sqrt(grad_norm_sq)\n"
        "        if bad_grad or not torch.isfinite(grad_norm).all():\n"
        "            print0(f\"BLISS_ABORT non-finite grad at step {step}\")\n"
        "            raise SystemExit(22)\n"
        "        clip_coef = min(1.0, BLISS_GRAD_CLIP / (float(grad_norm.item()) + 1e-6))\n"
        "        if clip_coef < 1.0:\n"
        "            for p in model.parameters():\n"
        "                if p.grad is not None:\n"
        "                    p.grad.mul_(clip_coef)\n"
        "    if scaler is not None:\n        scaler.unscale_(optimizer)\n",
    )

    text = replace_once(
        text,
        "            if consumed >= dataset_size:\n                last_step = True\n",
        "            if args.num_iterations <= 0 and consumed >= dataset_size:\n                last_step = True\n",
    )

    text = replace_once(
        text,
        "def get_lr_multiplier(progress):\n    if progress < args.warmup_ratio:\n",
        "def get_lr_multiplier(progress):\n    progress = max(0.0, min(float(progress), 1.0))\n    if progress < args.warmup_ratio:\n",
    )

    path.write_text(text)
    print(f"[patch] patched {path}")
    print(f"[patch] backup {backup}")


if __name__ == "__main__":
    main()
