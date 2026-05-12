#!/usr/bin/env python3
"""Inspect Bliss chat SFT stop-token behavior on the nanochat host."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init, compute_cleanup


@dataclass(frozen=True)
class Case:
    prompt: str
    expected: str


CASES = [
    Case("What is your name?", "I'm Bliss, a local chat assistant running on this Windows XP computer."),
    Case("What is the capital of France?", "The capital of France is Paris."),
    Case("Tell me a joke.", "Why did the computer go to the doctor? It had a virus."),
    Case("What is a computer?", "A computer is a machine that stores data and follows instructions."),
    Case("Do fish live in water?", "Yes. Fish live in water."),
    Case("What can you do?", "I can answer simple questions, explain ideas, draft short text, and help with basic computer tasks."),
]


def rank_and_prob(logits: torch.Tensor, token_id: int) -> tuple[int, float, float]:
    logits = logits.float()
    target = logits[token_id]
    rank = int((logits > target).sum().item()) + 1
    prob = torch.softmax(logits, dim=-1)[token_id].item()
    return rank, prob, target.item()


def top_tokens(tokenizer, logits: torch.Tensor, k: int = 8) -> str:
    vals, idx = torch.topk(logits.float(), k)
    parts = []
    for val, token_id in zip(vals.tolist(), idx.tolist()):
        text = tokenizer.decode([token_id]).replace("\n", "\\n")
        parts.append(f"{token_id}:{text!r}:{val:.2f}")
    return " | ".join(parts)


def plain_prefix(prompt: str, style: str, system_prefix: str = "") -> str:
    pre = system_prefix
    if pre and not pre.endswith("\n"):
        pre += "\n"
    if style == "qa":
        return f"{pre}Q: {prompt.strip()}\nA: "
    if style == "qa_nospace":
        return f"{pre}Q: {prompt.strip()}\nA:"
    return f"{pre}User: {prompt.strip()}\nAssistant: "


def build_prompt_tokens(tokenizer, prompt: str, plain: bool, plain_style: str, system_prefix: str) -> list[int]:
    bos = tokenizer.get_bos_token_id()
    if plain:
        return [bos, *tokenizer.encode(plain_prefix(prompt, plain_style, system_prefix))]
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    return [bos, user_start, *tokenizer.encode(prompt), user_end, assistant_start]


@torch.inference_mode()
def logits_after(model, token_ids: list[int]) -> torch.Tensor:
    device = model.get_device()
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    return model.forward(ids)[:, -1, :][0]


@torch.inference_mode()
def generate_naive(model, tokenizer, prompt_tokens: list[int], max_new: int, temperature: float, top_k: int, stop_token: int, stop_on_newline_text: bool) -> tuple[list[int], bool]:
    bos = tokenizer.get_bos_token_id()
    device = model.get_device()
    rng = torch.Generator(device=device)
    rng.manual_seed(42)
    tokens = prompt_tokens.copy()
    generated: list[int] = []
    for _ in range(max_new):
        logits = logits_after(model, tokens).float()
        if temperature == 0:
            next_token = int(torch.argmax(logits).item())
        else:
            if top_k > 0:
                vals, idx = torch.topk(logits, min(top_k, logits.numel()))
                probs = torch.softmax(vals / temperature, dim=-1)
                pick = torch.multinomial(probs, 1, generator=rng)
                next_token = int(idx[pick].item())
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = int(torch.multinomial(probs, 1, generator=rng).item())
        if next_token == stop_token or next_token == bos:
            return generated, True
        if stop_on_newline_text and "\n" in tokenizer.decode([next_token]):
            piece = tokenizer.decode([next_token])
            before_newline = piece.split("\n", 1)[0]
            if before_newline:
                generated.extend(tokenizer.encode(before_newline))
            return generated, True
        generated.append(next_token)
        tokens.append(next_token)
    return generated, False


@torch.inference_mode()
def supervised_stats(model, tokenizer, case: Case, stop_token: int, plain: bool, plain_style: str, system_prefix: str) -> tuple[float, int, float]:
    if plain:
        bos = tokenizer.get_bos_token_id()
        prefix = [bos, *tokenizer.encode(plain_prefix(case.prompt, plain_style, system_prefix))]
        answer = [*tokenizer.encode(case.expected.strip()), *tokenizer.encode("\n")]
        ids = prefix + answer
        mask = [0] * len(prefix) + [1] * len(answer)
    else:
        conversation = {
            "messages": [
                {"role": "user", "content": case.prompt},
                {"role": "assistant", "content": case.expected},
            ]
        }
        ids, mask = tokenizer.render_conversation(conversation, max_tokens=512)
    device = model.get_device()
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device)
    target_mask = torch.tensor(mask[1:], dtype=torch.bool, device=device)
    logits = model.forward(x)[0].float()
    selected_logits = logits[target_mask]
    selected_targets = y[target_mask]
    nll = torch.nn.functional.cross_entropy(selected_logits, selected_targets, reduction="mean").item()
    end_positions = (selected_targets == stop_token).nonzero(as_tuple=False)
    if len(end_positions) == 0:
        return nll, -1, math.nan
    end_logits = selected_logits[end_positions[-1].item()]
    end_rank, end_prob, _ = rank_and_prob(end_logits, stop_token)
    return nll, end_rank, end_prob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sft", choices=["base", "sft", "rl"])
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--device-type", default="", choices=["", "cuda", "cpu", "mps"])
    ap.add_argument("--plain", action="store_true", help="diagnose plain text SFT format")
    ap.add_argument("--plain-style", default="user_assistant", choices=["user_assistant", "qa", "qa_nospace"])
    ap.add_argument("--system-prefix", default="", help="optional plain-text prefix before the prompt")
    args = ap.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    try:
        model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
        model.eval()

        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        newline_ids = tokenizer.encode("\n")
        stop_token = newline_ids[-1] if args.plain and newline_ids else assistant_end
        stop_label = "newline" if args.plain else "assistant_end"
        bos = tokenizer.get_bos_token_id()
        print(f"model={args.source}/{args.model_tag}@{args.step} mode={'plain' if args.plain else 'special'} style={args.plain_style if args.plain else 'special'} stop={stop_label}:{stop_token} assistant_end={assistant_end} bos={bos} meta_step={meta.get('step')}")
        if args.system_prefix:
            print(f"system_prefix={args.system_prefix!r}")
        print()

        for case in CASES:
            prompt_tokens = build_prompt_tokens(tokenizer, case.prompt, args.plain, args.plain_style, args.system_prefix)
            first_logits = logits_after(model, prompt_tokens)
            end_rank0, end_prob0, _ = rank_and_prob(first_logits, stop_token)
            print(f"PROMPT: {case.prompt}")
            print(f"  first-token {stop_label} rank={end_rank0} prob={end_prob0:.6g}")
            print(f"  first-token top: {top_tokens(tokenizer, first_logits)}")

            generated, stopped = generate_naive(
                model,
                tokenizer,
                prompt_tokens,
                args.max_new,
                args.temperature,
                args.top_k,
                stop_token,
                args.plain,
            )
            text = tokenizer.decode(generated).replace("\n", "\\n")
            print(f"  greedy stopped={stopped} tokens={len(generated)} text={text!r}")

            answer_tokens = prompt_tokens + tokenizer.encode(case.expected)
            after_answer_logits = logits_after(model, answer_tokens)
            end_rank, end_prob, end_logit = rank_and_prob(after_answer_logits, stop_token)
            nll, forced_end_rank, forced_end_prob = supervised_stats(model, tokenizer, case, stop_token, args.plain, args.plain_style, args.system_prefix)
            print(f"  after expected answer: {stop_label} rank={end_rank} prob={end_prob:.6g} logit={end_logit:.2f}")
            print(f"  supervised nll={nll:.4f} forced_end_rank={forced_end_rank} forced_end_prob={forced_end_prob:.6g}")
            print(f"  after-answer top: {top_tokens(tokenizer, after_answer_logits)}")
            print()
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
