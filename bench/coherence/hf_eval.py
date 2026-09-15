#!/usr/bin/env python3
"""Local-files-only HF evaluation of the sealed Bliss cases. Parent launches GPU use.

Example: python hf_eval.py --model /cached/model/snapshot --out /new/results
This measures HF semantic quality and GPU generation latency, not XP performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

from evaluate import CASES, load_cases, score, sha256, summarize

SYSTEM = ("You are Bliss, a helpful local assistant running on Windows XP. "
          "Answer clearly and briefly, use the conversation and supplied notes, "
          "and say when you do not know.")


def render(tokenizer, system, history, max_prompt_tokens):
    history = list(history)
    removed = 0
    while True:
        messages = [{"role": "system", "content": system}] + history
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                            return_tensors="pt")
        if ids.shape[-1] <= max_prompt_tokens:
            return ids, removed
        if len(history) <= 1:
            raise ValueError(f"Current prompt has {ids.shape[-1]} tokens; budget is {max_prompt_tokens}")
        # Discard whole oldest user/assistant pairs; never substitute gold answers.
        history = history[2:]
        removed += 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cases", type=Path, default=CASES)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--cpu-threads", type=int, help="Set the CPU inference thread count explicitly")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="Explicitly override publisher generation defaults for backend parity")
    ap.add_argument("--system", default=SYSTEM)
    ap.add_argument("--categories")
    args = ap.parse_args()
    assert args.model.is_dir(), "--model must be an existing local model directory"
    tok_dir = args.tokenizer or args.model
    assert tok_dir.is_dir(), "tokenizer directory must exist locally"
    assert args.ctx > args.max_new_tokens > 0
    cases = load_cases(args.cases)
    if args.categories:
        selected = set(args.categories.split(","))
        cases = [c for c in cases if c["category"] in selected]
        assert cases, "no categories selected"
    args.out.mkdir(parents=True, exist_ok=False)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

    assert args.device != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable"
    if args.cpu_threads is not None:
        assert args.cpu_threads > 0
        torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(42)
    tokenizer_config_path = tok_dir / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text()) if tokenizer_config_path.exists() else {}
    # New publisher files can name the v5 generic backend class; v4's direct fast
    # tokenizer can still read the same tokenizer JSON and template without edits.
    tokenizer_class = (PreTrainedTokenizerFast if tokenizer_config.get("tokenizer_class") == "TokenizersBackend"
                       else AutoTokenizer)
    tokenizer_kwargs = {}
    if tokenizer_class is PreTrainedTokenizerFast and tokenizer_config.get("extra_special_tokens") == []:
        # v5 serializes the empty optional mapping as []; v4 requires a dict.
        tokenizer_kwargs["extra_special_tokens"] = {}
    tokenizer = tokenizer_class.from_pretrained(str(tok_dir), local_files_only=True, trust_remote_code=False,
                                                **tokenizer_kwargs)
    if not tokenizer.chat_template and (tok_dir / "chat_template.jinja").exists():
        tokenizer.chat_template = (tok_dir / "chat_template.jinja").read_text()
    assert tokenizer.chat_template, "The cached tokenizer must include its model-native chat template"
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=False,
        torch_dtype=getattr(torch, args.dtype), attn_implementation="sdpa")
    model = model.to(args.device).eval()
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    model_load_s = time.perf_counter() - started
    model_files = [p for p in sorted(args.model.iterdir()) if p.is_file() and
                   (p.suffix in (".safetensors", ".bin") or p.name in ("config.json", "generation_config.json", "adapter_config.json"))]
    metadata = {
        "mode": "HF GPU semantic evaluation" if args.device == "cuda" else "HF CPU semantic evaluation",
        "platform": platform.platform(), "torch": torch.__version__, "transformers": transformers.__version__,
        "model_path": str(args.model.resolve()), "model_files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in model_files],
        "tokenizer_path": str(tok_dir.resolve()), "chat_template_sha256": __import__("hashlib").sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        "cases_sha256": sha256(args.cases), "ctx": args.ctx, "max_new_tokens": args.max_new_tokens,
        "temperature": 0, "do_sample": False, "seed": 42, "dtype": args.dtype,
        "cpu_threads": torch.get_num_threads(),
        "repetition_penalty": args.repetition_penalty,
        "tokenizer_loader": tokenizer_class.__name__,
        "tokenizer_loader_overrides": tokenizer_kwargs,
        "device": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
        "system": args.system, "model_load_s": model_load_s, "categories": args.categories or "all",
        "context_policy": "Drop whole oldest user/assistant pairs only when needed; reserve max_new_tokens within ctx",
        "performance_scope": f"{args.device.upper()}/HF latency only; never report this as native C or XP response speed",
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    with (args.out / "predictions.jsonl").open("x") as output:
        for case in cases:
            system = args.system
            if case.get("notes"):
                system += "\nNotes: " + "; ".join(case["notes"])
            history = []
            row = {"id": case["id"], "category": case["category"], "turn_results": []}
            try:
                for prompt in case["turns"]:
                    history.append({"role": "user", "content": prompt})
                    input_ids, removed = render(tokenizer, system, history, args.ctx - args.max_new_tokens)
                    if removed:
                        history = history[removed:]
                    input_ids = input_ids.to(args.device)
                    attention_mask = torch.ones_like(input_ids)
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(
                            input_ids=input_ids, attention_mask=attention_mask,
                            do_sample=False, max_new_tokens=args.max_new_tokens,
                            repetition_penalty=args.repetition_penalty,
                            use_cache=True,
                            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                    elapsed_s = time.perf_counter() - started
                    generated_ids = generated[0, input_ids.shape[1]:].tolist()
                    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    history.append({"role": "assistant", "content": answer})
                    row["turn_results"].append({"answer": answer, "tokens": len(generated_ids),
                                                "elapsed_s": elapsed_s, "ttft_visible_s": None,
                                                "prompt_tokens": input_ids.shape[1],
                                                "removed_history_messages": removed,
                                                "hit_token_cap": len(generated_ids) >= args.max_new_tokens})
                row["answer"] = row["turn_results"][-1]["answer"]
            except (RuntimeError, ValueError) as error:
                row["error"] = f"{type(error).__name__}: {error}"
                row["answer"] = ""
            row.update(score(row["answer"], case["checks"]))
            output.write(json.dumps(row) + "\n")
            output.flush()
            rows.append(row)
            print(f"{case['id']}: {'PASS' if row['objective_pass'] else 'FAIL'} {row['answer'][:120]!r}", flush=True)
    summary = summarize(rows)
    summary["performance_scope"] = metadata["performance_scope"]
    if args.device == "cuda":
        summary["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    review = []
    by_id = {r["id"]: r for r in rows}
    for case in cases:
        review.append({"id": case["id"], "category": case["category"], "notes": case.get("notes", []),
                       "turns": case["turns"], "answer": by_id[case["id"]]["answer"],
                       "reference": case["reference"], "rubric": case["rubric"],
                       "relevance_0_2": None, "correctness_0_2": None, "coherence_0_2": None,
                       "instruction_0_2": None, "review_note": ""})
    (args.out / "review.jsonl").write_text("".join(json.dumps(r) + "\n" for r in review))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
