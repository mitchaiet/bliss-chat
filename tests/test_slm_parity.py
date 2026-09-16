#!/usr/bin/env python3
"""Independent HF/tokenizers parity for native FP32 and Q4/Q6/Q8 runtime.

CPU only. The quantized comparison restores the actual exported tensors into
HF and adds group64 activation quantization at Linear inputs, matching the
documented native arithmetic rather than comparing unrelated precisions.
"""
import argparse
import json
from pathlib import Path
import random
import struct
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf", type=Path, required=True)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--tokenizer-only", action="store_true")
    p.add_argument("--float-activations", action="store_true")
    p.add_argument("--skip-tokenizer", action="store_true", help="numeric-token forward checks while tokenizer port is independent")
    a = p.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(a.hf / "tokenizer.json"))
    cases = ["", "Hello, world!", "What is 12345 + 67890?", " 42", "12.34 1,234 -123", "don't can't I'M I'm WE'RE we'll", "  hello   world  ", "x\n\ny\r\n\t z", "\n   ", "café naïve résumé", "中文 العربية русский", "³²½Ⅷ１２٣٤", "👩🏽‍💻🌍✨", "a\u00a0b\u202fc\u2003d", "a\u0301 e\u0308", "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n", "<repo_name>abc<filename>test.py", "a\t \r\n   b", "a\x1cb"]
    rng = random.Random(41208)
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t\r!?,.-'é漢٣²\u00a0"
    cases += ["".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 160))) for _ in range(150)]
    failures = []
    for text in ([] if a.skip_tokenizer else cases):
        expected = tok.encode(text, add_special_tokens=False).ids
        got = json.loads(subprocess.check_output([str(a.binary), str(a.export / "MODEL.SLM"), str(a.export / "TOKENIZER.SLT"), "--tokenize", text, "--specials"], text=True))
        if expected != got:
            failures.append({"text": text, "expected": expected, "actual": got})
    result = {"tokenizer_cases": 0 if a.skip_tokenizer else len(cases), "tokenizer_failures": failures, "logit_checks": []}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    if failures or a.tokenizer_only:
        a.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"tokenizer_cases": len(cases), "failures": len(failures)}))
        raise SystemExit(bool(failures))
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(a.hf, torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True).eval()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from slm_to_hf import restore_slm
    meta = restore_slm(model, a.export / "MODEL.SLM")
    bits, group = meta['bits'], meta['group']
    if bits != 32 and not a.float_activations:
        def qactivation(module, args):
            x = args[0]
            shaped = x.reshape(*x.shape[:-1], -1, group)
            bound = 32767 if bits == 6 else 127
            scale = shaped.abs().amax(dim=-1, keepdim=True) / bound
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            if bits == 6:
                scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
                values = shaped * scale.reciprocal()
            else:
                values = shaped / scale
            return (values.round().clamp(-bound, bound).mul(scale).reshape_as(x),)
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.register_forward_pre_hook(qactivation)
    probes = ["Hello", "<|im_start|>user\nWhat is gravity?<|im_end|>\n<|im_start|>assistant\n", "<|im_start|>system\nAnswer briefly.<|im_end|>\n<|im_start|>user\nMy dog is named Pepper. Remember that.<|im_end|>\n<|im_start|>assistant\nYour dog is Pepper.<|im_end|>\n<|im_start|>user\nWhat is my dog's name?<|im_end|>\n<|im_start|>assistant\n"]
    with tempfile.TemporaryDirectory() as directory:
        for index, text in enumerate(probes):
            ids = tok.encode(text, add_special_tokens=False).ids
            target = Path(directory) / "logits.f32"
            subprocess.run([str(a.binary), str(a.export / "MODEL.SLM"), str(a.export / "TOKENIZER.SLT"), "--tokens", ",".join(map(str, ids)), "--logits", str(target)] + (["--float-activations"] if a.float_activations else []), check=True)
            with torch.no_grad():
                if bits != 32:
                    # LFM4.57's slow conv cache clamps absolute positions into
                    # the three-slot cache. Prefill >=3 tokens avoids its
                    # one-token-prefill edge case at position1; ordinary chats
                    # already prefill longer prompts. Native also matches the
                    # independent full-sequence reference in FP32 above.
                    prefill = min(3, len(ids)) if meta['architecture'] == 'lfm2' else 1
                    output = model(torch.tensor([ids[:prefill]]), use_cache=True)
                    cache = output.past_key_values
                    for token in ids[prefill:]:
                        output = model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
                        cache = output.past_key_values
                    expected = output.logits[0, -1].float().numpy()
                else:
                    expected = model(torch.tensor([ids]), use_cache=False).logits[0, -1].float().numpy()
            got = np.fromfile(target, dtype="<f4")
            delta = np.abs(got - expected)
            check = {"probe": index, "tokens": len(ids), "bits": bits, "max_abs": float(delta.max()), "mean_abs": float(delta.mean()), "cosine": float(np.dot(got, expected) / (np.linalg.norm(got) * np.linalg.norm(expected))), "argmax_native": int(got.argmax()), "argmax_hf": int(expected.argmax())}
            # Native FP32 reductions and SSE2 quantization rounding differ slightly.
            check["passed"] = check["cosine"] > 0.999 and check["mean_abs"] < (0.03 if bits != 32 else 0.001) and check["argmax_native"] == check["argmax_hf"]
            result["logit_checks"].append(check)
            print(json.dumps(check), flush=True)
    a.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    raise SystemExit(not all(x["passed"] for x in result["logit_checks"]))


if __name__ == "__main__":
    main()
