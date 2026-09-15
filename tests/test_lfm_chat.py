#!/usr/bin/env python3
"""Compare native plain-chat prefixes with the publisher's unmodified template."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hf', type=Path, required=True)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=str(a.hf / 'tokenizer.json'),
        bos_token='<|startoftext|>', eos_token='<|im_end|>', pad_token='<|pad|>',
        chat_template=(a.hf / 'chat_template.jinja').read_text())
    cases = [('', 'Hello', None), ('Answer briefly.', 'What is 123456?', None),
        ('You are Bliss.', 'And in café中文?', ('Hello!', 'Hi.'))]
    results = []
    for system, user, prior in cases:
        messages = [{'role': 'system', 'content': system}]
        if prior:
            messages += [{'role': 'user', 'content': prior[0]}, {'role': 'assistant', 'content': prior[1]}]
        messages += [{'role': 'user', 'content': user}]
        expected = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        cmd = [str(a.probe), str(a.export / 'MODEL.SLM'), str(a.export / 'TOKENIZER.SLT'), system, user]
        if prior:
            cmd += list(prior)
        got = json.loads(subprocess.check_output(cmd, text=True))
        results.append({'messages': messages, 'expected': expected, 'actual': got, 'passed': got == expected})
    a.report.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'cases': len(results), 'passed': sum(r['passed'] for r in results)}))
    raise SystemExit(not all(r['passed'] for r in results))


if __name__ == '__main__':
    main()
