#!/usr/bin/env python3
"""Batched prefill must leave exactly the state a token-by-token prefill leaves.

Runs the same prompt through the backend at several batch sizes and compares the
dumped logits byte for byte. Any difference in accumulation order, in the
convolution's carried state, or in a token's attention window shows up here.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROMPTS = [
    # Shorter than one batch, so it exercises the single-token fallback.
    'Name three planets.',
    # Crosses a batch boundary unevenly, leaving a short final chunk.
    'The Starlite Drive-In opened in Brenham, Texas, and showed films for decades '
    'before the screen came down. Explain what happened and why it mattered.',
    # Long enough to span many full batches and exercise context growth.
    ' '.join(['projector reels flickered over the gravel lot while families waited'] * 25),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--batches', default='1,2,3,8,16')
    a = p.parse_args()
    batches = [int(b) for b in a.batches.split(',')]
    failures = []
    with tempfile.TemporaryDirectory() as directory:
        for index, prompt in enumerate(PROMPTS):
            digests = {}
            for batch in batches:
                out = Path(directory) / f'logits-{index}-{batch}.f32'
                env = dict(os.environ, SLM_BATCH=str(batch))
                run = subprocess.run(
                    [str(a.binary), str(a.model), str(a.tokenizer),
                     '--raw', prompt, '--logits', str(out)],
                    env=env, capture_output=True, text=True)
                if run.returncode or not out.exists():
                    failures.append(f'prompt {index} batch {batch}: {run.stderr.strip()[:200]}')
                    continue
                digests[batch] = digest(out)
            reference = digests.get(batches[0])
            for batch, value in digests.items():
                status = 'identical' if value == reference else 'DIFFERS'
                print(f'  prompt {index} batch {batch:2d}: {status}')
                if value != reference:
                    failures.append(f'prompt {index} batch {batch} logits differ')
    if failures:
        print('\nFAIL')
        for f in failures:
            print(' ', f)
        return 1
    print(f'\nAll {len(PROMPTS)} prompts match across batch sizes {batches}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
