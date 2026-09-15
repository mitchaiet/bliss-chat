#!/usr/bin/env python3
"""Run a new candidate in an isolated directory, including a two-step smoke test.

Invoke after independently evaluating the frozen foundation. Dataset and
checkpoint selection use the adaptation split, never benchmark answers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--model-id', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--public-data', type=Path)
    p.add_argument('--prepared-data', type=Path, help='Previously prepared and checksummed data; copied into this run')
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--targets', default='q_proj,v_proj,out_proj')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--accumulation', type=int, default=2)
    p.add_argument('--learning-rate', default='2e-5')
    p.add_argument('--kl-weight', default='1.0')
    p.add_argument('--epochs', type=float, default=1.0)
    p.add_argument('--rank', type=int, default=16)
    a = p.parse_args()
    if not a.prepared_data and not a.public_data:
        p.error('Supply --public-data or --prepared-data')
    a.run.mkdir(parents=True, exist_ok=False)
    source = a.run / 'source'
    source.mkdir()
    for name in ('prepare_coherence_data.py', 'train_coherence.py', 'run_candidate.py'):
        shutil.copyfile(Path(__file__).parent / name, source / name)
    env = dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', CUDA_VISIBLE_DEVICES='0',
        PYTORCH_ALLOC_CONF='expandable_segments:True')

    def state(stage, **extra):
        value = {'stage': stage, 'time': time.time(), **extra}
        (a.run / 'state.json').write_text(json.dumps(value, indent=2) + '\n')
        print(json.dumps(value), flush=True)

    def execute(label, args):
        state(label)
        with (a.run / (label + '.log')).open('x') as log:
            subprocess.run([sys.executable, '-u', *map(str, args)], env=env,
                stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        if a.prepared_data:
            state('validate_prepared_data')
            manifest = json.loads((a.prepared_data/'manifest.json').read_text())
            assert manifest['tokenizer']['id'] == a.model_id
            assert manifest['tokenizer']['revision'] == a.revision
            for name in ('train.jsonl', 'val.jsonl'):
                actual = hashlib.sha256((a.prepared_data/name).read_bytes()).hexdigest()
                assert actual == manifest['output_sha256'][name], name + ' hash mismatch'
            shutil.copytree(a.prepared_data, a.run/'data')
        else:
            execute('prepare_data', [source/'prepare_coherence_data.py',
                '--public-data', a.public_data, '--tokenizer', a.model,
                '--tokenizer-id', a.model_id, '--tokenizer-revision', a.revision,
                '--out', a.run/'data'])
        base = [source/'train_coherence.py', '--model', a.model,
            '--train', a.run/'data/train.jsonl', '--validation', a.run/'data/val.jsonl',
            '--batch-size', a.batch_size, '--accumulation', a.accumulation,
            '--learning-rate', a.learning_rate, '--targets', a.targets,
            '--kl-weight', a.kl_weight, '--kl-examples', 4,
            '--epochs', a.epochs, '--rank', a.rank]
        execute('smoke', [*base, '--output', a.run/'smoke', '--max-steps', 2,
            '--max-train-examples', 256, '--max-validation-examples', 128, '--eval-steps', 1])
        execute('training', [*base, '--output', a.run/'training', '--eval-steps', 100])
        state('trained_awaiting_independent_evaluation')
    except Exception as error:
        state('failed', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
