#!/usr/bin/env python3
"""Compose an explicit public/procedural/calibration mixture without eval inputs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fingerprint(row):
    return hashlib.sha256(json.dumps(row['messages'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--procedural-count', type=int, default=4096)
    p.add_argument('--calibration-repeats', type=int, default=4)
    p.add_argument('--seed', type=int, default=20260915)
    a = p.parse_args()
    assert a.procedural_count > 0 and a.calibration_repeats > 0
    manifest = json.loads((a.base / 'manifest.json').read_text())
    for name in ('train.jsonl', 'val.jsonl'):
        assert sha(a.base / name) == manifest['output_sha256'][name]
    base_train, base_val = read(a.base / 'train.jsonl'), read(a.base / 'val.jsonl')
    cal_train, cal_val = read(a.calibration / 'train.jsonl'), read(a.calibration / 'validation.jsonl')
    assert all(row.get('retention') is False for row in cal_train + cal_val)
    assert set(row['family'] for row in cal_train).isdisjoint(row['family'] for row in cal_val)
    assert len({fingerprint(r) for r in base_train + cal_train}) == len(base_train + cal_train)
    assert len({fingerprint(r) for r in base_val + cal_val}) == len(base_val + cal_val)
    assert {fingerprint(r) for r in base_train + cal_train}.isdisjoint(fingerprint(r) for r in base_val + cal_val)
    public, groups = [], defaultdict(list)
    for row in base_train:
        if row['source'] == 'bliss-procedural-v1':
            groups[row['category']].append(dict(row, retention=False))
        else:
            public.append(dict(row, retention=True))
    rng = random.Random(a.seed)
    for rows in groups.values():
        rng.shuffle(rows)
    procedural = []
    while len(procedural) < a.procedural_count:
        added = False
        for key in sorted(groups):
            if groups[key] and len(procedural) < a.procedural_count:
                procedural.append(groups[key].pop())
                added = True
        if not added:
            raise ValueError('Requested more procedural rows than available')
    weighted_cal = [dict(row, repeat_index=repeat) for repeat in range(a.calibration_repeats) for row in cal_train]
    train = public + procedural + weighted_cal
    val = [dict(row, retention=row['source'] != 'bliss-procedural-v1') for row in base_val] + cal_val
    rng.shuffle(train)
    rng.shuffle(val)
    a.out.mkdir(parents=True, exist_ok=False)
    for name, rows in (('train.jsonl', train), ('val.jsonl', val)):
        (a.out / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    result = {
        'recipe': 'bliss-calibrated-mixture-v1', 'seed': a.seed,
        'tokenizer': manifest['tokenizer'],
        'inputs': {str(path): sha(path) for path in (a.base / 'train.jsonl', a.base / 'val.jsonl', a.calibration / 'train.jsonl', a.calibration / 'validation.jsonl')},
        'public_retention_rows': len(public), 'procedural_rows': len(procedural),
        'unique_calibration_train': len(cal_train), 'calibration_repeats': a.calibration_repeats,
        'train_rows': len(train), 'unique_train_rows': len(public) + len(procedural) + len(cal_train),
        'validation_rows': len(val), 'calibration_validation_rows': len(cal_val),
        'training_source_counts': dict(Counter(r['source'] for r in train)),
        'training_procedural_categories': dict(Counter(r['category'] for r in procedural)),
        'selection': 'No evaluation files consumed. Explicit calibration repetitions set before training; validation is unique and not repeated.',
        'output_sha256': {name: sha(a.out / name) for name in ('train.jsonl', 'val.jsonl')},
        'script_sha256': sha(Path(__file__)),
    }
    (a.out / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
