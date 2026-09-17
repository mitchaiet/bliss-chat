#!/usr/bin/env python3
"""Disposable prefix-cache integrity and invalidation checks on real weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--alternate-binary', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    a.binary = a.binary.resolve()
    checks, timings = {}, {}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cache = root / 'PREFIX.CACHE'
        def run(extra=(), binary=None, model=None, argv0=None):
            actual = str(binary or a.binary)
            args = [argv0 or actual, str(model or a.model), str(a.tokenizer),
                    '-n', '12', '--prefix-cache', str(cache), '--prompt', 'hi', *extra]
            start = time.monotonic()
            result = subprocess.run(args, executable=actual, capture_output=True, timeout=120)
            assert result.returncode == 0, (args, result.stderr)
            return result, time.monotonic() - start
        cold, timings['cold_seconds'] = run()
        initial = cache.read_bytes()
        warm, timings['warm_seconds'] = run()
        checks['cold_then_warm_identical_reply'] = cold.stdout == warm.stdout and b'prefix cache hit' in warm.stderr
        checks['full_model_and_runtime_hashes_match_hashlib'] = (
            initial[32:64] == hashlib.file_digest(a.model.open('rb'), 'sha256').digest() and
            initial[64:96] == hashlib.file_digest(a.binary.open('rb'), 'sha256').digest())
        checks['payload_sha256_matches_hashlib'] = initial[96:128] == hashlib.sha256(initial[128:]).digest()
        bad_payload = bytearray(initial); bad_payload[-1] ^= 1
        bad_size = bytearray(initial); bad_size[12:16] = b'\xff' * 4
        bad_runtime = bytearray(initial); bad_runtime[64] ^= 1
        bad_model = bytearray(initial); bad_model[32] ^= 1
        bad_ids = bytearray(initial); bad_ids[128] ^= 1
        bad_ids[96:128] = hashlib.sha256(bad_ids[128:]).digest()
        for label, content in {
            'corrupt_payload': bad_payload, 'untrusted_length': bad_size,
            'wrong_runtime_identity': bad_runtime, 'wrong_model_identity': bad_model,
            'different_ids_with_valid_checksum': bad_ids,
            'truncated_header': initial[:127], 'truncated_payload': initial[:-1],
            'trailing_bytes': initial + b'x',
        }.items():
            cache.write_bytes(content)
            result, _ = run()
            checks[label] = result.stdout == cold.stdout and b'prefix cache hit' not in result.stderr and cache.read_bytes() == initial
        for label, extra in {
            'changed_prefix': ['--system', 'Answer briefly.'],
            'changed_context': ['-c', '128'],
            'changed_activation_mode': ['--float-activations'],
        }.items():
            cache.write_bytes(initial)
            changed, _ = run(extra)
            repeated, _ = run(extra)
            checks[label] = b'prefix cache hit' not in changed.stderr and b'prefix cache hit' in repeated.stderr and changed.stdout == repeated.stdout
        cache.write_bytes(initial)
        changed, _ = run(binary=a.alternate_binary.resolve())
        checks['different_executable_recomputes'] = b'prefix cache hit' not in changed.stderr and cache.read_bytes()[64:96] != initial[64:96]
        changed_model = root / 'changed-model.slm'
        shutil.copyfile(a.model, changed_model)
        with changed_model.open('r+b') as f:
            f.seek(240); old = f.read(1); f.seek(240); f.write(bytes([old[0] ^ 1]))
        cache.write_bytes(initial)
        changed, _ = run(model=changed_model)
        checks['changed_model_bytes_recompute'] = b'prefix cache hit' not in changed.stderr and changed.stdout == cold.stdout and cache.read_bytes()[32:64] != initial[32:64]
        cache.write_bytes(initial)
        disabled, _ = run(argv0='/nonexistent-bliss-executable')
        checks['unreadable_executable_disables_cache'] = b'prefix cache' not in disabled.stderr and disabled.stdout == cold.stdout
        cache = root / 'missing-directory' / 'PREFIX.CACHE'
        unavailable, _ = run()
        checks['unwritable_cache_falls_back'] = unavailable.stdout == cold.stdout and not cache.exists()
        checks['atomic_write_leaves_no_temporary_files'] = not list(root.glob('*.tmp.*'))
        report = {'checks': checks, 'timings_native_host': timings,
                  'cache_bytes': len(initial), 'prefix_tokens': struct.unpack_from('<I', initial, 12)[0]}
    a.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert all(checks.values())


if __name__ == '__main__':
    main()
