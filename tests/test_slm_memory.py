#!/usr/bin/env python3
"""Fill a native model's context under a Linux virtual-memory limit; not XP proof."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import resource
import struct
import subprocess
import sys
import tempfile
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--context', type=int, default=512)
    p.add_argument('--limit-mib', type=int, default=384)
    p.add_argument('--timeout', type=float, default=300)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--runtime-arg', action='append', default=[])
    a = p.parse_args()
    if sys.platform != 'linux':
        raise SystemExit('This bounded-memory test requires Linux; it does not run an XP emulator.')
    with a.model.open('rb') as f:
        h = f.read(256)
    magic, version, bits, d, ff, layers, heads, kv, hd, vocab, capacity, group, bos, eos, arch, theta, eps = struct.unpack('<8s14I2f', h[:72])
    if magic != b'SLMODEL1' or not 32 <= a.context <= capacity:
        raise ValueError('Invalid model/context')
    limit = a.limit_mib * 1048576
    def constrain():
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / 'logits.f32'
        cmd = [str(a.binary.resolve()), str(a.model.resolve()), str(a.tokenizer.resolve()),
            '-c', str(a.context), '--tokens', ','.join([str(bos)] * a.context), '--logits', str(output)] + a.runtime_arg
        start = time.perf_counter()
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=constrain, timeout=a.timeout)
        elapsed = time.perf_counter() - start
        values = array.array('f')
        if output.exists():
            values.frombytes(output.read_bytes())
        finite = len(values) == vocab and all(map(math.isfinite, values))
    report = {'platform': sys.platform, 'scope': 'Linux native process under RLIMIT_AS; no physical Windows XP claim',
        'binary': str(a.binary), 'binary_sha256': digest(a.binary), 'model_sha256': digest(a.model),
        'runtime_args': a.runtime_arg, 'context_filled': a.context, 'virtual_memory_limit_mib': a.limit_mib,
        'peak_child_rss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        'elapsed_seconds': elapsed, 'returncode': result.returncode,
        'complete_finite_logits': finite, 'stderr': result.stderr.decode('utf-8', 'replace'),
        'passed': result.returncode == 0 and finite}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(not report['passed'])


if __name__ == '__main__':
    main()
