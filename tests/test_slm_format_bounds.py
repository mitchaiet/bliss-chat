#!/usr/bin/env python3
"""Exercise rejection of malformed hybrid/Q6 headers and tensor lengths."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--export', type=Path, required=True, help='small valid Q6 LFM fixture')
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    source = (a.export / 'MODEL.SLM').read_bytes()
    if len(source) > 32 * 1048576 or struct.unpack_from('<I', source, 12)[0] != 6:
        raise ValueError('Use a small Q6 fixture')
    variants = [('truncated_header', source[:71]), ('truncated_weights', source[:1024]),
        ('truncated_final_norm', source[:-1]), ('trailing_bytes', source + b'BAD!')]
    for name, offset, value in [('unknown_bits', 12, 5), ('q6_wrong_group', 48, 32),
        ('q6_legacy_architecture', 8, 1), ('bad_hybrid_type', 60, 99),
        ('bad_conv_width', 72, 9), ('invalid_dimension', 16, 0),
        ('overflow_layers', 24, 0xffffffff), ('invalid_start_id', 76, 65536)]:
        data = bytearray(source)
        struct.pack_into('<I', data, offset, value)
        variants.append((name, data))
    data = bytearray(source)
    data[80] = 9
    variants.append(('unknown_layer_type', data))
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        model, output = Path(tmp) / 'bad.slm', Path(tmp) / 'logits.f32'
        for name, data in variants:
            model.write_bytes(data)
            result = subprocess.run([str(a.binary.resolve()), str(model), str((a.export / 'TOKENIZER.SLT').resolve()),
                '--tokens', '1', '--logits', str(output)], capture_output=True, timeout=15)
            stderr = result.stderr.decode('utf-8', 'replace')
            passed = result.returncode == 2 and '[slm]' in stderr and not any(x in stderr for x in ('AddressSanitizer', 'runtime error:'))
            results.append({'case': name, 'returncode': result.returncode, 'stderr': stderr, 'passed': passed})
    a.report.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps({'cases': len(results), 'passed': sum(x['passed'] for x in results)}))
    raise SystemExit(not all(x['passed'] for x in results))


if __name__ == '__main__':
    main()
