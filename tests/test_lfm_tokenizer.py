#!/usr/bin/env python3
"""HF parity, bounds, and malformed-file checks; builds only the C tokenizer.

Requires a host clang-compatible compiler and Python tokenizers. No model
weights, GPU, benchmark cases, or network access are used by this test.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shlex
import struct
import subprocess
import tempfile
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]


def load_exporter(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.export_tokenizer


def base_fixtures():
    # Same 169 tokenizer-only fixtures as test_slm_parity.py, independent of
    # any chat-quality evaluation or checkpoint predictions.
    cases = ['', 'Hello, world!', 'What is 12345 + 67890?', ' 42', '12.34 1,234 -123',
             "don't can't I'M I'm WE'RE we'll", '  hello   world  ', 'x\n\ny\r\n\t z', '\n   ',
             'café naïve résumé', '中文 العربية русский', '³²½Ⅷ１２٣٤', '👩🏽‍💻🌍✨',
             'a\u00a0b\u202fc\u2003d', 'a\u0301 e\u0308',
             '<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n',
             '<repo_name>abc<filename>test.py', 'a\t \r\n   b', 'a\x1cb']
    rng = random.Random(41208)
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t\r!?,.-'é漢٣²\u00a0"
    cases += [''.join(rng.choice(alphabet) for _ in range(rng.randrange(1, 160))) for _ in range(150)]
    assert len(cases) == 169
    return cases


def lfm_fixtures(added):
    cases = ['python pythonic apython Python', 'Mathias aMathiasb mathias',
             '<think>x</think>', '<|tool_call_start|>hi<|tool_call_end|>',
             "I'ſ I'S he's HE'S we're WE'RE", '\u2028hello', '\u0085a',
             'x\x1cy\x1dz\x1e\x1f', '<|startoftext|><|im_start|>assistant\nHello<|im_end|>\n']
    for prefix in ('', ' ', 'x', '...', '\n'):
        cases.extend(prefix + ('1234567890123456789'[:n]) + 'z' for n in range(1, 20))
    for suffix in ('s', 't', 're', 've', 'm', 'll', 'd'):
        cases.extend(prefix + "'" + word for prefix in ('', 'word') for word in (suffix, suffix.upper()))
    spaces = (' ', '\t', '\n', '\r', '\r\n', '\u00a0', '\u2028', '  ')
    cases.extend('a' + left + right + 'z' for left in spaces for right in spaces)
    cases.extend('prefix' + item['content'] + 'suffix' for item in added)
    cases += [chr(cp) + 'a' + chr(cp) + '123' for cp in
              (0x017f, 0x0345, 0x05d0, 0x0661, 0x2167, 0x11f02, 0x11f50, 0x1e4d0, 0x1e4f0, 0x31350, 0x1f9e0)]
    return cases


def commands(path, requests):
    with path.open('wb') as stream:
        for op, parameter, capacity, text in requests:
            raw = text.encode('utf-8')
            stream.write(struct.pack('<IiiI', op, parameter, capacity, len(raw)))
            stream.write(raw)


def run_driver(binary, tokenizer, request_path):
    result = subprocess.run([str(binary), str(tokenizer), str(request_path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return [json.loads(line) for line in result.stdout.splitlines()]


def parity(binary, folder, exported, temporary, version):
    data = json.loads((folder / 'tokenizer.json').read_text())
    hf = Tokenizer.from_file(str(folder / 'tokenizer.json'))
    fixtures = base_fixtures() + (lfm_fixtures(data['added_tokens']) if version == 2 else [])
    requests, expected = [], []
    for specials in (0, 1):
        hf.encode_special_tokens = not bool(specials)
        for text in fixtures:
            requests.append((0, specials, 8192, text))
            expected.append(hf.encode(text, add_special_tokens=False).ids)
    request_path = temporary / ('requests-v%d.bin' % version)
    commands(request_path, requests)
    observed = run_driver(binary, exported, request_path)
    vocab = observed[0]['vocab']
    assert len(observed) == len(expected) + 1
    used = set()
    for request, wanted, got in zip(requests, expected, observed[1:]):
        assert got['ids'] == wanted and got['count'] == len(wanted), (version, request, wanted, got)
        used.update(wanted)
    added = {item['id']: item for item in data['added_tokens']}
    padded_start = max(max(data['model']['vocab'].values()), max(added)) + 1
    inspect_ids = sorted(used | set(added) | set(range(padded_start, vocab)))
    commands(request_path, [(1, token, 0, '') for token in inspect_ids])
    meta = dict(zip(inspect_ids, run_driver(binary, exported, request_path)[1:]))
    for token, entry in meta.items():
        assert entry['valid'] and entry['special'] in (0, 1)
        assert entry['length'] == len(bytes.fromhex(entry['hex']))
        if token in added:
            assert entry['special'] == int(added[token]['special'])
            assert bytes.fromhex(entry['hex']) == added[token]['content'].encode()
        elif token >= padded_start:
            assert entry['length'] == 0 and entry['special'] == 1
    for ids in expected:
        decoded = b''.join(bytes.fromhex(meta[token]['hex']) for token in ids if not meta[token]['special']).decode('utf-8', 'replace')
        assert decoded == hf.decode(ids, skip_special_tokens=True), (version, ids, decoded)
    commands(request_path, [(0, 1, 0, ''), (0, 1, 0, 'abc'), (0, 1, 1, 'a b c d e'),
                            (0, 1, -1, 'abc'), (1, -1, 0, ''), (1, vocab, 0, '')])
    bounds = run_driver(binary, exported, request_path)[1:]
    assert [row['count'] for row in bounds[:4]] == [0, -1, -1, -1]
    assert not bounds[4]['valid'] and not bounds[5]['valid']
    return {'format_version': version, 'vocab': vocab, 'fixtures_per_special_mode': len(fixtures),
            'exact_id_cases': len(expected), 'decode_cases': len(expected),
            'token_metadata_checked': len(meta), 'padded_metadata_checked': vocab - padded_start,
            'bounds_checks': len(bounds), 'tokenizer_sha256': hashlib.sha256(exported.read_bytes()).hexdigest()}


def malformed(binary, tokenizer, temporary):
    blob = tokenizer.read_bytes()
    _, version, vocab, merges, ranges = struct.unpack_from('<8s4I', blob)
    assert version == 2 and ranges > 1 and merges > 1
    offset = 24
    for _ in range(vocab):
        size, _ = struct.unpack_from('<2I', blob, offset)
        offset += 8 + size
    def changed(offset, value):
        data = bytearray(blob)
        struct.pack_into('<I', data, offset, value)
        return data
    bad_pair = bytearray(blob)
    bad_pair[offset + 12:offset + 24] = bad_pair[offset:offset + 12]
    range_start = offset + merges * 12
    corruptions = [blob[:10], blob[:-1], blob + b'x', changed(8, 3), changed(28, 4),
                   changed(offset, vocab), bad_pair, changed(range_start + 4, 0xffffffff),
                   changed(range_start + 12, 0)]
    requests = temporary / 'empty-requests.bin'
    requests.write_bytes(b'')
    for i, data in enumerate(corruptions):
        path = temporary / ('malformed-%d.slt' % i)
        path.write_bytes(data)
        result = subprocess.run([str(binary), str(path), str(requests)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 2, (i, result.returncode, result.stdout, result.stderr)
    return len(corruptions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lfm-dir', type=Path, required=True)
    parser.add_argument('--smol-dir', type=Path)
    parser.add_argument('--smol-tokenizer', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--no-sanitize', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='bliss-tokenizer-parity-') as name:
        temporary = Path(name)
        binary = temporary / 'driver'
        command = shlex.split(os.environ.get('CC', 'clang')) + ['-std=c99', '-Wall', '-Wextra', '-Werror', '-O1', '-g']
        if not args.no_sanitize:
            command += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        command += ['-I' + str(ROOT / 'src'), str(ROOT / 'src/slm_tokenizer.c'),
                    str(ROOT / 'tests/slm_tokenizer_driver.c'), '-o', str(binary)]
        subprocess.run(command, check=True, timeout=60)
        lfm = temporary / 'lfm.slt'
        export_info = load_exporter('export_lfm_tokenizer')(args.lfm_dir, lfm, model_vocab=65536)
        report = {'export': export_info, 'sanitizers': not args.no_sanitize,
                  'lfm': parity(binary, args.lfm_dir, lfm, temporary, 2),
                  'malformed_files_rejected': malformed(binary, lfm, temporary)}
        if args.smol_dir:
            smol = args.smol_tokenizer or temporary / 'smol.slt'
            if not args.smol_tokenizer:
                load_exporter('export_slm')(args.smol_dir, smol)
            report['smol_v1_regression'] = parity(binary, args.smol_dir, smol, temporary, 1)
        report['source_sha256'] = hashlib.sha256((ROOT / 'src/slm_tokenizer.c').read_bytes()).hexdigest()
        report['weights_loaded'] = False
        if args.report:
            with args.report.open('x') as stream:
                stream.write(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
