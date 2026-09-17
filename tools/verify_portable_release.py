#!/usr/bin/env python3
"""Verify an independently extracted portable EXE against its source package."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('exe', 'payload', 'extracted', 'selection', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    expected = {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size}
                for p in args.payload.iterdir() if p.is_file()}
    observed = {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size}
                for p in args.extracted.iterdir() if p.is_file()}
    assert expected == observed, 'Extracted payload differs from the verified folder'
    selection = json.loads(args.selection.read_text())
    for name, key in [('MODEL.NCB', 'model_sha256'), ('TOKENIZER.NCT', 'tokenizer_sha256'),
                      ('XPCHAT.EXE', 'gui_sha256'), ('NC_RUN.EXE', 'windows_binary_sha256'),
                      ('NC_RUN_SSE2.EXE', 'windows_binary_sha256'), ('NC_RUN_SSE3.EXE', 'windows_binary_sha256')]:
        assert expected[name]['sha256'] == selection[key], name
    header = subprocess.check_output(['i686-w64-mingw32-objdump', '-p', str(args.exe)], text=True)
    dlls = re.findall(r'DLL Name:\s+(\S+)', header)
    major = int(re.search(r'MajorSubsystemVersion\s+(\d+)', header).group(1))
    minor = int(re.search(r'MinorSubsystemVersion\s+(\d+)', header).group(1))
    assert re.search(r'Magic\s+010b\s+\(PE32\)', header)
    assert (major, minor) <= (5, 1)
    assert not any(re.search(r'api-ms-win|ucrt|vcruntime|libgcc|libwinpthread', d, re.I) for d in dlls)
    report = {
        'release': 'v' + selection.get('version', '2.0.0-rc.1'),
        'portable_exe': {'name': args.exe.name, 'bytes': args.exe.stat().st_size, 'sha256': digest(args.exe)},
        'independent_extraction': '7-Zip NSIS decoder',
        'all_extracted_payload_files_match': True,
        'frozen_model_and_binary_hashes_match': True,
        'payload_files': expected,
        'wrapper': {'pe32': True, 'subsystem': f'{major}.{minor}', 'dlls': dlls, 'modern_crt_imports': []},
        'limitations': ['Physical Windows XP remains untested.',
                       'Payload equality verifies packaging integrity, not runtime execution.',
                       'Independent extraction does not prove the wrapper GUI runs on physical XP.',
                       'The EXE is unsigned.'],
    }
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'payload_files'}, indent=2))

if __name__ == '__main__':
    main()
