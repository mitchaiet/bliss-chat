#!/usr/bin/env python3
"""Compare a real Win32 backend under isolated Wine with its Linux counterpart.

This checks executable loading and numerical/pipe behavior, not physical XP.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1048576), b''):
            h.update(data)
    return h.hexdigest()


def win(path):
    return 'Z:' + str(path.resolve()).replace('/', '\\')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('windows-binary', 'linux-binary', 'model', 'tokenizer', 'prefix', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--float-activations', action='store_true')
    a = p.parse_args()
    assert (a.prefix / 'system.reg').is_file(), 'Initialize a separate Wine prefix first'
    a.out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, WINEPREFIX=str(a.prefix.resolve()), WINEARCH='win32', DISPLAY='',
               WINEDLLOVERRIDES='mscoree,mshtml=', WINEDEBUG='-all')
    version = subprocess.run(['wine', '--version'], capture_output=True, text=True, check=True).stdout.strip()
    reg = subprocess.run(['wine', 'reg', 'add', 'HKCU\\Software\\Wine', '/v', 'Version', '/t', 'REG_SZ', '/d', 'winxp', '/f'],
        env=env, capture_output=True, text=True, timeout=60, check=True)
    (a.out / 'wine-xp-setting.txt').write_text(reg.stdout + reg.stderr)
    modes = {
        'linux': [str(a.linux_binary.resolve()), str(a.model.resolve()), str(a.tokenizer.resolve())],
        'wine': ['wine', str(a.windows_binary.resolve()), win(a.model), win(a.tokenizer)],
    }
    extra = ['--float-activations'] if a.float_activations else []
    report = {'wine_version': version, 'wine_version_setting': 'winxp', 'float_activations': a.float_activations,
        'scope': 'Win32 executable under Wine on modern Linux CPU; not physical XP execution or speed',
        'files': {key: {'path': str(getattr(a, key)), 'sha256': sha(getattr(a, key))}
                  for key in ('windows_binary', 'linux_binary', 'model', 'tokenizer')}, 'forward_cases': []}
    for index, prompt in enumerate(('The quiet garden is bright.', 'A small blue book rests beside two empty cups.',
        'Mara put the map in the desk and then moved the lamp to the shelf. The map stayed in the desk.')):
        arrays, elapsed = {}, {}
        for mode, base in modes.items():
            output = a.out / f'{mode}-{index}.f32'
            args = [*base, '-c', '512', *extra, '--raw', prompt, '--logits', win(output) if mode == 'wine' else str(output.resolve())]
            started = time.time()
            result = subprocess.run(args, env=env, capture_output=True, timeout=180, check=True)
            elapsed[mode] = time.time() - started
            (a.out / f'{mode}-{index}.stderr').write_bytes(result.stderr)
            arrays[mode] = np.fromfile(output, dtype='<f4')
        assert arrays['linux'].shape == arrays['wine'].shape
        assert np.isfinite(arrays['linux']).all() and np.isfinite(arrays['wine']).all()
        difference = float(np.max(np.abs(arrays['linux'] - arrays['wine'])))
        equal_argmax = int(arrays['linux'].argmax()) == int(arrays['wine'].argmax())
        report['forward_cases'].append({'prompt': prompt, 'max_abs_error': difference,
            'argmax_matches': equal_argmax, 'elapsed_seconds': elapsed})
        assert difference < 0.002 and equal_argmax
    frames = {}
    for mode, base in modes.items():
        notes = a.out / f'{mode}-notes.txt'
        args = [*base, '-c', '512', '-t', '0', '-n', '32', '-s', '42', *extra, '-m', win(notes) if mode == 'wine' else str(notes.resolve())]
        controls = '/info\n/remember The café ledger belongs to Zoë.\n/memories\n/reset\n'
        result = subprocess.run(args, input=controls.encode('utf-8'), env=env, capture_output=True, timeout=180, check=True)
        text = result.stdout.decode('utf-8').replace('\r\n', '\n')
        (a.out / f'{mode}-ipc.txt').write_text(text)
        assert '\x01READY\n' in text and text.count('\x01EOT ') == 4 and '\x01ERR' not in text
        assert 'café' in text and 'Zoë' in text
        assert 'café' in notes.read_text() and 'Zoë' in notes.read_text()
        frames[mode] = text
    assert frames['linux'] == frames['wine']
    report['ipc'] = {'matching_frames': True, 'controls_per_platform': 4, 'unicode_note_persistence': True}
    report['passed'] = True
    (a.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
