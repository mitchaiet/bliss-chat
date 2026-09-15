#!/usr/bin/env python3
"""Build a checksummed XP folder and ZIP from explicitly selected artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime', 'gui', 'model', 'tokenizer', 'model-card', 'license', 'out'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--name', required=True)
    p.add_argument('--readme', type=Path, help='Release-specific offline user guide to include as README.TXT')
    p.add_argument('--notice', type=Path, help='Publisher attribution and modification notice to include')
    p.add_argument('--selection', type=Path, help='Frozen selection JSON; checks exact chosen model and executables')
    p.add_argument('--data-license', type=Path, help='License for the public adaptation data')
    p.add_argument('--version', default='2.0.0-candidate.20260915')
    a = p.parse_args()
    archive = a.out.parent / (a.out.name + '.zip')
    if a.out.exists() or archive.exists():
        raise RuntimeError('Choose a new output path; existing packages are preserved.')
    for source in (a.runtime, a.gui, a.model, a.tokenizer, a.model_card, a.license):
        if not source.is_file():
            raise FileNotFoundError(source)
    for binary in (a.runtime, a.gui):
        with binary.open('rb') as f:
            if f.read(2) != b'MZ':
                raise ValueError('Expected a Windows executable: ' + str(binary))
    if a.readme and not a.readme.is_file():
        raise FileNotFoundError(a.readme)
    if a.notice and not a.notice.is_file():
        raise FileNotFoundError(a.notice)
    if a.data_license and not a.data_license.is_file():
        raise FileNotFoundError(a.data_license)
    selection = json.loads(a.selection.read_text()) if a.selection else None
    if selection:
        for path, key in ((a.runtime, 'windows_binary_sha256'), (a.gui, 'gui_sha256'),
                          (a.model, 'model_sha256'), (a.tokenizer, 'tokenizer_sha256')):
            if digest(path) != selection[key]:
                raise ValueError('Artifact differs from frozen selection: ' + str(path))
    a.out.mkdir(parents=True)
    payload = {'XPCHAT.EXE': a.gui, 'NC_RUN.EXE': a.runtime,
        'NC_RUN_SSE2.EXE': a.runtime, 'NC_RUN_SSE3.EXE': a.runtime,
        'MODEL.NCB': a.model, 'TOKENIZER.NCT': a.tokenizer,
        'MODEL_CARD.md': a.model_card, 'MODEL-LICENSE.txt': a.license}
    for destination, source in payload.items():
        shutil.copyfile(source, a.out / destination)
    if a.notice:
        shutil.copyfile(a.notice, a.out / 'NOTICE.txt')
    if a.data_license:
        shutil.copyfile(a.data_license, a.out / 'DATA-LICENSE.txt')
    (a.out / 'MODEL_VERSION.txt').write_text(a.name + '\n', encoding='ascii')
    (a.out / 'README.TXT').write_text(
        'Bliss Chat ' + a.version + '\n\n'
        'Extract the entire ZIP into a new folder, for example C:\\Bliss2.\n'
        'Double-click XPCHAT.EXE. Keep the files together in that folder.\n'
        'The model runs locally; no account or network connection is needed.\n\n'
        'Target: Windows XP SP2/SP3, Pentium M with SSE2, 512 MB RAM.\n'
        'Start with 512 context tokens and short answers. Close other large\n'
        'applications to leave memory for the model. The packaged backend\n'
        'variants all use the same SSE2-compatible executable.\n\n'
        'This is a candidate build. Native inference and executable imports\n'
        'were tested on development machines; physical XP startup, memory,\n'
        'and response speed have not yet been verified. See MODEL_CARD.md\n'
        'for evaluation results, model provenance, and known limitations.\n', encoding='ascii')
    if a.readme:
        shutil.copyfile(a.readme, a.out / 'README.TXT')
    manifest = {'version': a.version, 'name': a.name,
        'selection_sha256': digest(a.selection) if a.selection else None,
        'hardware_status': 'physical XP validation pending',
        'runtime_variants': 'identical SSE2 binaries',
        'files': {f.name: {'bytes': f.stat().st_size, 'sha256': digest(f)}
                  for f in sorted(a.out.iterdir()) if f.is_file()}}
    (a.out / 'release-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(a.out.iterdir()):
            z.write(f, a.out.name + '/' + f.name)
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError('ZIP verification failed: ' + bad)
    archive.with_suffix('.zip.sha256').write_text(digest(archive) + '  ' + archive.name + '\n')
    print(json.dumps({'folder': str(a.out.resolve()), 'zip': str(archive.resolve()),
                     'bytes': archive.stat().st_size, 'sha256': digest(archive)}, indent=2))


if __name__ == '__main__':
    main()
