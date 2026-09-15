#!/usr/bin/env python3
"""Record PE/CRT/ISA evidence; this does not replace physical XP execution."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('binaries', type=Path, nargs='+')
    p.add_argument('--objdump', default='i686-w64-mingw32-objdump')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    forbidden = set(('addsubpd addsubps fisttp haddpd haddps hsubpd hsubps lddqu '
        'movddup movshdup movsldup monitor mwait pabsb pabsw pabsd palignr '
        'phaddw phaddd phaddsw phsubw phsubd phsubsw pmaddubsw pmulhrsw '
        'pshufb psignb psignw psignd blendpd blendps blendvpd blendvps '
        'dppd dpps extractps insertps movntdqa mpsadbw packusdw pblendvb '
        'pblendw pcmpestri pcmpestrm pcmpistri pcmpistrm pcmpgtq pcmpeqq '
        'pextrb pextrd pextrq phminposuw pinsrb pinsrd pinsrq pmaxsb pmaxsd '
        'pmaxud pmaxuw pminsb pminsd pminud pminuw pmuldq pmulld ptest '
        'roundpd roundps roundsd roundss crc32 popcnt lzcnt '
        'rdrand rdseed rdtscp xgetbv xsave xsaveopt xrstor').split())
    rows = []
    for binary in a.binaries:
        headers = subprocess.check_output([a.objdump, '-p', str(binary)], text=True)
        asm = subprocess.check_output([a.objdump, '-d', str(binary)], text=True)
        mnemonics = set()
        for line in asm.splitlines():
            fields = line.split('\t')
            if len(fields) >= 3 and re.match(r'^\s*[0-9a-f]+:', fields[0]):
                op = fields[2].strip().split()
                if op:
                    mnemonics.add(op[0])
        suspect = sorted(op for op in mnemonics if op in forbidden or
            (op.startswith('v') and op not in {'verr', 'verw'}))
        dlls = re.findall(r'DLL Name:\s+(\S+)', headers)
        bad_dlls = [d for d in dlls if re.search(r'api-ms-win|ucrt|vcruntime|libgcc|libwinpthread', d, re.I)]
        major = int(re.search(r'MajorSubsystemVersion\s+(\d+)', headers).group(1))
        minor = int(re.search(r'MinorSubsystemVersion\s+(\d+)', headers).group(1))
        rows.append({'file': binary.name, 'bytes': binary.stat().st_size,
            'sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
            'pe32': bool(re.search(r'Magic\s+010b\s+\(PE32\)', headers)),
            'subsystem_version': f'{major}.{minor}', 'imported_dlls': dlls,
            'modern_crt_imports': bad_dlls, 'post_sse2_instruction_matches': suspect,
            'legacy_alias_instructions': sorted(mnemonics & {'tzcnt'}),
            'legacy_alias_note': 'TZCNT encoding executes BSF on pre-BMI CPUs. MinGW numeric conversion helpers use this encoding for nonzero bit scans; zero/flag assumptions require source or disassembly review.',
            'limits': 'Static evidence only. DLL presence does not verify every API or physical XP startup.'})
    a.out.write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows, indent=2))
    if any(not r['pe32'] or r['subsystem_version'] != '5.1' or
           r['modern_crt_imports'] or r['post_sse2_instruction_matches'] for r in rows):
        raise SystemExit('XP static audit failed; inspect the report.')


if __name__ == '__main__':
    main()
