# Bliss Chat XP v1.2.1 - automatic CPU compatibility build

This release keeps the single self-contained portable `.exe` format, but includes two native inference backends:

- `NC_RUN_SSE2.EXE` for Pentium M / older Windows XP systems with SSE2 but no SSE3.
- `NC_RUN_SSE3.EXE` for newer CPUs that advertise SSE3.

`XPCHAT.EXE` detects CPU features at startup and launches the best compatible backend automatically. This should avoid illegal-instruction crashes seen on Dell Inspiron 8600 / Pentium M class hardware while preserving the optimized path on newer XP-era machines.

## Artifact

- `bliss-chat-xp-v1.2.1-auto-cpu-portable.exe`
- SHA-256: `3f961cdcd532ef5d8e5004298da5d3a6cc1dbc9be135e938b33333d38364c182`

## Notes

- Still bundles the Bliss d12 curated int8 model.
- 512 MB RAM machines may remain memory-constrained; upgrading to 1-2 GB is recommended.
- If NC_RUN still fails after this build, check the generated `backend-*.log` next to the extracted runtime.
