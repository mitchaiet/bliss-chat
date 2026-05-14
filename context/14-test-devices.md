# Bliss Chat test devices

## Dell Inspiron 8600 Series

Source: BIOS photo from Mitch, 2026-05-14.

Visible hardware:
- Model: Dell Inspiron 8600 Series
- CPU: Intel Pentium M 1.40 GHz / 600 MHz
- L2 cache: 1024 KB
- RAM: 512 MB @ 333 MHz
- GPU: NVIDIA GeForce FX Go5200
- VRAM: 64 MB
- Panel: 15.4" Wide UXGA
- Audio: SigmaTel 9750
- Modem: Conexant D480 MDC
- Primary hard drive: 40 GB
- Modular bay: DVD+RW
- System D-Bay: Not installed
- Service tag / asset tag: Not installed in BIOS screen

Bliss Chat result:
- Installer runs, but NC_RUN crashes on this machine.

Likely diagnosis:
- Current XP build uses `-msse3 -march=pentium4` in `scripts/build-xp.sh`.
- Pentium M generation in the Inspiron 8600 class supports SSE/SSE2 but not SSE3.
- If MinGW emitted SSE3 instructions, NC_RUN will crash on launch with an illegal-instruction style failure.
- RAM is also tight: d12 int8 model is ~280 MB, plus KV/cache and Windows XP overhead on a 512 MB machine. That can cause load/allocation failures, but the CPU flag mismatch is the first thing to fix for an immediate NC_RUN crash.

Next build to test:
- Rebuild NC_RUN/XPCHAT with no SSE3, e.g. keep SSE2 only (`-msse -msse2`) and use a safer arch/tune for Pentium M/i686 rather than `-march=pentium4 -msse3`.
- If it still fails after the SSE2-only build, test a smaller/q4 model or lower-memory configuration for 512 MB systems.
