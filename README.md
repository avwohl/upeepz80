# upeepz80 - Universal Peephole Optimizer for Z80

A language-agnostic peephole optimization library for Z80 compilers that generate pure Z80 assembly.

## Overview

upeepz80 provides high-quality optimization passes for compilers targeting the Zilog Z80 processor. Unlike upeep80, this library expects **pure Z80 mnemonics in lowercase** as input (ld, jp, jr, etc.) and produces optimized Z80 assembly output with lowercase mnemonics.

If your compiler generates 8080 mnemonics (MOV, MVI, LXI, etc.) that need translation to Z80, use [upeep80](https://github.com/avwohl/upeep80) instead.

## Features

### Peephole Optimizations
- **Pattern-based optimization** on Z80 assembly
- **Redundant load/store elimination**
- **Jump optimization** (jp to jr conversion, jump threading)
- **djnz conversion** (dec b; jr nz → djnz)
- **Stack operation combining** (push/pop to ld conversions)
- **Dead store elimination**
- **Tail call optimization** (call x; ret → jp x)
- **Register copy optimization** (push hl; pop de → ld d,h; ld e,l)

### Z80-Specific Features
- Relative jump optimization (jp → jr where in range)
- djnz loop optimization
- Z80 block instruction awareness
- Direct ld de,(addr) usage (Z80-only instruction)

## Installation

```bash
pip install upeepz80
```

Or for development:

```bash
git clone https://github.com/avwohl/upeepz80.git
cd upeepz80
pip install -e ".[dev]"
```

## Usage

### Basic Usage

```python
from upeepz80 import optimize

# Optimize Z80 assembly code
assembly = """
    ld a,0
    or b
    push hl
    pop de
    jp LABEL
LABEL:
    ret
"""

optimized = optimize(assembly)
print(optimized)
# Output:
#     xor a          ; ld a,0 → xor a (smaller): or b sets the flags xor a changes
#     or b
#     ld d,h         ; push/pop → register moves (faster)
#     ld e,l
# LABEL:
#     ret            ; jp to next instruction eliminated
```

Without the `or b`, `ld a,0` stays: `xor a` also sets the flags, and the
caller this code returns to might read them.

For the `PeepholeOptimizer` class and its statistics, see [docs/INTEGRATION.md](docs/INTEGRATION.md).

## Documentation

- [docs/INTEGRATION.md](docs/INTEGRATION.md): integration guide, the `PeepholeOptimizer` class, statistics, input format, migrating from upeep80
- [docs/design.md](docs/design.md): optimization phases, correctness, architecture, performance, comparison with upeep80, history
- [docs/development.md](docs/development.md): running the tests, type checking, code formatting, contributing
- [CHANGELOG.md](CHANGELOG.md): release history

## License

This project is licensed under the GNU General Public License v2.0 - see [LICENSE](LICENSE) for details.

## Related Projects

- [80un](https://github.com/avwohl/80un) - Unpacker for the CP/M archive and compression formats LBR, ARC, squeeze, crunch, and CrLZH.
- [cpmdroid](https://github.com/avwohl/cpmdroid) - Z80/CP/M emulator for Android phones and tablets. It emulates the RomWBW HBIOS interface and a VT100 terminal.
- [cpmemu](https://github.com/avwohl/cpmemu) - Z80/CP/M emulator for Linux and Windows, with Z80 and 8080 CPU cores. It translates the BDOS and BIOS calls of CP/M 2.2 programs to the host file system.
- [ioscpm](https://github.com/avwohl/ioscpm) - Z80/CP/M emulator for iOS and macOS. It emulates the RomWBW HBIOS interface and runs CP/M 2.2 and CP/M 3.
- [learn-ada-z80](https://github.com/avwohl/learn-ada-z80) - Collection of more than 90 Ada example programs for uada80, the Ada compiler for the Z80 processor and CP/M.
- [mbasic](https://github.com/avwohl/mbasic) - Python interpreter for MBASIC 5.21, the Microsoft BASIC-80 for CP/M. Two compiler backends compile the programs to CP/M .COM files or to JavaScript.
- [mbasic2025](https://github.com/avwohl/mbasic2025) - Reconstruction of the lost source code of MBASIC 5.21, the Microsoft BASIC-80 for CP/M. The MACRO-80 source code assembles to a binary that matches mbasic.com byte for byte.
- [mbasicc](https://github.com/avwohl/mbasicc) - C++17 interpreter for MBASIC 5.21, the Microsoft BASIC-80 for CP/M. It runs on Linux and macOS.
- [mbasicc_web](https://github.com/avwohl/mbasicc_web) - Web browser interpreter for MBASIC 5.21, the Microsoft BASIC-80 for CP/M. Emscripten compiles the mbasicc interpreter to WebAssembly.
- [mpm2](https://github.com/avwohl/mpm2) - Z80 emulator for MP/M II, the multi-user CP/M operating system. Users connect over SSH, and SFTP clients transfer files.
- [romwbw_emu](https://github.com/avwohl/romwbw_emu) - Hardware-level Z80/CP/M emulator for Linux and macOS. It emulates the RomWBW HBIOS interface and switches banks in 512 KB of ROM and 512 KB of RAM.
- [scelbal](https://github.com/avwohl/scelbal) - Floating-point BASIC interpreter for the 8080 processor and CP/M. A translator converts the original 8008 source code to 8080 source code.
- [uada80](https://github.com/avwohl/uada80) - Ada compiler for the Z80 processor and CP/M 2.2. It compiles a subset of Ada 2012 to CP/M .COM files.
- [uc80](https://github.com/avwohl/uc80) - C compiler for the Z80 processor and CP/M. It optimizes for small code size.
- [ucow](https://github.com/avwohl/ucow) - Cowgol compiler for the Z80 processor and CP/M. It runs on Linux in Python.
- [um80_and_friends](https://github.com/avwohl/um80_and_friends) - Linux toolchain that is compatible with Microsoft MACRO-80. It has an assembler, a linker, a librarian, and a disassembler.
- [uplm80](https://github.com/avwohl/uplm80) - PL/M-80 compiler for the Z80 processor and CP/M. It writes Intel 8080 and Zilog Z80 assembly language.
- [z80cpmw](https://github.com/avwohl/z80cpmw) - Z80/CP/M emulator for Windows. It emulates the RomWBW HBIOS interface and boots CP/M from disk images.

## See Also

- [upeep80](https://github.com/avwohl/upeep80) - Optimizer with 8080 input support
- [Z80 CPU User Manual](http://www.z80.info/zip/z80cpu_um.pdf)
