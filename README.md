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
    push hl
    pop de
    jp LABEL
LABEL:
    ret
"""

optimized = optimize(assembly)
print(optimized)
# Output:
#     xor a          ; ld a,0 → xor a (smaller)
#     ld d,h         ; push/pop → register moves (faster)
#     ld e,l
#     ret            ; jp to next instruction eliminated
```

### Using the Optimizer Class

```python
from upeepz80 import PeepholeOptimizer

# Create optimizer
optimizer = PeepholeOptimizer()

# Optimize assembly code
optimized_asm = optimizer.optimize(assembly_text)

# Check statistics
print(f"xor a conversions: {optimizer.stats.get('xor_a', 0)}")
print(f"Jump threading: {optimizer.stats.get('jump_thread', 0)}")
print(f"djnz conversions: {optimizer.stats.get('djnz', 0)}")
```

## Optimization Phases

The optimizer runs multiple phases:

1. **Pattern Matching** - Apply peephole patterns (up to 10 passes)
2. **Z80-Specific Optimizations** - Inline patterns for Z80 instructions
3. **Jump Threading** - Thread through intermediate jumps
4. **Relative Jump Conversion** - Convert jp to jr where possible
5. **djnz Optimization** - Convert dec b; jr nz to djnz
6. **Dead Store Elimination** - Remove unused stores

## Architecture

upeepz80 is designed to be language-agnostic:

- Works directly on Z80 assembly text
- No knowledge of source language required
- Pattern-based transformation engine
- Zero runtime dependencies

## Comparison with upeep80

| Feature | upeep80 | upeepz80 |
|---------|---------|----------|
| Input | 8080 or Z80 mnemonics | Z80 mnemonics only |
| Output | Z80 or 8080 (configurable) | Z80 only |
| Translation | 8080 → Z80 translation | None needed |
| Use case | Compilers generating 8080 code | Compilers generating Z80 code |

Choose **upeepz80** if your compiler already generates Z80 mnemonics.
Choose **upeep80** if your compiler generates 8080 mnemonics.

## Used By

- **[uplm80](https://github.com/avwohl/uplm80)** - PL/M-80 compiler for Z80 (after migration)
- **[uada80](https://github.com/avwohl/uada80)** - Ada compiler for Z80 (after migration)

## Development

### Running Tests

```bash
pytest
```

### Type Checking

```bash
mypy upeepz80
```

### Code Formatting

```bash
black upeepz80
ruff check upeepz80
```

## Performance

Benchmarks on typical compiler workloads:

- Peephole optimization: ~50,000 instructions/second
- Memory usage: Minimal (pattern-based, no large data structures)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the GNU General Public License v2.0 - see [LICENSE](LICENSE) for details.

## History

upeepz80 is a sibling project to upeep80, designed for compilers that generate native Z80 assembly. It shares the same optimization algorithms but removes the 8080 translation layer for cleaner, more efficient code when 8080 support isn't needed.
## Related Projects

- [80un](https://github.com/avwohl/80un) - Unpacker for CP/M compression and archive formats (LBR, ARC, squeeze, crunch, CrLZH)
- [cpmdroid](https://github.com/avwohl/cpmdroid) - Z80/CP/M emulator for Android with RomWBW HBIOS compatibility and VT100 terminal
- [cpmemu](https://github.com/avwohl/cpmemu) - CP/M 2.2 emulator with Z80/8080 CPU emulation and BDOS/BIOS translation to Unix filesystem
- [ioscpm](https://github.com/avwohl/ioscpm) - Z80/CP/M emulator for iOS and macOS with RomWBW HBIOS compatibility
- [learn-ada-z80](https://github.com/avwohl/learn-ada-z80) - Ada programming examples for the uada80 compiler targeting Z80/CP/M
- [mbasic](https://github.com/avwohl/mbasic) - Modern MBASIC 5.21 Interpreter & Compilers
- [mbasic2025](https://github.com/avwohl/mbasic2025) - MBASIC 5.21 source code reconstruction - byte-for-byte match with original binary
- [mbasicc](https://github.com/avwohl/mbasicc) - C++ implementation of MBASIC 5.21
- [mbasicc_web](https://github.com/avwohl/mbasicc_web) - WebAssembly MBASIC 5.21
- [mpm2](https://github.com/avwohl/mpm2) - MP/M II multi-user CP/M emulator with SSH terminal access and SFTP file transfer
- [romwbw_emu](https://github.com/avwohl/romwbw_emu) - Hardware-level Z80 emulator for RomWBW with 512KB ROM + 512KB RAM banking and HBIOS support
- [scelbal](https://github.com/avwohl/scelbal) - SCELBAL BASIC interpreter - 8008 to 8080 translation
- [uada80](https://github.com/avwohl/uada80) - Ada compiler targeting Z80 processor and CP/M 2.2 operating system
- [ucow](https://github.com/avwohl/ucow) - Unix/Linux Cowgol to Z80 compiler
- [um80_and_friends](https://github.com/avwohl/um80_and_friends) - Microsoft MACRO-80 compatible toolchain for Linux: assembler, linker, librarian, disassembler
- [uplm80](https://github.com/avwohl/uplm80) - PL/M-80 compiler targeting Intel 8080 and Zilog Z80 assembly language
- [z80cpmw](https://github.com/avwohl/z80cpmw) - Z80 CP/M emulator for Windows (RomWBW)

