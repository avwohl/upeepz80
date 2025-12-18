# Integration Guide for upeepz80

## Overview

upeepz80 provides a language-agnostic peephole optimizer for Z80 assembly. It works directly on assembly text and requires no adaptation to your compiler's internals.

## Prerequisites

Your compiler must generate **pure Z80 mnemonics**:
- LD (not MOV, MVI, LDA, STA, etc.)
- JP, JR (not JMP, JZ, JNZ, etc.)
- CP (not CMP, CPI)
- INC, DEC (not INR, DCR, INX, DCX)
- ADD, SUB, AND, OR, XOR (not ANA, ORA, XRA, etc.)
- PUSH, POP with Z80 register names (AF not PSW, BC/DE/HL not B/D/H)

If your compiler generates 8080 mnemonics, use [upeep80](https://github.com/yourusername/upeep80) instead.

## Basic Usage

```python
from upeepz80 import optimize

# Your compiler generates Z80 assembly
assembly = generate_z80_code(ast)

# Optimize it
optimized = optimize(assembly)

# Write output
write_output(optimized)
```

## Using the Optimizer Class

For more control, use the `PeepholeOptimizer` class directly:

```python
from upeepz80 import PeepholeOptimizer

# Create optimizer
optimizer = PeepholeOptimizer()

# Optimize assembly code
optimized_asm = optimizer.optimize(assembly_text)

# Access statistics
print(f"Optimizations applied:")
for name, count in optimizer.stats.items():
    print(f"  {name}: {count}")
```

## Optimization Statistics

The optimizer tracks various statistics:

```python
# After optimization
print(f"XOR A conversions: {optimizer.stats.get('xor_a', 0)}")
print(f"Jump threading: {optimizer.stats.get('jump_thread', 0)}")
print(f"DJNZ conversions: {optimizer.stats.get('djnz', 0)}")
print(f"JP to JR: {optimizer.stats.get('jr_convert', 0)}")
print(f"Push/Pop to LD: {optimizer.stats.get('push_pop_copy_hl_de', 0)}")
print(f"Dead stores removed: {optimizer.stats.get('dead_store_elim', 0)}")
```

## Custom Patterns

You can add custom patterns to the optimizer:

```python
from upeepz80 import PeepholeOptimizer, PeepholePattern

# Create optimizer
optimizer = PeepholeOptimizer()

# Define custom pattern
custom_pattern = PeepholePattern(
    name="my_custom_pattern",
    pattern=[
        ("LD", "A,0"),
        ("LD", "B,A"),
    ],
    replacement=[
        ("XOR", "A"),
        ("LD", "B,A"),
    ]
)

# Add to optimizer
optimizer.patterns.append(custom_pattern)
```

## Integration Example

### Complete Compiler Integration

```python
from upeepz80 import PeepholeOptimizer

class MyCompiler:
    def __init__(self):
        self.optimizer = PeepholeOptimizer()

    def compile(self, source_code: str) -> str:
        # Parse source
        ast = self.parse(source_code)

        # Semantic analysis
        self.analyze(ast)

        # Generate Z80 assembly
        assembly = self.generate_code(ast)

        # Optimize assembly
        optimized = self.optimizer.optimize(assembly)

        # Report statistics
        self.report_optimization_stats()

        return optimized

    def report_optimization_stats(self):
        total = sum(self.optimizer.stats.values())
        if total > 0:
            print(f"Peephole optimizations: {total}")
            for name, count in sorted(self.optimizer.stats.items()):
                if count > 0:
                    print(f"  {name}: {count}")
```

## Expected Input Format

The optimizer expects standard Z80 assembly format:

```asm
; Comments start with semicolon
LABEL:
    LD A,0        ; Load instruction
    CP B          ; Compare
    JP Z,DONE     ; Conditional jump
    CALL FUNC     ; Call subroutine
    RET           ; Return
DONE:
    LD HL,1234H   ; 16-bit load
    RET
```

### Supported Features

- Labels (with colon)
- Standard Z80 mnemonics
- Hex numbers (0FFH format)
- Comments (semicolon)
- Directives (ORG, DB, DW, DS, EQU) are preserved

### Indentation

The optimizer preserves the structure of your code:
- Labels are not indented
- Instructions are indented with tab or spaces
- Comments are preserved

## Migrating from upeep80

If you're migrating from upeep80:

1. **Change your import**:
   ```python
   # Before
   from upeep80 import optimize_z80

   # After
   from upeepz80 import optimize
   ```

2. **Ensure Z80 output**: Your code generator must produce Z80 mnemonics directly. If it still generates 8080 mnemonics, either:
   - Update the code generator to emit Z80
   - Continue using upeep80 with `InputSyntax.I8080`

3. **Update function calls**:
   ```python
   # Before (upeep80)
   from upeep80 import PeepholeOptimizer, Target, InputSyntax
   optimizer = PeepholeOptimizer(
       target=Target.Z80,
       input_syntax=InputSyntax.Z80
   )

   # After (upeepz80)
   from upeepz80 import PeepholeOptimizer
   optimizer = PeepholeOptimizer()  # Z80-only, no options needed
   ```

## Notes

- The optimizer is completely language-agnostic
- Works on any Z80 assembly regardless of source language
- No runtime dependencies
- Safe to run multiple times (convergent)
- Thread-safe (no global state)

## See Also

- [README.md](../README.md) - Overview and features
- [upeep80](https://github.com/yourusername/upeep80) - For 8080 input support
