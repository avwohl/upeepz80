# upeepz80 Design

How the optimizer works, what keeps each rewrite correct, and how upeepz80 relates to upeep80.

## Optimization Phases

The optimizer runs multiple phases:

1. **Pattern Matching** and **Z80-Specific Optimizations** - peephole
   patterns and inline rewrites, repeated until nothing changes (up to 10
   passes)
2. **Jump Threading** - Thread through intermediate jumps
3. **Pattern Matching** once more, for what threading exposed
4. **Dead Store Elimination** - Remove a parameter's store at procedure
   entry when nothing can read the byte: no load reads it, nothing
   gives an address from which it can be computed, such as that of the
   variable before it (`ld hl,A / inc hl`), and no jump, call or
   instruction before it runs it as code
5. **Relative Jumps** - Convert jp to jr, and dec b; jp nz to djnz, where the
   target is in reach; last, because it counts bytes

## Correctness

A rewrite that changes what a register or flag holds afterwards is made only
where nothing reads the old value. `ld a,0` → `xor a` changes every flag, so
it is made only where the flags are overwritten before they are read; `ld
hl,5 / ld a,l` → `ld a,5` leaves HL as it was, so it is made only where HL
is. The optimizer knows what each Z80 instruction reads and writes
(`upeepz80/z80.py`), and follows every path from the rewritten code: on,
into both arms of a branch, round loops, into a routine the text calls and
back, from a `ret` to every call of the routine, and through a `push` to its
`pop`. A path that leaves what the text shows counts as reading
everything: a call or jump to a label defined elsewhere, `call 5`, `jp 0`,
`jp (hl)`, data, or the end of the text. So does a `ret` from a routine that
another module may call (`public`, `NAME::`, or its address taken), and one
that may not take the return address its routine was entered with. The
optimizer follows where that address is: through `pop`, `push` and `ex
(sp),hl`, and the calls of routines that take their arguments off the
stack, as under PL/M-80's calling convention, which uplm80 0.4.0 uses
(`pop hl / ex (sp),hl`). A `ret` goes back to the call only where the
address is at the top of the stack (not after `ex (sp),hl / ret`), where it
may not have been changed through a pointer made from SP (`ld hl,0 / add
hl,sp / ld (hl),e`), and where the routine has not called one that goes on
where the optimizer cannot follow (`jp (hl)`, `push hl / ret`).

`call x / ret` becomes `jp x` only where the routine's own return address is
at the top of the stack, as `ret` takes it. Jumped to, `x` finds on top of
the stack what `call x` would have put under its return address. That
matters to a callee that removes the arguments pushed for it. Nor is it made
where `x` is a routine of the text that reaches its return address or what
is above it, directly or through a routine it calls, or leaves the stack
other than it found it; nor where a jump or `ret` the optimizer cannot
follow (`push bc / ld hl,HND / jp (hl)`, `ex (sp),hl / ret`) may have
entered the routine with something pushed.

A program that computes an address from a label - `jp BIOS+3`, `ld hl,L+3`,
`jr $+3`, `dw START-3` - depends on the size of the code between, and may go
to the code there. No rewrite changes that code, and the code at the address
is taken to be entered from anywhere. So it is with the `jp` and `jr`
instructions, and any `nop`, `db` or `ds` that pads them (`jp H0 / nop / jp
H1 / nop`), after a label the text names other than as where a jump or call
goes, or a name an `equ` sets to `$` there (`TBL equ $`), a table the
program may enter at an offset it computes (`ld de,TBL / add hl,de / jp
(hl)`), and after a label the text exports, which another module may enter
at an offset, as a BIOS's jump vector is. Where the program may read or
write the code at the address (`ld hl,(VEC+1)`, `ld (VEC+1),hl`, `ld
(SW),a`), the instruction there may have been patched into any other: a path
that goes there counts as reading everything, whatever the instruction; that
instruction, and a jump of an exported vector, is taken to go where the
optimizer cannot follow, or on to the line after it (a `ret` made `nop`, a
`jp` made `ld hl,nn`), which may then be entered from anywhere; and no jump
is threaded through it. No tail call is made to a routine that reaches it,
nor, where something may be pushed there, in code that may be entered from
anywhere. A `dw L`, where L is `jp M`, becomes `dw M` only in a table that
is only jumped through, as uplm80's `DO CASE` tables are, and where the code
at M reads neither HL, which holds L or M, nor DE, which points at the
entry.

Numbers are read under the text's radix. Where it sets a `.radix` other than
ten, only numbers that mean the same under any radix are rewritten.

Every rewrite writes only instructions the Z80 has, and a relative jump is
made only where its target is known to be within reach, counting bytes.

`tests/peepfuzz.py` checks this: it generates random programs around every
shape the optimizer rewrites, runs each before and after optimization on a
Z80 interpreter (`tests/z80sim.py`, itself checked against a real Z80 core
by `tests/simcheck.py`), and compares every register, flag and byte of
memory.

## Architecture

upeepz80 is designed to be language-agnostic:

- Works directly on Z80 assembly text
- No knowledge of source language required
- Pattern-based transformation engine
- Zero runtime dependencies

## Performance

Benchmarks on typical compiler workloads:

- Peephole optimization: ~50,000 instructions/second
- Memory usage: Minimal (pattern-based, no large data structures)

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

## History

upeepz80 is a sibling project to upeep80, designed for compilers that generate native Z80 assembly. It shares the same optimization algorithms but removes the 8080 translation layer for cleaner, more efficient code when 8080 support isn't needed.
