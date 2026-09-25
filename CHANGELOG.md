# Changelog

Notable changes to upeepz80. Releases up to 0.2.4 are described on the
[GitHub releases page](https://github.com/avwohl/upeepz80/releases).

## Unreleased

### Fixed

- **Dead-store elimination removed a store that an address computed from
  another label reaches.** PL/M-80 lays out `declare (a, b) byte` one
  variable after the other, and `.a + 1` is the address of b. uplm80
  compiles `pq: procedure (a, b) byte; declare (a, b) byte; ... pp = .a +
  1; return c;`, where `c` is BASED on `pp`, to a store of `b` at pq's entry
  that no other line names. 0.2.5 removed that store, so `pq('A', 'B')`
  returned whatever the byte held. uplm80 found this, and 0.3.7 adds an
  `EQU` that names such a parameter to keep the store. 0.2.5 took storage
  to be reachable only by its own name. Now a store is removed only where
  nothing can compute the address of the byte. That is so where all of
  these are true:
  - The byte is in the data (`ds`, `db`, `dw`) between two instructions of
    a segment. The segment is `cseg` or `dseg`, and the text does not place
    it with `org` or `.phase`.
  - No operand, `db`, `dw` or `equ` uses a label of that data as a value
    (`ld hl,PA`, `db LOW PA`), and neither does `$` in the instruction
    before it (`jp $+3`). Nothing exports such a label. SP is not loaded
    from it.
  - The data is not run as code, as it is where code patches an
    instruction (`OPC: db 0`). No jump or call goes to a label in the
    data. The instruction before it is a `jp`, `jr`, `ret`, `reti`,
    `retn` or `jp (hl)` with no condition, so control does not go on into
    the data. A `call` or `rst` there would return into it.
  - No load reads the byte. `ld a,(PA+1)`, and `ld hl,(PA)`, which reads
    the byte after PA too, read PB where PB is PA+1.

  0.2.5 also removed a store to data that the code before it goes on
  into (`RUN: nop / OPC: db 0`). The reasons, and what they assume, are
  in the docstring of `_Storage` in `upeepz80/peephole.py`. In short: a
  program cannot depend on where the linker puts a segment, or on the
  size of code, which the optimizer changes. No store is removed from a
  text that has a conditional, a macro, an `include`, a name defined
  twice or an instruction the optimizer does not know.

  On MP/M II and 80un at `-O2`, this keeps 17 of the 28 stores that 0.2.5
  removed with uplm80 0.3.7. The 77 outputs that assemble grow by 45
  bytes, from 213,017 to 213,062. Without uplm80's `EQU`, it keeps 19 of
  30, and the code is the same as with it.
- **A value on the stack was taken to be read only by its `pop`, and by
  what reads SP.** Code can also read it through a pointer made from SP
  before the push. After `ld hl,0 / add hl,sp / dec hl / ex de,hl`, DE
  points where `push hl` puts H. In `ld hl,300 / ld a,l / push hl / ld
  a,(de) / pop bc / ld bc,0`, HL was followed through the push to the
  `pop`, into BC, which is overwritten. So HL was taken for dead, `ld
  hl,300 / ld a,l` became `ld a,02Ch`, and `ld a,(de)` read the old H.
  Now, where the text makes a pointer from SP anywhere (`add hl,sp`, `add
  ix,sp`, `ld (nn),sp`), a read of memory through HL, BC, DE, IX or IY
  (and `ldir` and the like) counts as a read of what is on the stack.
  Another module is taken not to reach below the SP that it calls with.
  On the corpus this costs 13 `cp 0` → `or a` and one `ld a,0` → `xor
  a`, 14 bytes in six outputs.
- **`push af / ld (hl),a / pop af` became `ld (hl),a`,** although HL can
  point at the slot the push fills. There the store changes the flags that
  the pop takes back. Where the text makes a pointer from SP, the pair is
  now kept around a store through a register. A store to an address the
  text gives is not to the stack (see Known issues). This changes nothing
  on the corpus.
- **A routine that changes its return address through a pointer made from
  SP was taken to return to its call.** 0.2.5 listed this as a known
  issue. `P: ld hl,0 / add hl,sp / ld de,THERE / ld (hl),e / inc hl / ld
  (hl),d / ld a,0 / ret` returns to THERE. Its `ret` was followed to the
  line after `call P`, where `cp b` writes the flags, so `ld a,0` became
  `xor a`, and THERE read the zero flag that `xor a` sets. Now, where the
  text makes a pointer from SP anywhere, a routine that writes through a
  pointer, or calls one that does, may return anywhere. Another routine
  may have made the pointer and kept it (`ld (W),sp`), or a routine it
  calls may use it on the caller's return address. Where the text makes
  no such pointer, nothing changes.
- **`call x / ret` became `jp x` for a routine that reads its return
  address, or what is above it.** 0.2.5 listed this as a known issue too.
  `RD: ld hl,4 / add hl,sp / ld a,(hl)` reads the word pushed before its
  caller was called. Jumped to, it finds one return address fewer on the
  stack, and reads another word. A routine can also read through a
  pointer that its caller made before the call: after `W: ld hl,0 / add
  hl,sp / dec hl / dec hl`, HL points where `call RB` puts RB's return
  address, which `RB: ld a,(hl)` reads, and `jp RB` puts none there. No
  tail call is now made to a routine that makes a pointer from SP, or,
  where the text makes one anywhere, that reads through a pointer; nor to
  a routine that calls one of those.

  Together these two cost 85 bytes on the corpus, in nine outputs: 30 `ld
  a,0` → `xor a`, 22 tail calls, 18 `cp 0` → `or a`, 7 relative jumps, 4
  `ld hl,n / ld r,l` and one threaded jump. 61 of the bytes are in 80un's
  two programs of several modules, whose procedures take their
  parameters on the stack and read them with `ld hl,2 / add hl,sp`.
- **`call x / ret` became `jp x` where the code that reads above its
  return address is reached in a way the optimizer does not follow.**
  0.2.5 did the same. With `RTN:` as RD above, `call rtn / ret` became
  `jp rtn`: the optimizer found no label `rtn`, but M80 does not tell
  case. So did `call DSP / ret` where DSP goes on to RTN through `jp
  (hl)`, `jp (ix)`, `push hl / ret`, or `jp ALIAS` with `ALIAS equ RTN`.
  No tail call is now made to a routine that goes on where the optimizer
  cannot follow, nor to one that calls such a routine. That is `jp
  (hl)`, `jp (ix)`, `jp (iy)`, a `ret` to what the routine pushed,
  `reti`, `retn`, `halt`, data, a directive or an instruction the
  optimizer does not know, and a jump or call to an address of the text
  that is not one of its labels as written (`jp ALIAS`, `call rtn`, `jp
  $+3`). Nor is a tail call made to such an address itself. A number,
  and a name the text does not define or sets to a number (`BDOS equ
  5`), are taken to be outside the text. On the corpus this costs 14
  bytes: five tail calls in each of the two builds of PIP.PLM, whose `DO
  CASE` goes through `jp (hl)`.
- **`push bc / call SHOWP / ret` became `push bc / jp SHOWP`,** as in
  0.2.5. Jumped to, SHOWP finds BC where its return address was, and its
  caller's return address where BC was. SHOWP here is in another module
  and takes its argument off the stack (`pop hl / pop de / push hl`). It
  took the return address for the argument and returned to BC, and the
  program hung. No tail call is now made where the caller may have pushed
  something: where the stack height at the `call` is other than 0, or not
  known, as after `ld sp,hl`. On the corpus this costs one byte, in
  LOAD.PLM.

### Added

- `tests/test_storage.py`: the program above, run before and after
  optimization, each way to a byte from a neighbour's address, and each
  way control gets to data that code patches. 44 of its 68 tests fail on
  0.2.5. `tests/z80sim.py` now lays out the data a text defines as an
  assembler does, so that `ld hl,A+1` finds the byte after A, and reads
  M80's `LOW A` and `HIGH A`.
- `tests/test_liveness.py` and `tests/test_control_flow.py`: a pushed
  value read, and a return address changed, through a pointer made from
  SP, three ways each, a store between `push af` and `pop af`, and the
  tail calls above, three ways. Each runs the code before and after
  optimization, and fails on 0.2.5. So do the tail calls to RTN through a
  name in lower case, an equate, `jp (hl)`, `jp (ix)`, `jp (iy)` and `push
  hl / ret`, those to a routine that goes on to code that pops what its
  caller pushed, or reads through its caller's pointer, and the tail call
  to SHOWP. Eleven more ways for a routine to go where the optimizer
  cannot follow each keep the call. `tests/z80sim.py` now jumps to a name
  an equate sets to a label.
- `tests/peepfuzz.py`: half the programs now also call a routine that
  stores its parameter at its entry, between two neighbours in storage the
  program defines, and then read the byte or not, by its own name or from
  a neighbour's address. The stored byte itself is not compared. 0.2.5
  differs on 69 of the first 400 programs; this release on none of 3,000.

### Known issues

- **The stack is taken to be reached only relative to SP:** by `pop`,
  `ret`, `ex (sp)`, and pointers made from SP. An address that the text
  gives, a label or a number, is taken not to be a slot on the stack. The
  optimizer changes how deep the stack is where a routine runs (a tail
  call runs it one return address higher), and what is below SP (it
  removes a `push` with its `pop`). So `push hl / ld hl,(x) / ex de,hl /
  pop hl` → `ld de,(x)`, and `push hl / ld (x),hl / pop hl` and `push af /
  ld (x),a / pop af` → the instruction in the middle, assume that x is not
  the slot the push fills. Code outside the text is taken to change no
  return address but its own, and not to reach below the SP it is called
  with. It is taken to come back with SP where it was before the call, not
  to take off the stack what its caller pushed. Called where its caller
  has pushed nothing, it is taken not to read above its return address,
  where its caller's return address is: `call EXT / ret` becomes `jp EXT`,
  and so does `call 5 / ret`. A number, or a name an `equ` sets to one, is
  taken to be an address outside the text, even where the text places
  itself with `org`. Nor is it taken to hand back a pointer into the
  stack, as a routine of another module that returns SP in HL would: only
  this text's pointers made from SP count. Nor, where it is handed such a
  pointer, to read through it what is above its own return address, which
  a tail call in this text changes: `W: ld hl,0 / add hl,sp / dec hl / dec
  hl / call RB / ret` with `RB: call EXT / ld a,1 / ret` still becomes `jp
  RB`, and EXT reads its own return address through HL, not RB's. Taking
  every routine that calls out of the text, where the text makes such a
  pointer, to read above its return address costs 16 bytes in four outputs
  of the corpus.
- **A program is taken not to depend on the size of its code,** which
  every rewrite changes. Dead-store elimination relies on this: an
  address is not computed across an instruction. A table of `jp`
  instructions entered at a computed offset, such as a BIOS's jump
  vector (`BIOS+3`), does depend on it. Relative jumps break it: `BIOS:
  jp BOOT / jp WBOOT` becomes `BIOS: jr BOOT / jr WBOOT`, as in 0.2.5.
  So does jump threading, which removes a jump whose label nothing names
  where the line before does not go on to it: `BIOS:: jp BOOT / WBE: jp
  WBOOT / CSE: jp CONST` keeps only its first line, as in 0.2.5.
  uplm80 writes such tables as `dw` lists, whose layout is not changed.
- **`dw L`, where L is `jp M`, becomes `dw M`,** as in 0.2.5. A word that
  holds the address of code is taken to be an address that is only
  jumped to. A table whose entries are compared, or used as data, would
  change.

## 0.2.5 - 2026-09-25

A rewrite that changes what a register or flag holds afterwards is now made
only where nothing reads the old value. Most rewrites were made wherever
their instructions matched, whatever came next. One, `ld hl,n / ld r,l`,
looked twelve instructions ahead for a read of HL, but missed most of the
ways HL can be read. uplm80's differential test found three defects of this
kind. An audit of every rewrite found the same mistake in most of the others,
and found rewrites that wrote instructions the Z80 does not have and relative
jumps that did not reach their target.

The optimizer now knows what each Z80 instruction reads and writes
(`upeepz80/z80.py`). It follows every path from the end of the rewritten
code until the value is overwritten: on, into both arms of a branch, round
loops, into a routine of the module that is called and back, from a `ret` to
the line after each call of the routine, and through a `push` to the `pop`
that takes it off the stack. A path that leaves what the text shows counts
as reading everything. Such paths go through a call or a jump to a label
defined elsewhere, `call 5`, `jp 0`, `jp (hl)`, data, or the end of the
text. They also go through a `ret` from a routine that another module may
call, or that may not return to its call. Another module may call a routine
that is `public`, exported with M80's `NAME::`, or whose address is taken.
A routine may not return to its call if it takes its return address off the
stack, or returns where the stack height is not known.

### Fixed

- **`ld a,(x) / inc a / ld (x),a` became `ld hl,x / inc (hl)` although A was
  read afterwards.** The rewrite also left HL holding x, where the original
  left HL alone. uplm80 compiles `b, w = -(NOT b)` to `ld a,(b) / cpl / cpl
  / inc a / push af / ld (b),a / pop af`. Other rewrites reduce that to this
  shape, so `w` got A's old value at `-O1` and `-O2`. The rewrite is now made
  only where A, H and L are all dead.
- **`ld hl,n / ld r,l` became `ld r,n` although HL was read afterwards.** The
  check that HL was dead did not count `ld (nn),hl` as a read. It also
  missed `(hl)` addressing, `ex (sp),hl`, `jp (hl)` and `sbc hl,rr`, gave up
  at a branch, and took HL for dead twelve instructions on. So `W = (B :=
  9)`, which is `ld hl,9 / ld a,l / ld (b),a / ld (w),hl / ld hl,3`, stored
  whatever HL held. uplm80's `LOW(SIZE(x))` and `w1, w0 = (sb := ...)`
  stored it too (uplm80's difftest seeds 80071 and 1063).
- **`ld a,(ix+n) / inc a / ld (ix+n),a` became `ld hl,ix+n / inc (hl)`,**
  which is not a Z80 instruction. Likewise `ld a,(hl) / inc a / ld (hl),a`
  became `ld hl,hl`. A REENTRANT procedure's BYTE loop did not assemble at
  `-O1` and above. The rewrite is now `inc (ix+n)` or `inc (hl)`, made where
  A is dead.
- **Rewrites that changed the flags, made whether or not the flags were
  read.** The original of each leaves some flags alone that its replacement
  sets, or sets them differently:
  - `ld a,0` → `xor a` sets every flag. uplm80 avoids `ld a,0 / rla`
    because of this one.
  - `cp 0` → `or a` changes P/V and N.
  - `and 0ffh` → `or a` changes H.
  - Removing `inc a / dec a` or `dec a / inc a` loses the dec's or inc's S,
    Z, H, P/V and N.
  - Removing `ccf / ccf` or `cpl / cpl` loses their H and N.
  - `ld de,1..3 / add hl,de` → `inc hl` loses the add's H, N and C (and
    DE's value).
  - `ld hl,0 / ld a,l / ld (x),a` → `xor a / ld (x),a / ld hl,0` sets every
    flag.
  - The 8080 right shift → `srl h / rr l` changes S, Z and P/V (and A).
  - `dec b / jp nz` → `djnz` loses the dec's S, Z, H, P/V and N.

  Each is now made only where the flags it changes are dead.
- **Rewrites that dropped a register write, made whether or not the register
  was read.**
  - `ld a,(hl) / ld r,a` → `ld r,(hl)` left A unloaded.
  - `ld h,0 / ld d,h / ld e,l` → `ld d,0 / ld e,l` left H as it was.
  - `ld l,a / ld h,0 / sub x` or `cp x`, `ld a,l / ld h,0 / ld (x),a`, `ld
    l,a / ld h,0 / ld (x),a` and `ld a,l / ld h,0 / or h` left H or L as
    they were. That includes the case where the kept instruction reads them
    itself, as `sub l`, `cp (hl)` and `ld (hl),a` do.
  - `ld hl,1 / ld c,l`, `ld hl,0 / ld a,l / or h` and `ld hl,1 / ld a,l /
    or h` left HL as it was.
  - `ld de,2^k / call ??mul16` → `add hl,hl` leaves A, BC, DE and the flags
    as they were. The routine leaves its own values there; uplm80's leaves
    DE and A at 0 and the Z flag set.
  - `ld de,0 / call ??subde` was removed. The routine's documented output
    includes the flags.

  Each is now made only where what it leaves different is dead.
- **`ld hl,0ffffh / ld a,l / or h` became `ld hl,0ffffh / or a`,** which
  tests A's old value instead of 0FFH. It is now `ld a,0ffh / or a`, where
  HL is dead.
- **`ld hl,0 / ld a,l / ld (hl),a` stored to HL's old address** after it
  became `xor a / ld (hl),a / ld hl,0`.
- **`ld l,(hl)` twice was taken for one,** although the first changes the
  address the second reads.
- **A routine that puts another address in place of its return address
  was taken to return to its call.** `ex (sp),hl / ret` goes to HL, and
  `pop hl / push de / ret` goes to DE. So before `call dispatch`, `ld a,0`
  became `xor a`, because the code after the call writes the flags. But the
  code that `dispatch` really goes to reads the carry. Now:
  - A routine that takes its return address off the stack can return
    anywhere. That is a `pop` or `ex (sp),rr` where nothing of the routine's
    own is pushed.
  - After a call of such a routine, the stack height is not known. The same
    is true after a call of a routine that returns where the height is not
    known.
  - `call x / ret` is not made `jp x` for such a routine. Entered by a jump,
    it finds its caller's return address where it looks for its own.
- **`call cc,x / ret` became `jp cc,x`,** which goes on, instead of
  returning, when the condition fails.
- **A label after `ret cc` was taken for unreachable.** When nothing named
  it, it was removed with the jump after it. So when the `ret` did not
  return, control fell into whatever came next.
- **A label named only in a `dw` of several entries** (`dw L0,L1`) counted
  as named nowhere, so jump threading could remove it.
- **A label with an instruction on the same line** (`L: ld a,5`) caused
  four faults:
  - A rewrite starting there dropped the label.
  - Making the jump on such a line relative dropped the label: `L1: jp L2`
    became `jr L2`, and jumps to L1 went nowhere. Jump threading did the same
    when it gave that jump a new target.
  - Jump threading took the next line's jump for the label's first
    instruction, and sent jumps past `ld a,5`.
  - Removing an unused label also removed the next jump.
- **A pattern could match across a data directive** as if it were a
  comment. `push hl / db 5 / pop hl` lost its push and pop.
- **Relative jumps.** `jp` became `jr`, and `dec b / jp nz` became `djnz`,
  where the target was within 40 or 50 *lines*. That is 125 bytes only if
  no line is long: `ds`, a `db` of a string or a run of four-byte
  instructions put the target out of reach. um80 turns such a `jr` back into
  `jp`, and such a `djnz` into `dec b / jp nz`, with a note. Other
  assemblers stop with an error. Now:
  - Distances are counted in bytes.
  - A jump is made relative only where it is known to reach.
  - No distance is measured across a line whose size is unknown (a macro,
    `ds` of a name, `org`).
  - Jump threading gave a `jr` a new target out of its reach. It now leaves
    such a `jr` as it is.
  - `ld de,2^k / call ??mul16` is made shifts only up to 64. Beyond that the
    shifts are longer than the call, and could put a relative jump already
    in the code out of reach.
- **Dead-store elimination** did not count `NAME: EQU PARAM`, the form
  uplm80 writes for `AT`, as a use of PARAM. It did catch the column-1 form
  `NAME equ PARAM`. It also did not count a sixteen-bit load from the byte
  before the stored one.
- **Dead-store elimination removed stores that something else could read.**
  It now removes a store only to storage that the module defines (`ds`,
  `db` or `dw`) and does not export. 0.2.4 removed `ld (BUF+1),a` when BUF
  was any of these:
  - set with `equ`, which can be an address outside the program, or
    another name for the same bytes;
  - exported with `public` or `BUF::`;
  - not defined in the module.

  It also removed a store when the load of the same byte was written in a
  different way, such as `ld a,(BUF + 13)` or `ld a,(BUF+0DH)`.
- **Under `.radix 16`, `ld de,64 / call ??mul16` became six shifts.** Under
  that radix, 64 is 100. A constant is now taken only where its value is the
  same under any radix: a single digit, or a number with the suffix H, O or
  Q. That is the case wherever the text sets a radix other than ten.

### Changed

- **Before an exit, a rewrite is no longer made.** That applies where the
  flags or registers it changes reach a call out of the module, a `ret` from
  an exported routine, `jp 0` or the end of the text. What is there may read
  them, and the text does not say whether it does. For uplm80 this is
  mostly `call 5`, which reads no flags. Over the MP/M II and 80un PL/M
  sources, which are 92 compiles, uplm80 0.3.6 at `-O2` now gets fewer of
  these rewrites than with 0.2.4:

  | Rewrite | Fewer |
  |---|---|
  | `cp 0` → `or a` | 261 |
  | `ld a,0` → `xor a` | 115 |
  | `xor_a_store` | 95 |
  | increments in memory | 41 |
  | `ld l,a / ld h,0` before `cp` removed | 12 |
  | `djnz` | 4 |
  | `ld a,(hl) / ld e,a` → `ld e,(hl)` | 2 |

  Other changes offset these:
  - 750 more jumps become relative.
  - 140 more `ld hl,n / ld r,l` are shortened, because the check that HL is
    dead now sees past branches, calls and returns.

  The code is 479 bytes smaller than with 0.2.4 (172,880 against 173,359),
  and 213 bytes smaller with uplm80's fix/expression-types branch. At `-O1`
  and `-O3` it is smaller by 132 to 487 bytes. PIP grows the most: 15 bytes
  at `-O2` (23 with fix/expression-types), and up to 63 at `-O3`.
- 0.2.4 made 22,679 rewrites on those sources, over `-O1` to `-O3` and both
  uplm80 versions. None of them changed a register or flag that the text
  shows being read, and none wrote an instruction the Z80 does not have or a
  relative jump out of reach. So every rewrite this release no longer makes
  there changes code size only. Every output assembles as it did (the same
  10 of the 92 fail, for uplm80's own reasons).
- Relative jumps and `djnz` are made last, after dead-store elimination,
  since they count bytes and nothing may lengthen the code after them. The
  patterns run once more after jump threading, before dead-store
  elimination.
- `ld hl,n / ld r,l` becomes `ld r,n` only where n's value is in the text:
  a number, or a name the text sets with `equ`. `ld hl,NAME / ld a,l` became
  `ld a,NAME`, a one-byte field holding part of a relocatable or external
  address. um80 and ul80 fill that in with the low byte, so for uplm80 it
  was right, but an assembler without byte relocation rejects it. The
  constant is kept as written when it is 0 to 255. Otherwise its low byte is
  written in hexadecimal with the suffix H (`ld hl,299 / ld a,l` →
  `ld a,02Bh`), which has the same value under any `.radix`. 0.2.4 wrote
  `ld a,299`. um80 assembles that as the low byte without an error or
  warning, but it is out of range, and other assemblers need not accept it.
- About four times faster than 0.2.4 on the corpus. Where a pass has
  already found a register or flag dead or live from a line, it does not
  look again. So straight code with no write to what is asked about now
  costs time in proportion to its length. 3000 copies of `ld a,0 /
  ld (v),a` take 3.9 s, where 0.2.4 takes 13.7 s, timed one after the
  other on the same machine.
- No tail call is made to a routine of the text whose stack height is not
  known at its `ret`. There are two such routines on the corpus: MP/M LOAD's
  BOOT, which restores SP from a saved copy and returns, and uplm80's
  REENTRANT procedures, which restore SP from IX. The tail calls would be
  correct, but the optimizer cannot tell them from calls of a routine that
  looks under its return address. This costs 30 tail calls, of one byte
  each, over the corpus's six sets of compiles. It also costs one `cp 0`
  → `or a` in ED at `-O3`, after a call of a routine that jumps into code
  that sets SP.
- `PeepholePattern` takes `clobbers`, the registers and flags a replacement
  leaves different. It also takes `dead_from`, the instruction of the match
  from which they must be dead. A custom pattern is applied only where they
  are dead.

### Added

- `upeepz80/z80.py`: what each Z80 instruction reads and writes, where
  control goes after it, and its size. Its sizes agree with um80's for every
  form it recognises (`tests/test_z80_model.py`).
- `tests/z80sim.py`, a Z80 interpreter that runs assembly text. On 1,500
  random sequences it agrees with cpmemu's Z80 core on every register, the
  six flags and memory (`tests/simcheck.py`).
- `tests/peepfuzz.py`, a differential tester. It builds random programs
  around every shape the optimizer rewrites: straight-line code, branches,
  loops, calls and returns, pushes and pops, conditional calls, routines
  that return to a pushed address, and routines that swap their return
  address for another. It runs each program before and after optimization
  from random states, and compares every register, flag and byte of
  memory. Where a program exports a routine with `::`, it also calls that
  routine from outside, as another module would. 0.2.4 differs on 214 of
  the first 300 programs. This release differs on none of 7,500: 6,000 of
  the usual length and 1,500 two and a half times as long.
  `tests/test_fuzz.py` runs 300 of them in the suite.
- Regression tests for each defect above: `tests/test_liveness.py`,
  `test_instructions.py`, `test_relative_jumps.py` and
  `test_control_flow.py`. Each test runs the code before and after
  optimization, or checks its output. 70 of their 75 tests fail on 0.2.4.
  Of the other five, two check that a rewrite is still made where it may
  be. One checks that the new call-following does not skip the path where
  a conditional call is not made. Two check that `NAME::` is taken for an
  exported label, which 0.2.4, following no calls, did not need.

### Known issues

- **Dead-store elimination rests on what PL/M's storage allocation
  guarantees.** It removes a parameter's store at a procedure's entry when
  the location is storage the module defines and does not export, and no
  line of the module names the location other than to store to it. Code
  could still read the location through an address computed from another
  name. In uplm80's output such a read belongs to another procedure's
  overlaid locals, which that procedure writes before it reads. The text
  alone does not show that.
- **A compiler cannot yet say what the code outside its text reads.** A
  PL/M program's BDOS calls and calls between its modules read no flags,
  and saying so would make back most of the rewrites listed under Changed.
- **uplm80 0.3.6 fails one of its own 112 tests with this release.** The
  test is
  `TestFoldedRelationalMatchesTheRuntimeValue::test_every_level_computes_it_the_same_way`.
  It requires `-O0`, which runs
  no peephole, to write the same instructions as `-O1` to `-O3`. This
  release makes `ld hl,1 / ld a,l` into `ld a,1` there, which is correct:
  HL is dead, because `ld l,a / ld h,0` follows. 0.2.4 did not look past
  the branch in between. uplm80's fix/expression-types branch changes the
  test. uplm80 0.3.6 asks for `upeepz80>=0.2.4`, so release this after
  uplm80 0.3.7. Otherwise 0.3.6's tests fail when it is installed with this
  release.
- **The stack is taken to be used only as calls and pushes use it.** A
  called routine is taken to return to its call with the stack as it was.
  The exceptions are routines that the text shows taking their return
  address off the stack, or returning where the stack height is not known.
  The optimizer cannot see a routine outside the text do this. Nor can it
  see a routine change its return address through a pointer (`ld hl,0 /
  add hl,sp / ld (hl),e`).
  - For the same reason, `call x / ret` → `jp x` assumes that x does not
    read the stack above its return address. uplm80's procedures that take
    parameters on the stack do read it (`ld hl,2 / add hl,sp`). But their
    callers pop the parameters after the call, so such a call is not
    followed by `ret`.
  - Memory below SP is taken to be unused. `push hl / pop hl` is removed,
    and with it the copy of HL that it leaves below SP.
  - Code that reads or writes its own instructions is not supported.
- **A `.radix` other than ten affects the whole text.** Where the text has
  one anywhere, even before the directive, only numbers whose value is the
  same under every radix are known: a single digit, or a number with the
  suffix H, O or Q. No other number is rewritten. A `ds` of such a number
  ends a segment where relative jumps are measured.
