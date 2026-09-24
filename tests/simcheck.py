"""Check tests/z80sim.py against a real Z80 core.

    python3 tests/simcheck.py [--seeds N] [--first S]

Random sequences of register, flag and memory instructions - the kinds the
peephole fuzzer generates, minus access through HL, BC and DE, whose memory
contents the emulator does not share with the interpreter - are assembled
with um80, linked with ul80 and run under cpmemu from a known state; the
program prints its final registers, flags and variables.  The same sequence
runs in the interpreter from the same state, and the two must agree.

Needs um80, ul80 and cpmemu (CPMEMU=path, or on PATH).
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from z80sim import FLAG_MASK, Machine  # noqa: E402
from peepfuzz import Gen, VARS  # noqa: E402

HARNESS = """\
\t.z80
V0\tequ\t8000h
V1\tequ\t8002h
V2\tequ\t8004h
V3\tequ\t8006h
V4\tequ\t8008h
V5\tequ\t800ah
V6\tequ\t800ch
V7\tequ\t800eh
FRAME\tequ\t9000h
\tld\tsp,0c000h
{meminit}
\tld\tix,09080h
\tld\thl,0{f_a:04x}h
\tpush\thl
\tpop\taf
\tld\tbc,0{bc:04x}h
\tld\tde,0{de:04x}h
\tld\thl,0{hl:04x}h
{body}
\tld\t(SAVE+10),sp
\tld\tsp,SAVE+10
\tpush\taf
\tpush\tbc
\tpush\tde
\tpush\thl
\tpush\tix
\tld\tsp,0c000h
\tld\thl,SAVE
\tld\tb,12
??p1:\tcall\tHEXB
\tinc\thl
\tdjnz\t??p1
\tld\thl,8000h
\tld\tb,16
??p2:\tcall\tHEXB
\tinc\thl
\tdjnz\t??p2
\tld\thl,9070h
\tld\tb,32
??p3:\tcall\tHEXB
\tinc\thl
\tdjnz\t??p3
\tjp\t0
HEXB:\tpush\tbc
\tpush\thl
\tld\ta,(hl)
\tpush\taf
\trrca
\trrca
\trrca
\trrca
\tcall\tHEXN
\tpop\taf
\tcall\tHEXN
\tld\te,' '
\tld\tc,2
\tcall\t5
\tpop\thl
\tpop\tbc
\tret
HEXN:\tand\t0fh
\tadd\ta,90h
\tdaa
\tadc\ta,40h
\tdaa
\tld\te,a
\tld\tc,2
\tcall\t5
\tret
SAVE:\tds\t12
\tend
"""


def sequence(rng: random.Random, n: int) -> list[str]:
    g = Gen(rng)
    out: list[str] = []
    while len(out) < n:
        ins = g.instr()
        text = " ".join(ins)
        if "(hl)" in text or "(de)" in text or "(bc)" in text or "push" in text or "pop" in text:
            continue
        if any(i.startswith(("ld hl,V", "ld de,V")) for i in ins):
            continue
        out += ins
    return out


def cpmemu() -> str | None:
    path = os.environ.get("CPMEMU") or shutil.which("cpmemu")
    if not path and os.path.exists("/Users/wohl/src/cpmemu/src/cpmemu"):
        path = "/Users/wohl/src/cpmemu/src/cpmemu"
    return path


def one(seed: int, emu: str) -> str | None:
    rng = random.Random(seed)
    body = sequence(rng, rng.randrange(4, 25))
    mem = {a: rng.randrange(256) for a in list(range(0x8000, 0x8010)) + list(range(0x9070, 0x9090))}
    a, f = rng.randrange(256), rng.randrange(256)
    bc, de, hl = (rng.randrange(0x10000) for _ in range(3))
    meminit = "\n".join(f"\tld\ta,{v}\n\tld\t(0{addr:04x}h),a" for addr, v in mem.items())
    src = HARNESS.format(meminit=meminit, f_a=(a << 8) | f, bc=bc, de=de, hl=hl,
                         body="\n".join("\t" + b for b in body))
    with tempfile.TemporaryDirectory() as d:
        mac, rel, com = (os.path.join(d, x) for x in ("T.MAC", "T.REL", "T.COM"))
        with open(mac, "w") as fh:
            fh.write(src)
        for cmd in (["um80", "-o", rel, mac], ["ul80", "-o", com, rel]):
            r = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if r.returncode:
                return f"seed {seed}: {cmd[0]} failed: {r.stdout}{r.stderr}\n" + "\n".join(body)
        r = subprocess.run([emu, com], capture_output=True, text=True, timeout=20, check=False)
    got = [int(x, 16) for x in r.stdout.split()]
    if len(got) != 12 + 16 + 32:
        return f"seed {seed}: emulator printed {r.stdout!r} {r.stderr[-200:]!r}"
    # SAVE: ix(2) hl(2) de(2) bc(2) af(2) sp(2), little-endian pushes from SAVE+10 down
    ix = got[0] | got[1] << 8
    hl_ = got[2] | got[3] << 8
    de_ = got[4] | got[5] << 8
    bc_ = got[6] | got[7] << 8
    af_ = got[8] | got[9] << 8
    emu_state = {"a": af_ >> 8, "f": af_ & FLAG_MASK, "b": bc_ >> 8, "c": bc_ & 0xFF,
                 "d": de_ >> 8, "e": de_ & 0xFF, "h": hl_ >> 8, "l": hl_ & 0xFF,
                 "ix": ix, "vars": bytes(got[12:28]), "frame": bytes(got[28:60])}

    m = Machine("\n".join("\t" + b for b in body) + "\n\tret\n", dict(VARS, FRAME=0x9000))
    m.r.update(a=a, b=bc >> 8, c=bc & 0xFF, d=de >> 8, e=de & 0xFF, h=hl >> 8, l=hl & 0xFF)
    m.f = f
    m.ix = 0x9080
    m.sp = 0xC000
    for addr, v in mem.items():
        m.mem[addr] = v
    m.run()
    sim_state = dict(m.r, f=m.f & FLAG_MASK, ix=m.ix, vars=bytes(m.mem[0x8000:0x8010]),
                     frame=bytes(m.mem[0x9070:0x9090]))
    diffs = [f"{k}: emu {emu_state[k]!r} sim {sim_state[k]!r}" for k in emu_state if emu_state[k] != sim_state[k]]
    if diffs:
        return f"seed {seed}: " + "; ".join(diffs) + "\n" + "\n".join(body)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--first", type=int, default=1)
    args = ap.parse_args()
    emu = cpmemu()
    if not emu:
        print("no cpmemu")
        return 2
    bad = 0
    for seed in range(args.first, args.first + args.seeds):
        rep = one(seed, emu)
        if rep:
            bad += 1
            print(rep)
    print(f"{bad} of {args.seeds} sequences differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
