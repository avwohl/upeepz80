# No-optimization regions and disabled rewrites

A compiler can ask upeepz80 to leave a piece of code alone, or to stop one kind of rewrite. A debugger needs both. The design is in [debugger_plan.md](https://github.com/avwohl/cpmemu/blob/main/docs/debugger_plan.md) of cpmemu (D6). The marker format is in `docs/debug_info.md` of um80_and_friends.

## Regions of no optimization

The compiler writes a `;#noopt begin` comment line and a `;#noopt end` comment line around the code:

```
	;#noopt begin
	call	G
	ret
	;#noopt end
```

`optimize()` does not change a line between the two markers. No pattern changes the line. The Z80 rewrites and the jump threading leave the line as it is. The removal of a jump to the next label and the dead store at the entry of a procedure leave the line as it is too. A `jp` in a region stays a `jp`, and `dec b` with `jp nz` does not become `djnz`. A label in a region stays, also when no jump names the label.

The code around a region changes as before, with one rule. The liveness analysis treats a region as code that reads every register and every flag. A rewrite before a region keeps each value that the region could read. A call from a region to a procedure outside the region is still a call, so the analysis still finds the procedure.

A region can hold a region. A `;#noopt begin` with no `;#noopt end` runs to the end of the text. A `;#noopt end` with no `;#noopt begin` has no effect. Text with no `;#noopt` marker gives the same output as before: the 116 `--no-optimize` texts of the uc80 examples and tests give the same bytes with the old and the new package.

## Disable a rewrite

`PeepholeOptimizer(disable={...})` and `optimize(text, disable={...})` stop the rewrites by name. A name is the `name` of a pattern, for example `tail_call`. A name can also be a key of the `stats` dictionary. The keys are `jr_convert`, `djnz`, `jump_thread`, `jump_to_next`, `dw_thread`, `dead_label_removed` and `dead_store_elim`. The name of a Z80 rewrite, for example `subde_zero`, is also a key.

```python
from upeepz80 import PeepholeOptimizer

out = PeepholeOptimizer(disable={"tail_call"}).optimize(text)
```

upeepz80 ignores a name that no rewrite has. With `disable` empty, the output is the same as before.

`tail_call` changes `call X` and `ret` to `jp X`. The change removes the frame of the caller, so a backtrace does not show the caller. Use `disable={"tail_call"}` for code that a debugger steps through.

## Tests

`tests/test_noopt.py` tests the regions (kept to the letter, nested, open, with markers inside, a label that outside code jumps to, a `jp` that stays) and the disabled names. `tests/conftest.py` passes `disable` to the second run of each test, the run with markers.
