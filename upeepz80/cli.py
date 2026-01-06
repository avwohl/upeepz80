"""Command-line interface for upeepz80."""

import argparse
import sys

from . import __version__, optimize


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="upeepz80",
        description="Universal peephole optimizer for Z80 assembly",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="Input assembly file (default: stdin)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Output file (default: stdout)",
    )

    args = parser.parse_args()

    asm_text = args.input.read()
    optimized = optimize(asm_text)
    args.output.write(optimized)

    return 0


if __name__ == "__main__":
    sys.exit(main())
