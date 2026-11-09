"""
upeepz80 - Universal Peephole Optimizer for Z80

A language-agnostic optimization library for compilers targeting
the Zilog Z80 processor with pure Z80 assembly input/output.

It does not accept 8080 mnemonics (MOV, MVI, etc.). upeep80, which
accepted them, is retired.
"""

__version__ = "0.2.8"
__author__ = "upeepz80 project"

from .peephole import (
    PeepholeOptimizer,
    PeepholePattern,
    optimize,
)

__all__ = [
    # Version
    "__version__",

    # Peephole Optimization
    "PeepholeOptimizer",
    "PeepholePattern",
    "optimize",
]
