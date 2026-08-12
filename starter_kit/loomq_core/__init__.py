"""LoomQ unified core: OpenQASM 2.0 parsing, simulation and target-IR rendering.

This package is the single intermediate layer (通用中间层) shared by all three
L1 backends. It is intentionally dependency-free so the organizer's isolated
build environment can always run it: parsing and state-vector simulation are
pure Python, and every backend renderer emits the contract-listed IR subset.
"""

from .parser import parse_qasm, Gate, Measure, Circuit
from .simulator import sample_counts
from .transpiler import transpile_to_target
from .result import build_result

__all__ = [
    "parse_qasm",
    "Gate",
    "Measure",
    "Circuit",
    "sample_counts",
    "transpile_to_target",
    "build_result",
]
