#!/usr/bin/env python3
"""LoomQ submission adapter contract v1.0.

L1: unified intermediate layer. One QASM 2.0 parser + dependency-free simulator
     + three target-IR renderers (spinq / originq / braket).
L2: optional natural-language agent (see l2_agent.py, reads LOOMQ_LLM_*).
L3: optional hybrid compiler (see l3_compiler.py, emits RISC-V assembly).

The organizer extracts starter_kit/ as the evaluation root, so all imports stay
relative and dependency-free.
"""

from typing import Any, Dict, List, Tuple

try:
    from .loomq_core import parse_qasm, sample_counts, transpile_to_target, build_result
except ImportError:  # running as a top-level script (evaluator imports `adapter`)
    from loomq_core import parse_qasm, sample_counts, transpile_to_target, build_result

try:
    from .l2_agent import agent_chat  # L2 (optional)
except (ImportError, AttributeError):  # pragma: no cover
    try:
        from l2_agent import agent_chat  # top-level script run
    except ImportError:
        def agent_chat(prompt: str) -> str:  # type: ignore[no-redef]
            raise NotImplementedError("L2 is optional; implement agent_chat(prompt) to enter")


try:
    from .l3_compiler import compile_hybrid  # L3 (optional)
except (ImportError, AttributeError):  # pragma: no cover
    try:
        from l3_compiler import compile_hybrid  # top-level script run
    except ImportError:
        def compile_hybrid(hybrid_qasm_str: str) -> Tuple[List[str], str]:  # type: ignore[no-redef]
            raise NotImplementedError(
                "L3 is optional; implement compile_hybrid(hybrid_qasm_str) to enter"
            )


SUPPORTED_TARGETS = ("spinq", "originq", "braket")


def transpile(qasm_str: str, target: str) -> str:
    """Translate OpenQASM 2.0 into the target backend's native representation."""
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"unsupported target {target!r}; expected one of {SUPPORTED_TARGETS}")
    circuit = parse_qasm(qasm_str)
    return transpile_to_target(circuit, target)


def run(qasm_str: str, target: str, shots: int) -> Dict[str, Any]:
    """Execute a circuit and return the unified result schema from the rules."""
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"unsupported target {target!r}; expected one of {SUPPORTED_TARGETS}")
    circuit = parse_qasm(qasm_str)
    counts = sample_counts(circuit, shots)
    return build_result(qasm_str, target, shots, counts)
