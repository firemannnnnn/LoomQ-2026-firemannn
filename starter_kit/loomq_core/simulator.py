"""Dependency-free state-vector simulator for the LoomQ 12-gate whitelist.

Bit-order convention (Qiskit-style, mandatory for the contest):
    state index bit k <-> qubit q[k]
    counts key = bin(index) padded to n bits, so the RIGHTMOST char is c[0].
Noise-free deterministic simulator: fidelity of a correct transpile is ~0.99+.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List

from .parser import Circuit, Gate, Measure

_HALF_SQRT2 = 1.0 / math.sqrt(2.0)


def _apply_single(state: List[complex], k: int, matrix) -> None:
    """Apply a 2x2 matrix to qubit k in place. matrix = ((a,b),(c,d))."""
    a, b = matrix[0]
    c, d = matrix[1]
    bit = 1 << k
    for i in range(len(state)):
        if i & bit:
            continue
        j = i | bit
        x, y = state[i], state[j]
        state[i] = a * x + b * y
        state[j] = c * x + d * y


def _apply_cx(state: List[complex], ctl: int, tgt: int) -> None:
    bit_c, bit_t = 1 << ctl, 1 << tgt
    for i in range(len(state)):
        if (i & bit_c) and not (i & bit_t):
            j = i | bit_t
            state[i], state[j] = state[j], state[i]


def _apply_swap(state: List[complex], a: int, b: int) -> None:
    bit_a, bit_b = 1 << a, 1 << b
    for i in range(len(state)):
        if (i & bit_a) != (i & bit_b):
            j = i ^ bit_a ^ bit_b
            if i < j:
                state[i], state[j] = state[j], state[i]


def _apply_ccx(state: List[complex], c1: int, c2: int, t: int) -> None:
    bit1, bit2, bit_t = 1 << c1, 1 << c2, 1 << t
    for i in range(len(state)):
        if (i & bit1) and (i & bit2) and not (i & bit_t):
            j = i | bit_t
            state[i], state[j] = state[j], state[i]


def _apply_cu1(state: List[complex], ctl: int, tgt: int, theta: float) -> None:
    bit_c, bit_t = 1 << ctl, 1 << tgt
    factor = complex(math.cos(theta), math.sin(theta))
    for i in range(len(state)):
        if (i & bit_c) and (i & bit_t):
            state[i] *= factor


def run_gates(circuit: Circuit) -> List[complex]:
    """Simulate all gates (ignoring measures) and return the final state vector."""
    n = circuit.num_qubits
    state: List[complex] = [0j] * (1 << n)
    state[0] = 1.0 + 0j

    for gate in circuit.gates:
        name, qs, ps = gate.name, gate.qubits, gate.params
        if name == "h":
            for q in qs:
                _apply_single(state, q, ((_HALF_SQRT2, _HALF_SQRT2), (_HALF_SQRT2, -_HALF_SQRT2)))
        elif name == "x":
            for q in qs:
                _apply_single(state, q, ((0, 1), (1, 0)))
        elif name == "s":
            for q in qs:
                _apply_single(state, q, ((1, 0), (0, 1j)))
        elif name == "sdg":
            for q in qs:
                _apply_single(state, q, ((1, 0), (0, -1j)))
        elif name == "t":
            phase = complex(math.cos(math.pi / 4), math.sin(math.pi / 4))
            for q in qs:
                _apply_single(state, q, ((1, 0), (0, phase)))
        elif name == "tdg":
            phase = complex(math.cos(math.pi / 4), -math.sin(math.pi / 4))
            for q in qs:
                _apply_single(state, q, ((1, 0), (0, phase)))
        elif name == "rz":
            for q in qs:
                theta = ps[0]
                e1 = complex(math.cos(theta / 2), -math.sin(theta / 2))
                e2 = complex(math.cos(theta / 2), math.sin(theta / 2))
                _apply_single(state, q, ((e1, 0), (0, e2)))
        elif name == "ry":
            for q in qs:
                theta = ps[0]
                c, s = math.cos(theta / 2), math.sin(theta / 2)
                _apply_single(state, q, ((c, -s), (s, c)))
        elif name == "cx":
            _apply_cx(state, qs[0], qs[1])
        elif name == "cu1":
            _apply_cu1(state, qs[0], qs[1], ps[0])
        elif name == "swap":
            _apply_swap(state, qs[0], qs[1])
        elif name == "ccx":
            _apply_ccx(state, qs[0], qs[1], qs[2])
        else:  # pragma: no cover
            raise ValueError(f"unsupported gate {name!r}")
    return state


def sample_counts(
    circuit: Circuit,
    shots: int,
    seed: int | None = None,
) -> Dict[str, int]:
    """Simulate and sample, returning counts keyed per the contest bit-order rule."""
    state = run_gates(circuit)
    n = circuit.num_qubits
    probs = [abs(amp) ** 2 for amp in state]

    # Measure: map measured qubit -> measured value in the final state.
    # If measures are declared, only the measured qubits' bits are projected;
    # unmeasured qubits are traced out (their probabilities marginalised).
    measured_qubits = [m.qubit_index for m in circuit.measures]
    use_qubits = measured_qubits if measured_qubits else list(range(n))

    rng = random.Random(seed)
    counts: Dict[str, int] = {}
    # Build a lookup: for each measured combination, sum probabilities.
    # Fast path: all qubits measured -> index -> bin string.
    if sorted(use_qubits) == list(range(n)):
        for _ in range(shots):
            idx = rng.choices(range(len(state)), weights=probs, k=1)[0]
            key = format(idx, f"0{n}b")
            counts[key] = counts.get(key, 0) + 1
        return counts

    # Partial measurement: marginalise probabilities over unmeasured qubits.
    mask_measured = 0
    for q in use_qubits:
        mask_measured |= 1 << q
    bucket: Dict[int, float] = {}
    for idx, prob in enumerate(probs):
        if prob == 0.0:
            continue
        key = idx & mask_measured
        bucket[key] = bucket.get(key, 0.0) + prob
    keys = list(bucket.keys())
    weights = [bucket[k] for k in keys]
    for _ in range(shots):
        val = rng.choices(keys, weights=weights, k=1)[0]
        # key string: bits ordered from measured qubit with the largest index down
        # to the smallest, so that qubit q[0] lands at the rightmost char.
        chars = []
        for q in sorted(use_qubits, reverse=True):
            chars.append(str((val >> q) & 1))
        key = "".join(chars)
        counts[key] = counts.get(key, 0) + 1
    return counts
