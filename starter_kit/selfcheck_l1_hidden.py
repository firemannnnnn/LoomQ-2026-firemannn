#!/usr/bin/env python3
"""L1 hidden-circuit regression: QFT-4 / Grover-3 / Random-Circuit x3.

The official evaluator generates these hidden circuits from a private seed and
scores the *transpiled* IR via noise-free simulation with Hellinger fidelity
>= 0.97 (contest rule). We cannot replicate the organizer's private seed, but we
can prove the intermediate layer is generic:

  1. Build the hidden circuit *types* using only the 12-gate whitelist.
  2. Cross-validate loomq_core's simulator against an independent NumPy
     reference simulator (different code path: dense Kronecker matrices vs.
     in-place bit-loop updates).
  3. Sample with the contest bit-order and check Hellinger fidelity >= 0.97.
  4. For every target: transpile -> parse the target IR back to gates ->
     re-simulate -> fidelity >= 0.97 (proves IR is semantically equivalent).
  5. Validate the unified result schema returned by adapter.run().

Usage:  python selfcheck_l1_hidden.py  (run from starter_kit/ or repo root)
"""

import math
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adapter import run, transpile  # noqa: E402

FIDELITY_THRESHOLD = 0.97
SHOTS = 8192

# ---------------------------------------------------------------------------
# Hidden-circuit generators (12-gate whitelist only)
# ---------------------------------------------------------------------------


def qft4_qasm() -> str:
    return """OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
cu1(pi/2) q[1], q[0];
h q[1];
cu1(pi/4) q[2], q[0];
cu1(pi/2) q[2], q[1];
h q[2];
cu1(pi/8) q[3], q[0];
cu1(pi/4) q[3], q[1];
cu1(pi/2) q[3], q[2];
h q[3];
swap q[0], q[3];
swap q[1], q[2];
measure q -> c;
"""


def grover3_qasm(marked: str = "101") -> str:
    """3-qubit Grover with one iteration. Oracle flips the marked basis state
    (phase kickback through a CCZ realized as H-CCX-H on the target qubit)."""
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        "qreg q[3];",
        "creg c[3];",
        "h q[0];",
        "h q[1];",
        "h q[2];",
    ]
    # Oracle: flip |s> phase. Marked bits that are 0 get wrapped in X.
    assert len(marked) == 3 and set(marked) <= {"0", "1"}
    for i, bit in enumerate(marked):
        if bit == "0":
            lines.append(f"x q[{i}];")
    lines += ["h q[2];", "ccx q[0], q[1], q[2];", "h q[2];"]
    for i, bit in enumerate(marked):
        if bit == "0":
            lines.append(f"x q[{i}];")
    # Diffusion operator: H^3 X^3 CCZ X^3 H^3
    lines += ["h q[0];", "h q[1];", "h q[2];"]
    lines += ["x q[0];", "x q[1];", "x q[2];"]
    lines += ["h q[2];", "ccx q[0], q[1], q[2];", "h q[2];"]
    lines += ["x q[0];", "x q[1];", "x q[2];"]
    lines += ["h q[0];", "h q[1];", "h q[2];"]
    lines.append("measure q -> c;")
    return "\n".join(lines) + "\n"


_RANDOM_GATES = [
    ("h", 1), ("x", 1), ("s", 1), ("sdg", 1), ("t", 1), ("tdg", 1),
    ("rz", 1), ("ry", 1), ("cx", 2), ("cu1", 2), ("swap", 2), ("ccx", 3),
]


def random_qasm(n: int, seed: int, depth: int = 32) -> str:
    """Random circuit from the 12-gate whitelist (deterministic for a seed)."""
    rng = random.Random(seed)
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{n}];",
        f"creg c[{n}];",
    ]
    for _ in range(depth):
        name, arity = rng.choice(_RANDOM_GATES)
        qs = rng.sample(range(n), arity)
        args = ", ".join(f"q[{q}]" for q in qs)
        if name in ("rz", "ry", "cu1"):
            theta = rng.uniform(-math.pi, math.pi)
            lines.append(f"{name}({theta}) {args};")
        else:
            lines.append(f"{name} {args};")
    lines.append("measure q -> c;")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Independent reference simulator (dense Kronecker matrices, NumPy)
# ---------------------------------------------------------------------------

_SINGLE = {
    "h": np.array([[1, 1], [1, -1]]) / math.sqrt(2),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
    "t": np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex),
    "tdg": np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=complex),
}
_CNOT = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def _rz(theta):
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])


def _ry(theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _cu1(theta):
    return np.diag([1, 1, 1, np.exp(1j * theta)])


def _toffoli():
    g = np.eye(8, dtype=complex)
    # row/col index = (target<<2)|(q[b]<<1)|q[a]; flip |011> <-> |111> (a,b control)
    g[3, 7] = g[7, 3] = 1
    g[3, 3] = g[7, 7] = 0
    return g


def _two_qubit_matrix(name, params):
    if name == "cx":
        return _CNOT
    if name == "swap":
        return _SWAP
    if name == "cu1":
        return _cu1(params[0])
    raise ValueError(name)


def _lift_single(U, k, n):
    """Kronecker lift of U acting on qubit k (bit k <-> q[k])."""
    lo, hi = 1 << k, 1 << (n - 1 - k)
    return np.kron(np.eye(hi), np.kron(U, np.eye(lo)))


def _lift_two(G, p, q, n):
    """Lift a 4x4 gate G on qubits (p, q) in circuit order (p = first operand).
    G row index = (bit_q<<1)|bit_p, i.e. q is the high bit."""
    P = np.zeros((1 << n, 1 << n), dtype=complex)
    mp, mq = 1 << p, 1 << q
    for i in range(1 << n):
        x = (((i & mq) >> q) << 1) | ((i & mp) >> p)
        rest = i & ~(mp | mq)
        for y, v in enumerate(G[:, x]):
            if v == 0:
                continue
            j = rest | ((y & 1) << p) | (((y >> 1) & 1) << q)
            P[j, i] = v
    return P


def _lift_three(G, a, b, c, n):
    """Lift an 8x8 gate on qubits a, b (controls) and c (target).
    G row index = (bit_c<<2)|(bit_b<<1)|bit_a."""
    P = np.zeros((1 << n, 1 << n), dtype=complex)
    ma, mb, mc = 1 << a, 1 << b, 1 << c
    for i in range(1 << n):
        x = (((i & mc) >> c) << 2) | (((i & mb) >> b) << 1) | ((i & ma) >> a)
        rest = i & ~(ma | mb | mc)
        for y, v in enumerate(G[:, x]):
            if v == 0:
                continue
            j = rest | ((y & 1) << a) | (((y >> 1) & 1) << b) | (((y >> 2) & 1) << c)
            P[j, i] = v
    return P


def simple_parse(qasm: str):
    """Tiny independent QASM line parser -> (n, gates, measures).

    gates = [(name, (qubits...), (params...))]; measures = [(qidx, cidx)].
    Deliberately shares no code with loomq_core.parser.
    """
    n = 0
    gates = []
    measures = []
    for raw in qasm.splitlines():
        line = raw.strip()
        if not line or line.startswith("OPENQASM") or line.startswith("include"):
            continue
        m = re.match(r"qreg\s+(\w+)\s*\[\s*(\d+)\s*\]", line)
        if m:
            n = int(m.group(2))
            continue
        m = re.match(r"creg\s+(\w+)\s*\[\s*(\d+)\s*\]", line)
        if m:
            continue
        m = re.match(r"measure\s+(q)\s*->\s*(c)\s*;", line)
        if m:
            measures = [(i, i) for i in range(n)]
            continue
        m = re.match(r"measure\s+q\[(\d+)\]\s*->\s*c\[(\d+)\]", line)
        if m:
            measures.append((int(m.group(1)), int(m.group(2))))
            continue
        m = re.match(r"(\w+)\s*(?:\(\s*([^)]*)\s*\))?\s+((?:q\[\d+\](?:\s*,\s*)?)+);?", line)
        if m:
            name = m.group(1).lower()
            params = []
            if m.group(2):
                for tok in m.group(2).split(","):
                    tok = tok.strip().replace("pi", str(math.pi))
                    try:
                        params.append(float(tok))
                    except ValueError:
                        params.append(float(eval(tok, {"__builtins__": {}}, {"pi": math.pi})))
            qs = tuple(int(x) for x in re.findall(r"q\[(\d+)\]", m.group(3)))
            gates.append((name, qs, tuple(params)))
            continue
        raise ValueError(f"simple_parse cannot handle: {line!r}")
    return n, gates, measures


def simulate_numpy(gates, n):
    """Dense state-vector simulation via Kronecker-lifted operators."""
    state = np.zeros(1 << n, dtype=complex)
    state[0] = 1.0
    for name, qs, params in gates:
        if name in _SINGLE:
            state = _lift_single(_SINGLE[name], qs[0], n) @ state
        elif name == "rz":
            state = _lift_single(_rz(params[0]), qs[0], n) @ state
        elif name == "ry":
            state = _lift_single(_ry(params[0]), qs[0], n) @ state
        elif name in ("cx", "swap", "cu1"):
            state = _lift_two(_two_qubit_matrix(name, params), qs[0], qs[1], n) @ state
        elif name == "ccx":
            state = _lift_three(_toffoli(), qs[0], qs[1], qs[2], n) @ state
        else:
            raise ValueError(f"reference simulator: unsupported gate {name!r}")
    return state


def probs_dict(state, n):
    """Measured-bit distribution as {bitstring: prob} (rightmost char = q[0])."""
    probs = np.abs(state) ** 2
    return {format(i, f"0{n}b"): float(p) for i, p in enumerate(probs) if p > 0}


def hellinger_fidelity(p_counts, q_probs):
    """Contest formula: Fidelity = 1 - (1/sqrt(2))*sqrt(sum((sqrt(p)-sqrt(q))^2))."""
    states = set(p_counts) | set(q_probs)
    h2 = sum(
        (math.sqrt(p_counts.get(s, 0.0)) - math.sqrt(q_probs.get(s, 0.0))) ** 2
        for s in states
    )
    return 1.0 - math.sqrt(h2 / 2.0)


# ---------------------------------------------------------------------------
# Reverse-parse the transpiled target IR back to gate tuples
# ---------------------------------------------------------------------------


def reverse_parse_ir(ir: str, target: str):
    """Parse transpile() output back to (n, gates, measures) for re-simulation."""
    if target == "spinq":
        return simple_parse(ir)
    if target == "braket":
        return _braket_ir(ir)
    if target == "originq":
        return _originq_ir(ir)
    raise ValueError(target)


def _braket_ir(ir: str):
    n = None
    gates = []
    measures = []
    _REV = {"h": "h", "x": "x", "s": "s", "sdg": "sdg", "t": "t", "tdg": "tdg",
            "rz": "rz", "ry": "ry", "swap": "swap", "cnot": "cx",
            "ccnot": "ccx", "cphaseshift": "cu1"}
    for raw in ir.splitlines():
        line = raw.strip()
        m = re.match(r"qubit\[(\d+)\]", line)
        if m:
            n = int(m.group(1))
            continue
        m = re.match(r"c\[(\d+)\]\s*=\s*measure\s+q\[(\d+)\]", line)
        if m:
            measures.append((int(m.group(2)), int(m.group(1))))
            continue
        if line.startswith("c = measure"):
            continue
        m = re.match(r"(\w+)\s*(?:\(\s*([^)]+)\s*\))?\s+((?:q\[\d+\](?:\s*,\s*)?)+);", line)
        if m:
            name = _REV.get(m.group(1).lower())
            if name is None:
                raise ValueError(f"braket IR: unknown gate {m.group(1)!r}")
            params = tuple(float(t) for t in m.group(2).split(",")) if m.group(2) else ()
            qs = tuple(int(x) for x in re.findall(r"q\[(\d+)\]", m.group(3)))
            gates.append((name, qs, params))
            continue
    if n is None:
        raise ValueError("braket IR: missing qubit declaration")
    return n, gates, measures


def _originq_ir(ir: str):
    n = None
    gates = []
    measures = []
    _REV = {"H": "h", "X": "x", "S": "s", "SDAG": "sdg", "T": "t", "TDAG": "tdg",
            "RZ": "rz", "RY": "ry", "CNOT": "cx", "CU1": "cu1",
            "SWAP": "swap", "TOFFOLI": "ccx"}
    for raw in ir.splitlines():
        line = raw.strip()
        m = re.match(r"QINIT\s+(\d+)", line)
        if m:
            n = int(m.group(1))
            continue
        m = re.match(r"MEASURE\s+q\[(\d+)\],\s*c\[(\d+)\]", line)
        if m:
            measures.append((int(m.group(1)), int(m.group(2))))
            continue
        m = re.match(r"(\w+)\s*(?:\(\s*([^)]+)\s*\))?\s+((?:q\[\d+\](?:\s*,\s*)?)+)", line)
        if m:
            name = _REV.get(m.group(1))
            if name is None:
                raise ValueError(f"originq IR: unknown gate {m.group(1)!r}")
            params = tuple(float(t) for t in m.group(2).split(",")) if m.group(2) else ()
            qs = tuple(int(x) for x in re.findall(r"q\[(\d+)\]", m.group(3)))
            gates.append((name, qs, params))
            continue
    if n is None:
        raise ValueError("originq IR: missing QINIT")
    return n, gates, measures


# ---------------------------------------------------------------------------
# Schema validation (mirrors evaluator.py)
# ---------------------------------------------------------------------------


def validate_schema(payload):
    if not isinstance(payload, dict):
        return False, "result must be a dict"
    required = ("backend", "job_id", "shots", "counts", "bit_order", "timestamp")
    missing = [f for f in required if f not in payload]
    if missing:
        return False, "missing fields: " + ", ".join(missing)
    counts = payload["counts"]
    if not isinstance(counts, dict) or not counts:
        return False, "counts must be non-empty"
    if not all(set(k) <= {"0", "1"} for k in counts):
        return False, "counts keys must be binary strings"
    if sum(counts.values()) != payload["shots"]:
        return False, "counts total must equal shots"
    if payload["bit_order"] != "little":
        return False, "bit_order must be little"
    if payload.get("meta", {}).get("is_mock"):
        return False, "mock results never pass"
    return True, "ok"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_hidden_case(name, qasm, targets):
    results = []
    n, gates, measures = simple_parse(qasm)

    # 1) Reference (NumPy) distribution = theoretical ideal.
    state = simulate_numpy(gates, n)
    ideal = probs_dict(state, n)

    # 2) Cross-validate loomq_core against the reference simulator.
    from loomq_core import parse_qasm, sample_counts
    from loomq_core.simulator import run_gates
    core_state = run_gates(parse_qasm(qasm))
    core_probs = probs_dict(core_state, n)
    shared = set(ideal) | set(core_probs)
    max_dev = max(abs(ideal.get(s, 0.0) - core_probs.get(s, 0.0)) for s in shared)
    cross = max_dev < 1e-9
    results.append(f"    cross-validate loomq_core vs NumPy: max|dp|={max_dev:.2e} -> {'OK' if cross else 'FAIL'}")

    # 3) Contest sampling: fidelity of 8192-shot counts vs ideal.
    counts = sample_counts(parse_qasm(qasm), SHOTS, seed=42)
    counts_probs = {k: v / SHOTS for k, v in counts.items()}
    fid = hellinger_fidelity(counts_probs, ideal)
    results.append(f"    sample 8192 Fidelity: {fid:.4f} -> {'PASS' if fid >= FIDELITY_THRESHOLD else 'FAIL'}")

    # 4) Transpile -> reverse-parse IR -> re-simulate (semantic equivalence).
    for target in targets:
        ir = transpile(qasm, target)
        if not ir.strip():
            results.append(f"    transpile[{target}]: EMPTY IR -> FAIL")
            continue
        n2, gates2, _m2 = reverse_parse_ir(ir, target)
        state2 = simulate_numpy(gates2, n2)
        ir_probs = probs_dict(state2, n2)
        ir_fid = hellinger_fidelity(ir_probs, ideal)  # deterministic -> 1.0 if equal
        results.append(
            f"    IR[{target}] {len(gates2)} gates semantic fidelity: "
            f"{ir_fid:.6f} -> {'PASS' if ir_fid >= 0.999 else 'FAIL'}"
        )

        # 5) Unified result schema + counts fidelity through adapter.run().
        payload = run(qasm, target, SHOTS)
        valid, why = validate_schema(payload)
        run_probs = {k: v / SHOTS for k, v in payload["counts"].items()}
        run_fid = hellinger_fidelity(run_probs, ideal)
        ok = valid and run_fid >= FIDELITY_THRESHOLD
        results.append(
            f"    run[{target}] schema={'ok' if valid else why}, "
            f"counts Fidelity={run_fid:.4f} -> {'PASS' if ok else 'FAIL'}"
        )

    all_ok = all(not line.rstrip().endswith("FAIL") for line in results)
    status = "PASS" if all_ok else "FAIL"
    print(f"[{status}] {name}")
    for line in results:
        print(line)
    return status


def main():
    targets = ("spinq", "originq", "braket")
    cases = [
        ("QFT-4", qft4_qasm()),
        ("Grover-3(marked=101)", grover3_qasm("101")),
        ("Random-Circuit#1 (4q seed=11)", random_qasm(4, 11, 28)),
        ("Random-Circuit#2 (5q seed=22)", random_qasm(5, 22, 32)),
        ("Random-Circuit#3 (4q seed=33)", random_qasm(4, 33, 40)),
    ]
    passed = 0
    for name, qasm in cases:
        status = run_hidden_case(name, qasm, targets)
        if status == "PASS":
            passed += 1
    print(f"\nL1 hidden-circuit regression: {passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
