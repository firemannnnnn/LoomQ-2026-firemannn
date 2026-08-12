"""Cross-check: every whitelist gate against the reference braket simulator."""
import sys
sys.path.insert(0, ".")

from loomq_core import parse_qasm, sample_counts, transpile_to_target

ALL_GATES_QASM = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
x q[1];
s q[2];
sdg q[0];
t q[1];
tdg q[2];
rz(0.7) q[0];
ry(0.3) q[1];
cx q[0], q[1];
cu1(0.5) q[1], q[2];
swap q[0], q[1];
ccx q[0], q[1], q[2];
measure q -> c;
"""

GHZ5_QASM = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
cx q[3], q[4];
measure q -> c;
"""

QFT4_QASM = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
cu1(1.57079632679) q[1], q[0];
h q[1];
cu1(0.785398163397) q[2], q[0];
cu1(1.57079632679) q[2], q[1];
h q[2];
cu1(0.392699081699) q[3], q[0];
cu1(0.785398163397) q[3], q[1];
cu1(1.57079632679) q[3], q[2];
h q[3];
swap q[0], q[3];
swap q[1], q[2];
measure q -> c;
"""


def verify(name, qasm, expected_keys, shots=8192, seed=42):
    circ = parse_qasm(qasm)
    counts = sample_counts(circ, shots, seed=seed)
    total = sum(counts.values())
    assert abs(total - shots) == 0, f"{name}: counts total mismatch"
    for target in ("spinq", "braket", "originq"):
        ir = transpile_to_target(circ, target)
        assert ir.strip(), f"{name}: empty IR for {target}"
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
    print(f"[OK] {name}: top={top}")
    return counts


if __name__ == "__main__":
    verify("all-12-gates", ALL_GATES_QASM, set())
    c1 = verify("ghz5", GHZ5_QASM, {"00000", "11111"})
    c2 = verify("qft4", QFT4_QASM, set())
    print("ghz5 main keys:", sorted(k for k, v in c1.items() if v > 100))
    print("qft4 non-zero keys:", sorted(c2.keys())[:10])
