"""End-to-end L3 test: compile Hybrid-QASM and verify with the official RISC-V emulator."""
import sys

sys.path.insert(0, ".")

from riscv_emulator import TinyRISCVEmulator
from l3_compiler import compile_hybrid

CASES = [
    (
        "public-branch",
        """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
measure q[0] -> c[0];
classical { if (c[0] == 1) { r1 = 7; } else { r1 = 3; } }
""",
        [(0, 3), (1, 7)],
    ),
    (
        "if-else-arithmetic",
        """OPENQASM 2.0;
qreg q[2];
creg c[2];
classical {
  if (c[0] == 1) { r1 = 100; } else { r1 = 10; }
  r1 = r1 + 5;
}
""",
        [(0, 15), (1, 105)],
    ),
    (
        "reg-compare",
        """OPENQASM 2.0;
qreg q[2];
creg c[2];
classical {
  r1 = 42;
  if (r1 == 42) { r1 = 1; } else { r1 = 0; }
}
""",
        [(0, 1), (1, 1)],
    ),
    (
        "nested-else",
        """OPENQASM 2.0;
qreg q[2];
creg c[2];
classical {
  if (c[0] == 1) {
    r1 = 10;
  } else {
    if (c[1] == 1) { r1 = 20; } else { r1 = 30; }
  }
}
""",
        # (c[0], c[1], expected x1)
        [(0, 0, 30), (1, 0, 10), (0, 1, 20), (1, 1, 10)],
    ),
]


def run_case(qasm, injections, num_cbits=1):
    quantum_ops, assembly = compile_hybrid(qasm)
    assert isinstance(quantum_ops, list), "quantum_ops must be a list"
    assert isinstance(assembly, str) and assembly.strip(), "assembly must be non-empty"
    results = {}
    for injection in injections:
        emulator = TinyRISCVEmulator()
        emulator.load_program(assembly)
        for k in range(num_cbits):
            emulator.set_register(f"x{10 + k}", injection[k])
        expected = injection[num_cbits]
        state = emulator.execute()
        results[injection[:num_cbits]] = state.get("x1", 0)
        assert state.get("x1", 0) == expected, (
            f"{injection[:num_cbits]=}: got {state.get('x1')}, expected {expected}\n{assembly}"
        )
    return quantum_ops, assembly, results


def main():
    for i, (name, qasm, injections) in enumerate(CASES):
        num_cbits = 2 if name == "nested-else" else 1
        quantum_ops, assembly, results = run_case(qasm, injections, num_cbits)
        print(f"[OK] {name}: {results}")
        if name == "public-branch":
            print("--- assembly ---")
            print(assembly)
            print("--- quantum ops ---")
            print(quantum_ops)


if __name__ == "__main__":
    main()
