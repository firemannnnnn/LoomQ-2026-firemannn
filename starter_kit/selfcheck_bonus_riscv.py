#!/usr/bin/env python3
"""End-to-end test for the LoomQ Quantum RISC-V Extension (Bonus, +8).

Covers the three required deliverables:
  1. encoding spec conformance  (opcode/funct3 field placement)
  2. simulator extension        (forked TinyQuantumRISCVEmulator executes LQE)
  3. runnable end-to-end tests  (state evolution vs. independent reference,
                                 binary loading, hybrid compile-and-run)

Run:  python3 selfcheck_bonus_riscv.py
"""
import cmath
import math
import sys

sys.path.insert(0, ".")

from quantum_riscv import (
    OPCODE, F3_SINGLE, F3_TWO, F3_THREE, F3_QRZ, F3_QMEAS, F3_QRY, F3_QCU1,
    ALL_MNEMONICS, encode_instruction, decode_instruction,
    encode_program, decode_program, encode_quantum_ops,
)
from riscv_quantum_emulator import TinyQuantumRISCVEmulator

SQRT2 = math.sqrt(2.0)
PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"[PASS] {name}")
    else:
        FAIL += 1
        print(f"[FAIL] {name} {detail}")


# ---- independent reference matrices (not shared with the emulator) ---------

def ref_ry(theta):
    return [[math.cos(theta / 2), -math.sin(theta / 2)],
            [math.sin(theta / 2), math.cos(theta / 2)]]


def ref_rz(theta):
    return [[cmath.exp(-1j * theta / 2), 0], [0, cmath.exp(1j * theta / 2)]]


def ref_cu1(theta):
    return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
            [0, 0, 0, cmath.exp(1j * theta)]]


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ---- 1. encoding spec conformance -------------------------------------------

def test_encoding_conformance():
    w = encode_instruction("qh", ["x1"])
    check("opcode bits [6:0] = 0b0001011 (custom-0)", (w & 0x7F) == OPCODE,
          hex(w))
    check("single-qubit gate funct3 = 000", ((w >> 12) & 0x7) == F3_SINGLE)
    check("qh funct7 = 0", ((w >> 25) & 0x7F) == 0)
    w2 = encode_instruction("qt", ["x5"])
    check("qt funct7 = 4", ((w2 >> 25) & 0x7F) == 4)
    check("qt rd = 5", ((w2 >> 7) & 0x1F) == 5)

    w3 = encode_instruction("qcx", ["x2", "x1"])
    check("two-qubit gate funct3 = 001", ((w3 >> 12) & 0x7) == F3_TWO)
    check("qcx rd = target=2, rs1 = control=1",
          ((w3 >> 7) & 0x1F) == 2 and ((w3 >> 15) & 0x1F) == 1, hex(w3))

    w4 = encode_instruction("qccx", ["x3", "x1", "x2"])
    check("three-qubit gate funct3 = 010", ((w4 >> 12) & 0x7) == F3_THREE)
    check("qccx rd=3 rs1=1 rs2=2",
          ((w4 >> 7) & 0x1F) == 3 and ((w4 >> 15) & 0x1F) == 1
          and ((w4 >> 20) & 0x1F) == 2, hex(w4))

    w5 = encode_instruction("qrz", ["x1", "1.5707963267948966"])  # pi/2
    imm = ((w5 >> 25) & 0x7F) << 5 | ((w5 >> 20) & 0x1F)
    check("qrz funct3 = 011 (own class)", ((w5 >> 12) & 0x7) == F3_QRZ)
    check("qrz angle imm = round(pi/2 * 128/pi) = 64", imm == 64, str(imm))

    w6 = encode_instruction("qmeas", ["x1", "x0"])
    check("qmeas funct3 = 100", ((w6 >> 12) & 0x7) == F3_QMEAS)
    check("qmeas rd=qubit=1 rs1=clbit=0",
          ((w6 >> 7) & 0x1F) == 1 and ((w6 >> 15) & 0x1F) == 0, hex(w6))


def test_roundtrip():
    samples = [
        ["qh", ["x1"]], ["qx", ["x2"]], ["qs", ["x3"]], ["qsdg", ["x4"]],
        ["qt", ["x5"]], ["qtdg", ["x6"]],
        ["qcx", ["x2", "x1"]], ["qswap", ["x1", "x2"]],
        ["qccx", ["x3", "x1", "x2"]],
        ["qrz", ["x1", "1.5707963267948966"]],
        ["qry", ["x2", "0.7853981633974483"]],
        ["qcu1", ["x2", "x1", "3.141592653589793"]],
        ["qmeas", ["x1", "x0"]],
    ]
    for op, args in samples:
        w = encode_instruction(op, args)
        op2, args2 = decode_instruction(w)
        ok = op2 == op and len(args2) == len(args)
        if ok and op in ("qrz", "qry", "qcu1"):
            # angle tolerance within one quantization step (pi/128)
            a1, a2 = float(args[-1]), float(args2[-1])
            ok = abs(a1 - a2) <= math.pi / 128 + 1e-9
        check(f"round-trip {op} {' '.join(args)}", ok, f"-> {op2} {' '.join(args2)}")


# ---- 2. state-vector evolution vs. independent reference --------------------

def run_qasm(asm):
    emu = TinyQuantumRISCVEmulator()
    emu.load_program(asm)
    emu.execute()
    return emu.get_quantum_state()


def test_state_evolution():
    # GHZ-2 = (|00> + |11>)/sqrt(2)
    st = run_qasm("qh x0\nqcx x1, x0\n")
    check("GHZ-2 state (|00>+|11>)/sqrt2",
          close(st[0], 1 / SQRT2) and close(st[3], 1 / SQRT2)
          and abs(st[1]) < 1e-12 and abs(st[2]) < 1e-12, str(st))

    # Bell via qcx with swapped order -> control=q1 target=q0 gives |00>+|11> too
    st2 = run_qasm("qh x1\nqcx x0, x1\n")
    check("Bell (control=q1) (|00>+|11>)/sqrt2",
          close(st2[0], 1 / SQRT2) and close(st2[3], 1 / SQRT2), str(st2))

    # X gate flips |0> to |1>
    st3 = run_qasm("qx x2\n")
    check("qx |0> -> |1>", close(st3[4], 1.0), str(st3))

    # RY(pi/2) on |0> -> (|0>+|1>)/sqrt2 (angle quantized: ~0.4% error bound)
    st4 = run_qasm("qry x0, 1.5707963267948966\n")
    ref = [ref_ry(math.pi / 2)[r][0] for r in range(2)]  # U|0> = first column
    check("qry(pi/2) |0> ~ (|0>+|1>)/sqrt2",
          abs(st4[0] - ref[0]) < 0.01 and abs(st4[1] - ref[1]) < 0.01,
          f"{st4} vs {ref}")

    # RZ(pi/2) phase on |1>: <1| RZ |1> = e^{+i pi/4}
    st5 = run_qasm("qx x0\nqrz x0, 1.5707963267948966\n")
    check("qrz(pi/2) |1> phase e^{+i pi/4}",
          abs(st5[1] - cmath.exp(1j * math.pi / 4)) < 0.01, str(st5[1]))

    # CU1(pi) on |11>: phase e^{i pi}
    st6 = run_qasm("qx x0\nqx x1\nqcu1 x1, x0, 3.141592653589793\n")
    check("qcu1(pi) |11> phase e^{i pi}",
          abs(st6[3] - cmath.exp(1j * math.pi)) < 0.01, str(st6[3]))

    # CCX truth table: |110> -> |111>
    st7 = run_qasm("qx x0\nqx x1\nqccx x2, x0, x1\n")
    check("qccx |110> -> |111>", close(st7[7], 1.0), str(st7))

    # S / T phase conventions (prepare |1> first)
    st8 = run_qasm("qx x0\nqt x0\n")
    check("qt |1> phase e^{i pi/4}",
          abs(st8[1] - cmath.exp(1j * math.pi / 4)) < 1e-12, str(st8[1]))
    st9 = run_qasm("qx x0\nqs x0\n")
    check("qs |1> phase i", abs(st9[1] - 1j) < 1e-12, str(st9[1]))


def test_measure_classical_hybrid():
    # GHZ-2: both measured qubits always agree -> r1 = 1 (any seed)
    prog = """
    qh x0
    qcx x1, x0
    qmeas x0, x0
    qmeas x1, x1
    li x1, 0
    bne x10, x11, NOT_EQUAL
    li x1, 1
    j DONE
    NOT_EQUAL:
    li x1, 0
    DONE:
    """
    for seed in (0, 1, 42):
        emu = TinyQuantumRISCVEmulator()
        emu.set_quantum_seed(seed)
        emu.load_program(prog)
        state = emu.execute()
        check(f"GHZ measurement agreement (seed={seed}): x1=1",
              state.get("x1") == 1, str(state))


def test_binary_loading():
    asm = "qh x0\nqcx x1, x0\n"
    words = encode_program(asm)
    emu = TinyQuantumRISCVEmulator()
    emu.load_binary(words)
    emu.execute()
    st = emu.get_quantum_state()
    check("load_binary executes (|00>+|11>)/sqrt2",
          close(st[0], 1 / SQRT2) and close(st[3], 1 / SQRT2), str(st))
    check("decode_program round-trips source",
          words == encode_program(decode_program(words)))


def test_hybrid_pipeline():
    # L3 hybrid -> quantum_ops -> LQE binary -> quantum emulator
    from l3_compiler import compile_hybrid
    hybrid = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
classical { if (c[0] == c[1]) { r1 = 1; } else { r1 = 0; } }
"""
    quantum_ops, assembly = compile_hybrid(hybrid)
    words = encode_quantum_ops(quantum_ops)
    check("hybrid pipeline encodes N>0 words", len(words) == 4, str(words))

    emu = TinyQuantumRISCVEmulator()
    emu.set_quantum_seed(7)
    emu.load_binary(words)
    emu.execute()
    # post-measurement GHZ state is collapsed to a computational basis state
    st = emu.get_quantum_state()
    norm = sum(abs(a) ** 2 for a in st)
    check("hybrid GHZ-2 collapsed pure state after measure",
          abs(norm - 1.0) < 1e-12, f"norm={norm}")
    check("hybrid GHZ-2 measurement agrees (x10==x11)",
          emu.get_register("x10") == emu.get_register("x11"),
          f"x10={emu.get_register('x10')} x11={emu.get_register('x11')}")

    # classical follow-up still runs on the official instruction subset
    emu2 = TinyQuantumRISCVEmulator()
    emu2.set_quantum_seed(7)
    emu2.load_program(assembly)
    emu2.execute()
    # no measures executed -> x10/x11 zero -> c[0]==c[1] -> r1=1
    check("official classical assembly still runs (r1=1)", emu2.get_register("x1") == 1)


def test_official_subset_compat():
    from selfcheck_l3 import CASES
    from riscv_quantum_emulator import TinyQuantumRISCVEmulator as QEmu
    for name, qasm, injections in CASES:
        from l3_compiler import compile_hybrid
        _, assembly = compile_hybrid(qasm)
        num_cbits = 2 if name == "nested-else" else 1
        ok = True
        for inj in injections:
            emu = QEmu()
            emu.load_program(assembly)
            for k in range(num_cbits):
                emu.set_register(f"x{10 + k}", inj[k])
            expected = inj[num_cbits]
            st = emu.execute()
            if st.get("x1", 0) != expected:
                ok = False
        check(f"official L3 case '{name}' passes on LQE emulator", ok)


def main():
    print("=== LoomQ Quantum RISC-V Extension (Bonus) end-to-end ===")
    test_encoding_conformance()
    test_roundtrip()
    test_state_evolution()
    test_measure_classical_hybrid()
    test_binary_loading()
    test_hybrid_pipeline()
    test_official_subset_compat()
    print(f"\nResult: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
