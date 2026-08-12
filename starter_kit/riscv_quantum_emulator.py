#!/usr/bin/env python3
"""
LoomQ Quantum RISC-V Emulator (LQE) — fork of the official TinyRISCVEmulator
with the custom quantum instruction extension.

Compatibility: keeps the official API (load_program / execute / set_register /
get_register / x0-x31 registers) and the official instruction subset
(li/add/sub/addi/beq/bne/j). Adds the LQE quantum mnemonics from
`quantum_riscv.py` (qh/qx/qs/qsdg/qt/qtdg/qcx/qswap/qccx/qrz/qry/qcu1/qmeas).

The emulator maintains a full complex state vector of `2^n` amplitudes. Quantum
instructions evolve it exactly (no sampling). `qmeas xd, xs` performs ONE
measurement sample (seeded, reproducible) of qubit `xd`, collapses the state,
and stores the outcome into classical register x10+xs — matching the L3 rule
that measured bit c[k] is injected as x10+k.

New API:
  load_binary(words: List[int])   load a program from encoded 32-bit LQE words
  get_quantum_state()             current complex state vector (for verification)
  set_quantum_seed(seed)          fix the measurement RNG
"""

from __future__ import annotations

import cmath
import math
import random
from typing import Dict, List, Tuple

from quantum_riscv import ALL_MNEMONICS, encode_program

# ---- quantum gate matrices (qelib1 definitions) -----------------------------

_SQRT2 = math.sqrt(2.0)


def _H() -> List[List[complex]]:
    return [[1 / _SQRT2, 1 / _SQRT2], [1 / _SQRT2, -1 / _SQRT2]]


def _X() -> List[List[complex]]:
    return [[0, 1], [1, 0]]


def _S() -> List[List[complex]]:
    return [[1, 0], [0, 1j]]


def _SDG() -> List[List[complex]]:
    return [[1, 0], [0, -1j]]


def _T() -> List[List[complex]]:
    return [[1, 0], [0, cmath.exp(1j * math.pi / 4)]]


def _TDG() -> List[List[complex]]:
    return [[1, 0], [0, cmath.exp(-1j * math.pi / 4)]]


def _RZ(theta: float) -> List[List[complex]]:
    return [[cmath.exp(-1j * theta / 2), 0], [0, cmath.exp(1j * theta / 2)]]


def _RY(theta: float) -> List[List[complex]]:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return [[c, -s], [s, c]]


def _CX() -> List[List[complex]]:
    # control = targets[0] (local bit0), target = targets[1] (local bit1)
    # |00>->|00>, |01>->|11>, |10>->|10>, |11>->|01>
    return [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]


def _SWAP() -> List[List[complex]]:
    return [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]


def _CU1(theta: float) -> List[List[complex]]:
    return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
            [0, 0, 0, cmath.exp(1j * theta)]]


def _CCX() -> List[List[complex]]:
    # Toffoli on targets (ctrl0, ctrl1, target). In local-bit encoding
    # (bit0=ctrl0, bit1=ctrl1, bit2=target): flip |111> <-> |011>
    U = [[0.0] * 8 for _ in range(8)]
    for i in range(8):
        U[i][i] = 1
    U[3][3], U[3][7] = 0, 1
    U[7][3], U[7][7] = 1, 0
    return U


class TinyQuantumRISCVEmulator:
    """Official TinyRISCVEmulator + LQE quantum extension."""

    def __init__(self):
        # ---- official state ----
        self.registers = [0] * 32
        self.pc = 0
        self.labels: Dict[str, int] = {}
        self.instructions: List[Tuple[str, List[str]]] = []
        self.max_steps = 1000
        # ---- LQE state ----
        self.num_qubits = 0
        self.qstate: List[complex] = []
        self.qrng = random.Random(0)

    # ---- official register API ---------------------------------------------

    def set_register(self, reg: str, value: int):
        idx = self._parse_reg_idx(reg)
        if idx != 0:
            self.registers[idx] = value

    def get_register(self, reg: str) -> int:
        idx = self._parse_reg_idx(reg)
        return self.registers[idx]

    def _parse_reg_idx(self, reg: str) -> int:
        reg = reg.strip().replace(",", "")
        if not reg.startswith("x") and not reg.startswith("X"):
            raise ValueError(f"无效的寄存器名称: {reg}")
        idx = int(reg[1:])
        if idx < 0 or idx > 31:
            raise ValueError(f"寄存器索引超出范围 (x0-x31): {reg}")
        return idx

    # ---- program loading ---------------------------------------------------

    def load_program(self, asm_code: str):
        self.instructions = []
        self.labels = {}
        self.pc = 0
        self.registers = [0] * 32

        lines = asm_code.split("\n")
        temp_instructions = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            if "#" in line:
                line = line.split("#")[0].strip()

            if line.endswith(":"):
                label_name = line[:-1].strip()
                self.labels[label_name] = len(temp_instructions)
                continue
            elif ":" in line:
                parts = line.split(":", 1)
                label_name = parts[0].strip()
                self.labels[label_name] = len(temp_instructions)
                line = parts[1].strip()

            tokens = line.replace(",", " ").split()
            op = tokens[0].lower()
            args = tokens[1:]
            temp_instructions.append((op, args))

        self.instructions = temp_instructions
        self._refresh_quantum_state()

    def load_binary(self, words: List[int]):
        """Load a program from 32-bit LQE words (quantum mnemonics only)."""
        words = [w & 0xFFFFFFFF for w in words]
        self.instructions = []
        self.labels = {}
        self.pc = 0
        self.registers = [0] * 32
        from quantum_riscv import decode_instruction
        for w in words:
            op, args = decode_instruction(w)
            self.instructions.append((op, args))
        self._refresh_quantum_state()

    # ---- LQE quantum support -----------------------------------------------

    def set_quantum_seed(self, seed: int):
        self.qrng = random.Random(seed)

    def get_quantum_state(self) -> List[complex]:
        return list(self.qstate)

    def _refresh_quantum_state(self):
        """Infer qubit count from the maximum qubit index used anywhere."""
        n = 0
        for op, args in self.instructions:
            if op not in ALL_MNEMONICS:
                continue
            for arg in args:
                if arg.startswith("x") and arg[1:].isdigit():
                    n = max(n, int(arg[1:]) + 1)
        self.num_qubits = n
        self.qstate = [0.0 + 0.0j] * (1 << n)
        if n > 0:
            self.qstate[0] = 1.0 + 0.0j

    def _apply_unitary(self, U: List[List[complex]], targets: Tuple[int, ...]):
        n = self.num_qubits
        k = len(targets)
        others = [i for i in range(n) if i not in targets]
        dim = 1 << k
        for mask in range(1 << len(others)):
            base = 0
            for j, oi in enumerate(others):
                if (mask >> j) & 1:
                    base |= 1 << oi
            idxs = []
            for local in range(dim):
                idx = base
                for j, t in enumerate(targets):
                    if (local >> j) & 1:
                        idx |= 1 << t
                idxs.append(idx)
            v = [self.qstate[i] for i in idxs]
            nv = [sum(U[r][c] * v[c] for c in range(dim)) for r in range(dim)]
            for i, val in zip(idxs, nv):
                self.qstate[i] = val

    def _exec_quantum(self, op: str, args: List[str]):
        def qbit(token: str) -> int:
            return self._parse_reg_idx(token)

        if op in ("qh", "qx", "qs", "qsdg", "qt", "qtdg"):
            U = {"qh": _H(), "qx": _X(), "qs": _S(),
                 "qsdg": _SDG(), "qt": _T(), "qtdg": _TDG()}[op]
            self._apply_unitary(U, (qbit(args[0]),))
        elif op == "qcx":
            self._apply_unitary(_CX(), (qbit(args[1]), qbit(args[0])))
        elif op == "qswap":
            self._apply_unitary(_SWAP(), (qbit(args[0]), qbit(args[1])))
        elif op == "qccx":
            self._apply_unitary(_CCX(), (qbit(args[1]), qbit(args[2]), qbit(args[0])))
        elif op == "qrz":
            self._apply_unitary(_RZ(float(args[-1])), (qbit(args[0]),))
        elif op == "qry":
            self._apply_unitary(_RY(float(args[-1])), (qbit(args[0]),))
        elif op == "qcu1":
            self._apply_unitary(_CU1(float(args[-1])), (qbit(args[1]), qbit(args[0])))
        elif op == "qmeas":
            self._measure(qbit(args[0]), qbit(args[1]))
        else:
            raise ValueError(f"unsupported LQE instruction: {op}")

    def _measure(self, qubit: int, clbit: int):
        n = self.num_qubits
        p0 = 0.0
        for i in range(1 << n):
            if not ((i >> qubit) & 1):
                p0 += abs(self.qstate[i]) ** 2
        outcome = 0 if self.qrng.random() < p0 else 1
        # collapse + renormalise
        norm = 0.0
        for i in range(1 << n):
            if ((i >> qubit) & 1) == outcome:
                norm += abs(self.qstate[i]) ** 2
        if norm > 0:
            for i in range(1 << n):
                if ((i >> qubit) & 1) != outcome:
                    self.qstate[i] = 0.0 + 0.0j
                else:
                    self.qstate[i] /= math.sqrt(norm)
        self.set_register(f"x{10 + clbit}", outcome)

    # ---- execution ---------------------------------------------------------

    def execute(self) -> Dict[str, int]:
        steps = 0
        num_instr = len(self.instructions)

        while 0 <= self.pc < num_instr:
            steps += 1
            if steps > self.max_steps:
                raise RuntimeError("程序执行超出最大步数限制，疑似发生死循环")

            op, args = self.instructions[self.pc]
            next_pc = self.pc + 1

            if op in ALL_MNEMONICS:
                self._exec_quantum(op, args)
            elif op == "li":
                rd, imm = args[0], int(args[1])
                self.set_register(rd, imm)
            elif op == "add":
                rd, rs1, rs2 = args[0], args[1], args[2]
                self.set_register(rd, self.get_register(rs1) + self.get_register(rs2))
            elif op == "sub":
                rd, rs1, rs2 = args[0], args[1], args[2]
                self.set_register(rd, self.get_register(rs1) - self.get_register(rs2))
            elif op == "addi":
                rd, rs1, imm = args[0], args[1], int(args[2])
                self.set_register(rd, self.get_register(rs1) + imm)
            elif op == "beq":
                rs1, rs2, label = args[0], args[1], args[2]
                if self.get_register(rs1) == self.get_register(rs2):
                    if label not in self.labels:
                        raise ValueError(f"未定义的跳转标签: {label}")
                    next_pc = self.labels[label]
            elif op == "bne":
                rs1, rs2, label = args[0], args[1], args[2]
                if self.get_register(rs1) != self.get_register(rs2):
                    if label not in self.labels:
                        raise ValueError(f"未定义的跳转标签: {label}")
                    next_pc = self.labels[label]
            elif op == "j":
                label = args[0]
                if label not in self.labels:
                    raise ValueError(f"未定义的跳转标签: {label}")
                next_pc = self.labels[label]
            else:
                raise ValueError(f"不支持的指令操作: {op}")

            self.pc = next_pc

        result = {}
        for idx, val in enumerate(self.registers):
            if val != 0:
                result[f"x{idx}"] = val
        return result


if __name__ == "__main__":
    # GHZ-2: qh x0; qcx x1, x0; qmeas x0, x0; qmeas x1, x1
    prog = """
    qh x0
    qcx x1, x0
    qmeas x0, x0
    qmeas x1, x1
    """
    emu = TinyQuantumRISCVEmulator()
    emu.load_program(prog)
    state = emu.execute()
    print("GHZ-2 measurement outcome ->", state)
    print("post-measure state:", emu.get_quantum_state())
    print("Quantum RISC-V emulator smoke test OK")
