#!/usr/bin/env python3
"""LoomQ Quantum RISC-V Extension (LQE) — custom opcode encoder/decoder.

Design spec (full document: docs/quantum_riscv_spec.md):

    Word format (32-bit, RISC-V custom-0 opcode 0b0001011):
      [31:25] funct7   [24:20] rs2   [19:15] rs1   [14:12] funct3
      [11:7]  rd       [6:0]   opcode = 0b0001011

    funct3 classes:
      000 single-qubit gates (funct7 selects H/X/S/SDG/T/TDG)
      001 two-qubit gates    (funct7 selects CX/SWAP)
      010 three-qubit gates  (funct7 selects CCX)
      011 parameter gate qrz (rd=qubit, funct7+rs2 = 12-bit angle)
      100 measurement        (rd = qubit, rs1 = classical-bit index)
      101 parameter gate qry (rd=qubit, funct7+rs2 = 12-bit angle)
      110 parameter gate qcu1 (rd=target, rs1=control, funct7+rs2 = 12-bit angle)
      111 reserved

    Angle encoding:  theta = imm * pi / 128   (12-bit signed, ~1.4 deg step)
      imm = round(theta * 128 / pi);  imm in [-2048, 2047]
      imm stored as funct7[6:0] << 5 | rs2[4:0]

    Classical-bit mapping follows L3: measured bit c[k] is stored in x10+k.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

OPCODE = 0b0001011  # RISC-V custom-0

F3_SINGLE = 0b000
F3_TWO = 0b001
F3_THREE = 0b010
F3_QRZ = 0b011
F3_QMEAS = 0b100
F3_QRY = 0b101
F3_QCU1 = 0b110

# funct7 selectors for the fixed-structure classes
F7_SINGLE = {"qh": 0, "qx": 1, "qs": 2, "qsdg": 3, "qt": 4, "qtdg": 5}
F7_TWO = {"qcx": 0, "qswap": 1}
F7_THREE = {"qccx": 0}

# reverse lookup: (funct3, funct7) -> mnemonic
MNEMONIC_BY = {}
for _m, _f in F7_SINGLE.items():
    MNEMONIC_BY[(F3_SINGLE, _f)] = _m
for _m, _f in F7_TWO.items():
    MNEMONIC_BY[(F3_TWO, _f)] = _m
for _m, _f in F7_THREE.items():
    MNEMONIC_BY[(F3_THREE, _f)] = _m
MNEMONIC_BY[(F3_QRZ, 0)] = "qrz"
MNEMONIC_BY[(F3_QRY, 0)] = "qry"
MNEMONIC_BY[(F3_QCU1, 0)] = "qcu1"
MNEMONIC_BY[(F3_QMEAS, 0)] = "qmeas"

ALL_MNEMONICS = tuple(MNEMONIC_BY.values())

# 12-gate whitelist (contract) mapped to LQE mnemonics
QASM_TO_LQE = {
    "h": "qh", "x": "qx", "s": "qs", "sdg": "qsdg", "t": "qt", "tdg": "qtdg",
    "cx": "qcx", "swap": "qswap", "ccx": "qccx",
    "rz": "qrz", "ry": "qry", "cu1": "qcu1", "measure": "qmeas",
}

# qubit operand count per mnemonic (parameter gates take one extra angle arg)
OPERAND_COUNT = {
    "qh": 1, "qx": 1, "qs": 1, "qsdg": 1, "qt": 1, "qtdg": 1,
    "qcx": 2, "qswap": 2, "qccx": 3,
    "qrz": 1, "qry": 1, "qcu1": 2,
    "qmeas": 2,
}
PARAM_GATES = {"qrz", "qry", "qcu1"}

_F3_BY_OP = {"qrz": F3_QRZ, "qry": F3_QRY, "qcu1": F3_QCU1}


def _reg_index(token: str) -> int:
    token = token.strip().replace(",", "").lstrip("xX")
    idx = int(token)
    if not 0 <= idx <= 31:
        raise ValueError(f"register index out of range x0-x31: {token}")
    return idx


def _quantize(theta: float) -> int:
    """Float radians -> 12-bit signed integer angle token."""
    imm = int(round(theta * 128.0 / math.pi))
    if imm < -2048 or imm > 2047:
        raise ValueError(f"angle out of 12-bit range: theta={theta} rad")
    return imm


def _dequantize(imm: int) -> float:
    return imm * math.pi / 128.0


def encode_instruction(op: str, args: List[str]) -> int:
    """Encode one LQE assembly instruction to a 32-bit word."""
    op = op.lower()
    if op not in ALL_MNEMONICS:
        raise ValueError(f"unknown LQE mnemonic: {op}")
    if len(args) != OPERAND_COUNT[op] + (1 if op in PARAM_GATES else 0):
        raise ValueError(f"{op}: expected {OPERAND_COUNT[op] + (1 if op in PARAM_GATES else 0)} operand(s), got {len(args)}")

    if op in F7_SINGLE:
        f3, f7, rd, rs1, rs2 = F3_SINGLE, F7_SINGLE[op], _reg_index(args[0]), 0, 0
    elif op in F7_TWO:
        # qcx xd, xs  -> rs1 = control, rd = target ; qswap xa, xb -> rs1=a, rd=b
        if op == "qcx":
            rs1, rd = _reg_index(args[1]), _reg_index(args[0])
        else:
            rs1, rd = _reg_index(args[0]), _reg_index(args[1])
        f3, f7, rs2 = F3_TWO, F7_TWO[op], 0
    elif op in F7_THREE:
        # qccx xd, xa, xb -> rs1 = ctrl0, rs2 = ctrl1, rd = target
        f3, f7 = F3_THREE, F7_THREE[op]
        rs1, rs2 = _reg_index(args[1]), _reg_index(args[2])
        rd = _reg_index(args[0])
    elif op in PARAM_GATES:
        f3 = _F3_BY_OP[op]
        theta = float(args[-1])
        imm = _quantize(theta)
        f7 = (imm >> 5) & 0x7F
        rs2 = imm & 0x1F
        if op == "qcu1":
            rd, rs1 = _reg_index(args[0]), _reg_index(args[1])
        else:
            rd, rs1 = _reg_index(args[0]), 0
    else:  # qmeas
        # qmeas xd, xs -> rd = qubit, rs1 = classical-bit index, result -> x10+rs1
        f3, f7 = F3_QMEAS, 0
        rd, rs1, rs2 = _reg_index(args[0]), _reg_index(args[1]), 0

    word = (f7 & 0x7F) << 25 | (rs2 & 0x1F) << 20 | (rs1 & 0x1F) << 15 \
        | (f3 & 0x7) << 12 | (rd & 0x1F) << 7 | OPCODE
    return word & 0xFFFFFFFF


def decode_instruction(word: int) -> Tuple[str, List[str]]:
    """Decode a 32-bit word back to (mnemonic, args)."""
    word &= 0xFFFFFFFF
    opcode = word & 0x7F
    if opcode != OPCODE:
        raise ValueError(f"not an LQE word: opcode=0x{opcode:02x}")
    f3 = (word >> 12) & 0x7
    f7 = (word >> 25) & 0x7F
    rd = (word >> 7) & 0x1F
    rs1 = (word >> 15) & 0x1F
    rs2 = (word >> 20) & 0x1F

    # parameter gates: funct3 identifies the gate, funct7+rs2 carry the angle
    if f3 in (F3_QRZ, F3_QRY, F3_QCU1):
        imm = (f7 << 5) | rs2
        if imm & 0x800:
            imm -= 0x1000
        angle = f"{_dequantize(imm):.10g}"
        if f3 == F3_QCU1:
            return "qcu1", [f"x{rd}", f"x{rs1}", angle]
        return "qrz" if f3 == F3_QRZ else "qry", [f"x{rd}", angle]

    key = (f3, f7)
    if key not in MNEMONIC_BY:
        raise ValueError(f"unknown LQE (funct3={f3:b}, funct7={f7:07b})")
    op = MNEMONIC_BY[key]

    if op in F7_SINGLE:
        return op, [f"x{rd}"]
    if op in F7_TWO:
        if op == "qcx":
            return op, [f"x{rd}", f"x{rs1}"]
        return op, [f"x{rs1}", f"x{rd}"]
    if op in F7_THREE:
        return op, [f"x{rd}", f"x{rs1}", f"x{rs2}"]
    return op, [f"x{rd}", f"x{rs1}"]


def encode_program(asm: str) -> List[int]:
    """Encode a block of LQE assembly lines (mnemonics only) to 32-bit words."""
    words = []
    for line in asm.splitlines():
        line = line.split("#", 1)[0].split(";", 1)[0].strip()
        if not line:
            continue
        tokens = line.replace(",", " ").split()
        words.append(encode_instruction(tokens[0], tokens[1:]))
    return words


def encode_quantum_ops(quantum_ops: List[str]) -> List[int]:
    """Encode L3-style quantum ops (e.g. "h q[0]", "cx q[0], q[1]",
    "measure q[0] -> c[1]") into LQE 32-bit words."""
    words = []
    for line in quantum_ops:
        line = line.strip()
        if not line:
            continue
        if "->" in line:  # measure q[a] -> c[b]
            lhs, rhs = line.split("->")
            qtok = lhs.strip().split()[1]           # q[0]
            cltok = rhs.strip()                     # c[1]
            qidx = int(qtok[2:-1])
            cidx = int(cltok[2:-1])
            words.append(encode_instruction("qmeas", [f"x{qidx}", f"x{cidx}"]))
            continue
        tokens = line.replace(",", " ").split()
        gate = tokens[0].lower()
        if gate not in QASM_TO_LQE:
            raise ValueError(f"gate not in 12-gate whitelist: {gate}")
        lqe = QASM_TO_LQE[gate]
        # qubit args first, then parameter (if any)
        args = []
        for tok in tokens[1:]:
            tok = tok.strip()
            if tok.startswith("q["):
                args.append(f"x{int(tok[2:-1])}")
            else:
                args.append(tok)
        words.append(encode_instruction(lqe, args))
    return words


def decode_program(words: List[int]) -> str:
    return "\n".join(" ".join([op] + args) for op, args in
                     (decode_instruction(w) for w in words)) + "\n"


if __name__ == "__main__":
    # smoke round-trip over every mnemonic
    samples = [
        "qh x1", "qx x2", "qs x3", "qsdg x4", "qt x5", "qtdg x6",
        "qcx x2, x1", "qswap x1, x2", "qccx x3, x1, x2",
        "qrz x1, 1.5707963267948966", "qry x2, 0.7853981633974483",
        "qcu1 x2, x1, 3.141592653589793", "qmeas x1, x0",
    ]
    for s in samples:
        tokens = s.replace(",", " ").split()
        op, args = tokens[0], tokens[1:]
        w = encode_instruction(op, args)
        op2, args2 = decode_instruction(w)
        print(f"{w:08x}  {s:<40} -> {op2} {' '.join(args2)}")
