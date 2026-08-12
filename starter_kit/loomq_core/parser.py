"""OpenQASM 2.0 parser for the LoomQ 12-gate whitelist.

The contract guarantees every evaluation circuit (public, hidden, and contest-day
variants) only uses the 12 qelib1 gates below, so this parser is deliberately
small: version line, optional include, qreg/creg declarations, gate calls (with
optional parameters), and measure statements. No loops / conditionals / custom
gates are expected.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Whitelisted gate names -> (arity of qubit operands, has parameters)
WHITELIST = {
    "h": (1, False),
    "x": (1, False),
    "s": (1, False),
    "sdg": (1, False),
    "t": (1, False),
    "tdg": (1, False),
    "rz": (1, True),
    "ry": (1, True),
    "cx": (2, False),
    "cu1": (2, True),
    "swap": (2, False),
    "ccx": (3, False),
}


@dataclass
class Gate:
    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = ()


@dataclass
class Measure:
    """measure q[a] -> c[b];  a/b are register indices (None = whole register)."""

    qubit_index: int
    clbit_index: int


@dataclass
class Circuit:
    num_qubits: int
    num_clbits: int
    gates: List[Gate] = field(default_factory=list)
    measures: List[Measure] = field(default_factory=list)

    @property
    def operations(self):
        """Yield gates in order. Measures are returned separately via `.measures`."""
        return self.gates


_QREG_RE = re.compile(r"^\s*qreg\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\d+)\s*\]\s*;")
_CREG_RE = re.compile(r"^\s*creg\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\d+)\s*\]\s*;")
_GATE_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?:\(\s*([^)]*)\s*\))?"  # optional parameter list
    r"\s+(.+?)\s*;\s*$"
)
_MEASURE_RE = re.compile(
    r"^\s*measure\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[\s*(\d+)\s*\])?\s*->\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[\s*(\d+)\s*\])?\s*;\s*$"
)


def _strip_comments(text: str) -> str:
    # Remove /* ... */ block comments first, then // line comments.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    lines = []
    for line in text.splitlines():
        marker = line.find("//")
        if marker >= 0:
            line = line[:marker]
        lines.append(line)
    return "\n".join(lines)


def _parse_float_list(raw: str) -> List[float]:
    raw = raw.strip()
    if not raw:
        return []
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(_eval_number(token))
    return values


def _eval_number(token: str) -> float:
    """Evaluate a numeric token which may contain pi / arithmetic (e.g. pi/2)."""
    token = token.replace("pi", str(math.pi))
    try:
        return float(eval(token, {"__builtins__": {}}, {"pi": math.pi}))  # noqa: S307
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"cannot parse number token: {token!r}") from exc


def parse_qasm(qasm_str: str) -> Circuit:
    """Parse an OpenQASM 2.0 string into a :class:`Circuit`.

    Raises ValueError on any construct outside the whitelist.
    """
    text = _strip_comments(qasm_str)
    if not re.search(r"^\s*OPENQASM\s+2\.0\s*;", text, flags=re.MULTILINE):
        raise ValueError("missing 'OPENQASM 2.0;' header")

    qregs: dict = {}
    cregs: dict = {}
    qreg_order: List[str] = []
    creg_order: List[str] = []
    num_qubits = 0
    num_clbits = 0
    qname = cname = None

    gates: List[Gate] = []
    measures: List[Measure] = []
    seen_measure = False
    current_qreg_qubits: dict = {}  # qreg name -> base index
    current_creg_bits: dict = {}  # creg name -> base index

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^\s*OPENQASM\s+", line):
            continue
        if re.match(r"^\s*include\s+", line):
            continue

        m = _QREG_RE.match(line)
        if m:
            name, size = m.group(1), int(m.group(2))
            current_qreg_qubits[name] = num_qubits
            qregs[name] = size
            qreg_order.append(name)
            num_qubits += size
            qname = name
            continue

        m = _CREG_RE.match(line)
        if m:
            name, size = m.group(1), int(m.group(2))
            current_creg_bits[name] = num_clbits
            cregs[name] = size
            creg_order.append(name)
            num_clbits += size
            cname = name
            continue

        m = _MEASURE_RE.match(line)
        if m:
            src_reg, src_idx, dst_reg, dst_idx = m.groups()
            if src_reg not in qregs:
                raise ValueError(f"measure uses unknown qreg {src_reg!r}")
            if dst_reg not in cregs:
                raise ValueError(f"measure uses unknown creg {dst_reg!r}")
            q_base = current_qreg_qubits[src_reg]
            c_base = current_creg_bits[dst_reg]
            if src_idx is None:
                q_indices = list(range(q_base, q_base + qregs[src_reg]))
            else:
                q_indices = [q_base + int(src_idx)]
            if dst_idx is None:
                c_indices = list(range(c_base, c_base + cregs[dst_reg]))
            else:
                c_indices = [c_base + int(dst_idx)]
            if len(q_indices) != len(c_indices):
                raise ValueError("measure register size mismatch")
            for qi, ci in zip(q_indices, c_indices):
                measures.append(Measure(qubit_index=qi, clbit_index=ci))
            seen_measure = True
            continue

        m = _GATE_RE.match(line)
        if m:
            gname, param_raw, operand_raw = m.group(1).lower(), m.group(2), m.group(3)
            if gname not in WHITELIST:
                raise ValueError(f"gate {gname!r} is outside the 12-gate whitelist")
            arity, has_params = WHITELIST[gname]
            if has_params:
                if param_raw is None:
                    raise ValueError(f"gate {gname} requires parameters")
                params = _parse_float_list(param_raw)
            else:
                params = ()
            # Parse operands like "q[0], q[1]"
            operands = [op.strip() for op in operand_raw.split(",") if op.strip()]
            qubits: List[int] = []
            for op in operands:
                mm = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\d+)\s*\]", op)
                if not mm:
                    raise ValueError(f"invalid gate operand {op!r}")
                reg, idx = mm.group(1), int(mm.group(2))
                if reg not in qregs:
                    raise ValueError(f"gate uses unknown qreg {reg!r}")
                qubits.append(current_qreg_qubits[reg] + idx)
            if len(qubits) != arity:
                raise ValueError(
                    f"gate {gname} expects {arity} qubit operand(s), got {len(qubits)}"
                )
            gates.append(Gate(name=gname, qubits=tuple(qubits), params=tuple(params)))
            continue

        raise ValueError(f"unparseable line: {line!r}")

    if num_qubits == 0:
        raise ValueError("circuit declares no qubits")
    return Circuit(num_qubits=num_qubits, num_clbits=num_clbits, gates=gates, measures=measures)
