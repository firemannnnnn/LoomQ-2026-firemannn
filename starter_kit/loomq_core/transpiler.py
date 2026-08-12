"""Render the parsed Circuit into each target backend's native IR.

Output follows `target_ir_contract.md` exactly:
  - spinq   -> full executable OpenQASM 2.0
  - braket  -> full OpenQASM 3 (stdgates.inc)
  - originq -> OriginIR text (QINIT/CREG + whitelist gate names)
"""

from __future__ import annotations

from typing import List

from .parser import Circuit, Gate, Measure

# Parameter gates that exist on all three targets under the same name.
_DIRECT_GATES = {"h", "x", "s", "t", "rz", "ry", "cx", "swap"}


def _fmt_float(value: float) -> str:
    """Format a parameter value compactly (python repr of the float)."""
    text = repr(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _fmt_params(params: List[float]) -> str:
    return ",".join(_fmt_float(p) for p in params)


# --------------------------------------------------------------------------
# spinq : OpenQASM 2.0
# --------------------------------------------------------------------------

def _render_spinq(circuit: Circuit) -> str:
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";']
    lines.append(f"qreg q[{circuit.num_qubits}];")
    if circuit.num_clbits:
        lines.append(f"creg c[{circuit.num_clbits}];")
    for gate in circuit.gates:
        if gate.params:
            lines.append(
                f"{gate.name}({_fmt_params(gate.params)}) "
                + ", ".join(f"q[{q}]" for q in gate.qubits) + ";"
            )
        else:
            lines.append(f"{gate.name} " + ", ".join(f"q[{q}]" for q in gate.qubits) + ";")
    if circuit.measures:
        if circuit.num_clbits == circuit.num_qubits:
            # Whole-register measure when it is a 1:1 mapping.
            lines.append("measure q -> c;")
        else:
            for m in circuit.measures:
                lines.append(f"measure q[{m.qubit_index}] -> c[{m.clbit_index}];")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# braket : OpenQASM 3.0
# --------------------------------------------------------------------------

def _render_braket(circuit: Circuit) -> str:
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";']
    lines.append(f"qubit[{circuit.num_qubits}] q;")
    if circuit.num_clbits:
        lines.append(f"bit[{circuit.num_clbits}] c;")
    for gate in circuit.gates:
        name, qs, ps = gate.name, gate.qubits, gate.params
        if name == "cx":
            lines.append(f"cnot q[{qs[0]}], q[{qs[1]}];")
        elif name == "ccx":
            lines.append(f"ccnot q[{qs[0]}], q[{qs[1]}], q[{qs[2]}];")
        elif name == "cu1":
            lines.append(f"cphaseshift({_fmt_float(ps[0])}) q[{qs[0]}], q[{qs[1]}];")
        elif name == "sdg":
            lines.append(f"sdg q[{qs[0]}];")
        elif name == "tdg":
            lines.append(f"tdg q[{qs[0]}];")
        elif ps:
            lines.append(
                f"{name}({_fmt_params(ps)}) " + ", ".join(f"q[{q}]" for q in qs) + ";"
            )
        else:
            lines.append(f"{name} " + ", ".join(f"q[{q}]" for q in qs) + ";")
    if circuit.measures:
        if circuit.num_clbits == circuit.num_qubits:
            lines.append("c = measure q;")
        else:
            for m in circuit.measures:
                lines.append(f"c[{m.clbit_index}] = measure q[{m.qubit_index}];")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# originq : OriginIR
# --------------------------------------------------------------------------

_ORIGINQ_GATE = {
    "h": "H",
    "x": "X",
    "s": "S",
    "sdg": "SDAG",
    "t": "T",
    "tdg": "TDAG",
    "rz": "RZ",
    "ry": "RY",
    "cx": "CNOT",
    "cu1": "CU1",
    "swap": "SWAP",
    "ccx": "TOFFOLI",
}


def _render_originq(circuit: Circuit) -> str:
    lines = [f"QINIT {circuit.num_qubits}"]
    if circuit.num_clbits:
        lines.append(f"CREG {circuit.num_clbits}")
    for gate in circuit.gates:
        name, qs, ps = gate.name, gate.qubits, gate.params
        qargs = ", ".join(f"q[{q}]" for q in qs)
        gate_name = _ORIGINQ_GATE[name]
        if ps:
            lines.append(f"{gate_name}({_fmt_params(ps)}) {qargs}")
        else:
            lines.append(f"{gate_name} {qargs}")
    for m in circuit.measures:
        lines.append(f"MEASURE q[{m.qubit_index}], c[{m.clbit_index}]")
    return "\n".join(lines) + "\n"


def transpile_to_target(circuit: Circuit, target: str) -> str:
    """Render a parsed circuit to the target backend's native IR text."""
    if target == "spinq":
        return _render_spinq(circuit)
    if target == "braket":
        return _render_braket(circuit)
    if target == "originq":
        return _render_originq(circuit)
    raise ValueError(f"unsupported target {target!r}")
