"""Validate transpiled IR executes on the real local SDKs (braket / spinqit / pyqpanda)."""
import os
import shutil
import sys

sys.path.insert(0, ".")
from loomq_core import parse_qasm, transpile_to_target


def _ensure_stdgates():
    """LocalSimulator resolves `include` from cwd; copy the SDK's gate file once."""
    src = os.path.join(
        os.path.dirname(shutil.which("python") or ""),
        "Lib", "site-packages", "braket", "default_simulator", "openqasm", "braket_gates.inc",
    )
    candidates = [
        src,
        r"D:\ATool\Python310\Lib\site-packages\braket\default_simulator\openqasm\braket_gates.inc",
    ]
    if not os.path.exists("stdgates.inc"):
        for cand in candidates:
            if os.path.exists(cand):
                shutil.copy(cand, "stdgates.inc")
                return
        raise FileNotFoundError("braket_gates.inc not found; place stdgates.inc in cwd")


_ensure_stdgates()

QASM = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
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


def check_braket(qasm3):
    from braket.devices import LocalSimulator
    from braket.ir.openqasm import Program
    device = LocalSimulator()
    task = device.run(Program(source=qasm3), shots=1024)
    counts = task.result().measurement_counts
    return dict(counts)


def check_spinq(qasm2):
    import os
    import tempfile
    import spinqit as sq
    from spinqit import BasicSimulatorConfig, get_basic_simulator, get_compiler
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".qasm", delete=False, encoding="utf-8")
    try:
        tmp.write(qasm2)
        tmp.close()
        compiler = get_compiler("qasm")
        ir = compiler.compile(tmp.name, 0)
    finally:
        os.unlink(tmp.name)
    engine = get_basic_simulator()
    config = BasicSimulatorConfig()
    config.configure_shots(1024)
    result = engine.execute(ir, config)
    return {str(k): v for k, v in result.counts.items()}


def check_originq(originir):
    import pyqpanda as pq
    # Parse OriginIR lines ourselves into a QProg.
    machine = pq.CPUQVM()
    machine.init_qvm()
    lines = originir.strip().splitlines()
    qinit = next(int(x.split()[1]) for x in lines if x.startswith("QINIT"))
    creg = next(int(x.split()[1]) for x in lines if x.startswith("CREG"))
    qbits = machine.qAlloc_many(qinit)
    cbits = machine.cAlloc_many(creg)
    prog = pq.QProg()
    for line in lines:
        if line.startswith(("QINIT", "CREG", "MEASURE")):
            continue
        parts = line.replace(",", " ").split()
        name = parts[0]
        if "(" in name:
            gname, arg = name.split("(", 1)
            theta = float(arg.rstrip(")"))
            qubit = int(parts[-1].strip("q[]"))
            if gname == "RZ":
                prog << pq.RZ(qbits[qubit], theta)
            elif gname == "RY":
                prog << pq.RY(qbits[qubit], theta)
            elif gname == "CU1":
                q2 = int(parts[-2].strip("q[]"))
                prog << pq.CU(qbits[qubit], qbits[q2], 0, 0, theta)
        else:
            args = [p.strip("q[]") for p in parts[1:]]
            if name == "H":
                prog << pq.H(qbits[int(args[0])])
            elif name == "X":
                prog << pq.X(qbits[int(args[0])])
            elif name == "S":
                prog << pq.S(qbits[int(args[0])])
            elif name == "SDAG":
                prog << pq.S(qbits[int(args[0])]).dagger()
            elif name == "T":
                prog << pq.T(qbits[int(args[0])])
            elif name == "TDAG":
                prog << pq.T(qbits[int(args[0])]).dagger()
            elif name == "CNOT":
                prog << pq.CNOT(qbits[int(args[0])], qbits[int(args[1])])
            elif name == "SWAP":
                prog << pq.SWAP(qbits[int(args[0])], qbits[int(args[1])])
            elif name == "TOFFOLI":
                prog << pq.Toffoli(qbits[int(args[0])], qbits[int(args[1])], qbits[int(args[2])])
    for m_line in lines:
        if m_line.startswith("MEASURE"):
            parts = m_line.replace(",", " ").split()
            qi = int(parts[1].strip("q[]"))
            ci = int(parts[2].strip("c[]"))
            prog << pq.Measure(qbits[qi], cbits[ci])
    result = machine.run_with_configuration(prog, cbits, 1024)
    n = creg
    return {k if isinstance(k, str) else bin(int(k))[2:].zfill(n): v for k, v in result.items()}


for name, qasm in (("ghz3", QASM), ("ghz5", GHZ5_QASM)):
    circ = parse_qasm(qasm)
    print(f"===== {name} =====")
    try:
        counts = check_braket(transpile_to_target(circ, "braket"))
        print("braket:", counts)
    except Exception as e:
        print("braket FAIL:", type(e).__name__, e)
    try:
        counts = check_spinq(transpile_to_target(circ, "spinq"))
        print("spinq :", counts)
    except Exception as e:
        print("spinq FAIL:", type(e).__name__, e)
    try:
        counts = check_originq(transpile_to_target(circ, "originq"))
        print("originq:", counts)
    except Exception as e:
        print("originq FAIL:", type(e).__name__, e)
