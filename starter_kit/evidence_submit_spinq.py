#!/usr/bin/env python3
"""Submit GHZ-3 to the SpinQ cloud real hardware (superconducting SQC_25 preferred)
and save evidence.

Evidence requirements (see evidence/README.md):
  - raw result.json kept in evidence/files/
  - job id traceable in the SpinQ cloud workbench (task code returned by submit)
  - counts' top-K states must match the ideal GHZ peaks (000 / 111)

SpinQ cloud uses RSA signature auth: you upload the PUBLIC key at cloud.spinq.cn
and sign requests locally with the PRIVATE key file. Credentials are read ONLY
from environment variables (never committed):
  SPINQ_CLOUD_USERNAME  - account name registered at cloud.spinq.cn
  SPINQ_CLOUD_KEYFILE   - path to the RSA PRIVATE key file
  SPINQ_CLOUD_HOST      - default http://cloud.spinq.cn:6060
  SPINQ_PLATFORM        - optional platform code override (e.g. sqc_25_vp)

Requires Python 3.10 with spinqit:
  D:\\ATool\\Python310\\python.exe evidence_submit_spinq.py
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "evidence", "files")
os.makedirs(OUT_DIR, exist_ok=True)

SHOTS = 2048
HOST = os.environ.get("SPINQ_CLOUD_HOST", "http://cloud.spinq.cn:6060")
PLATFORM_OVERRIDE = os.environ.get("SPINQ_PLATFORM", "").strip()

# SpinQ cloud measures automatically at the end; explicit measure is rejected.
GHZ3_NO_MEASURE_QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
"""

# Platform preference for a 3-qubit circuit on real hardware (machine online).
PREFERRED_REAL = ["sqc_25_vp", "superconductor_vp", "hercules_vp", "triangulum_vp"]


def _pick_platform(backend, platforms):
    """First preferred real platform that has an online machine and >=3 qubits."""
    for code in PREFERRED_REAL:
        try:
            p = backend.get_platform(code)
        except Exception:
            continue
        if not p.simu and p.machine_count > 0 and p.max_bitnum >= 3:
            return p
    for p in platforms:
        if not p.simu and p.machine_count > 0 and p.max_bitnum >= 3:
            return p
    return None


def _normalize_counts(raw_counts):
    """Keep only 0/1 digits, pad to circuit width, aggregate duplicates."""
    width = 3
    out = {}
    for key, val in raw_counts.items():
        if isinstance(key, int):
            key = bin(key)[2:].zfill(width)
        key = "".join(ch for ch in str(key) if ch in "01")
        if len(key) < width:
            key = key.zfill(width)
        out[key] = out.get(key, 0) + int(val)
    return out


def main() -> int:
    username = os.environ.get("SPINQ_CLOUD_USERNAME", "").strip()
    keyfile = os.environ.get("SPINQ_CLOUD_KEYFILE", "").strip()
    if not username or not keyfile or not os.path.exists(keyfile):
        print("ERROR: set SPINQ_CLOUD_USERNAME and SPINQ_CLOUD_KEYFILE (RSA private key path)")
        return 2

    from spinqit import get_compiler
    from spinqit.backend.spinq_cloud_backend import SpinQCloudBackend, SpinQCloudConfig

    print(f"[spinq] logging in as {username} @ {HOST} ...")
    backend = SpinQCloudBackend(username, keyfile, HOST)

    platforms = backend.platforms
    print(f"[spinq] platforms available: {[p.code for p in platforms]}")
    for p in platforms:
        print(f"    {p.code}: name={p.name} maxbits={p.max_bitnum} "
              f"machines={p.machine_count} simu={p.simu}")

    target = None
    if PLATFORM_OVERRIDE:
        target = backend.get_platform(PLATFORM_OVERRIDE)
    else:
        target = _pick_platform(backend, platforms)
    if target is None:
        print("[spinq] no suitable real platform found (need >=3 qubits, not simulator, machine online)")
        return 1
    print(f"[spinq] target platform: {target.code} (name={target.name}, "
          f"machines={target.machine_count}, maxbits={target.max_bitnum})")

    # Compile the QASM (no measure) via the qasm frontend.
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".qasm", delete=False,
                                      encoding="utf-8")
    try:
        tmp.write(GHZ3_NO_MEASURE_QASM)
        tmp.close()
        compiler = get_compiler("qasm")
        ir = compiler.compile(tmp.name, 0)
    finally:
        os.unlink(tmp.name)

    config = SpinQCloudConfig()
    config.configure_platform(target.code)
    config.configure_shots(SHOTS)
    config.configure_task("LoomQ-2026-GHZ3", "GHZ-3 on SpinQ real hardware (LoomQ)")

    print(f"[spinq] submitting GHZ-3 to {target.code} ...")
    result = backend.execute(ir, config)

    counts = _normalize_counts(result.counts)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "backend": f"spinq_cloud_{target.code}",
        "job_id": result.task_code,
        "shots": SHOTS,
        "counts": counts,
        "bit_order": "little",
        "timestamp": ts,
        "meta": {
            "platform": target.code,
            "platform_name": target.name,
            "circuit": "ghz3",
            "ideal_peaks": ["000", "111"],
            "is_mock": False,
            "raw_counts": dict(result.counts),
            "raw_probabilities": dict(result.probabilities or {}),
        },
    }

    out_path = os.path.join(OUT_DIR, "spinq-ghz3-result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[spinq] saved {out_path}")

    peak_share = sum(counts.get(s, 0) for s in ("000", "111")) / SHOTS
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    print(f"[spinq] top states: {top}")
    print(f"[spinq] GHZ peak (000+111) share: {peak_share:.1%}")
    ok = peak_share >= 0.5
    print(f"[spinq] {'PASS' if ok else 'FAIL'}: dominant peaks present in hardware result")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
