#!/usr/bin/env python3
"""Submit GHZ-3 to the OriginQ Wukong real superconducting chip and save evidence.

Evidence requirements (see evidence/README.md):
  - raw result.json kept in evidence/files/
  - job id traceable in the OriginQ cloud workbench (taskId from the async API)
  - counts' top-K states must match the ideal GHZ peaks (000 / 111)

Credentials are read ONLY from environment variables (never committed):
  ORIGINQ_API_TOKEN   - API token from https://qcloud.originqc.com.cn workbench
  ORIGINQ_CHIP_ID     - chip id, default 72 (origin_72 Wukong)

Requires Python 3.8-3.11 with pyqpanda:  D:\\ATool\\Python310\\python.exe evidence_submit_originq.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "evidence", "files")
os.makedirs(OUT_DIR, exist_ok=True)

SHOTS = 2048
CHIP_ID = int(os.environ.get("ORIGINQ_CHIP_ID", "72"))
POLL_SEC = 10
TIMEOUT_SEC = 900

GHZ3_QASM = os.path.join(HERE, "circuits", "ghz3.qasm")


def main() -> int:
    token = os.environ.get("ORIGINQ_API_TOKEN", "").strip()
    if not token:
        print("ERROR: set ORIGINQ_API_TOKEN (from qcloud.originqc.com.cn workbench)")
        return 2

    import pyqpanda as pq

    machine = pq.QCloud()
    machine.init_qvm(token, enable_logging=False, log_to_console=False)

    # Build GHZ-3 directly (no parse step needed).
    q = machine.qAlloc_many(3)
    c = machine.cAlloc_many(3)
    prog = pq.QProg()
    prog << pq.H(q[0])
    prog << pq.CNOT(q[0], q[1])
    prog << pq.CNOT(q[1], q[2])
    for i in range(3):
        prog << pq.Measure(q[i], c[i])

    print(f"[originq] submitting GHZ-3 to chip_id={CHIP_ID}, shots={SHOTS} ...")
    task_id = machine.async_real_chip_measure(
        prog, SHOTS, chip_id=CHIP_ID,
        is_amend=True, is_mapping=True, is_optimization=True,
        task_name="LoomQ-2026-GHZ3",
    )
    print(f"[originq] task id: {task_id}")

    # Poll until finished / failed / timeout.
    deadline = time.time() + TIMEOUT_SEC
    probs = None
    while time.time() < deadline:
        status, result = machine.query_task_state_result(task_id)
        if status == machine.TaskStatus.FINISHED.value:
            probs = result
            break
        if status == machine.TaskStatus.FAILED.value:
            print(f"[originq] task {task_id} FAILED")
            return 1
        print(f"[originq] status={status} (WAITING=1/COMPUTING=2), retrying in {POLL_SEC}s ...")
        time.sleep(POLL_SEC)

    if probs is None:
        print(f"[originq] timeout after {TIMEOUT_SEC}s")
        return 1

    # Convert probabilities to integer counts (little-endian bit strings).
    counts = {}
    for key, p in probs.items():
        n = int(round(p * SHOTS))
        if n > 0:
            counts[key] = counts.get(key, 0) + n

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "backend": f"originq_wukong_chip{CHIP_ID}",
        "job_id": task_id,
        "shots": SHOTS,
        "counts": counts,
        "bit_order": "little",
        "timestamp": ts,
        "meta": {
            "platform": "OriginQ Wukong (origin_72)",
            "chip_id": CHIP_ID,
            "circuit": "ghz3",
            "ideal_peaks": ["000", "111"],
            "is_mock": False,
            "raw": probs,
        },
    }

    out_path = os.path.join(OUT_DIR, "originq-wukong-ghz3-result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[originq] saved {out_path}")

    peak_share = sum(counts.get(s, 0) for s in ("000", "111")) / SHOTS
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    print(f"[originq] top states: {top}")
    print(f"[originq] GHZ peak (000+111) share: {peak_share:.1%}")
    ok = peak_share >= 0.5
    print(f"[originq] {'PASS' if ok else 'FAIL'}: dominant peaks present in hardware result")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
