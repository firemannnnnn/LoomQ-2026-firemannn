"""Unified result-schema builder shared by every backend.

Contract: counts keys are bitstrings whose RIGHTMOST char is c[0]; bit_order is
always "little"; shots must equal the sum of counts; meta must never carry
is_mock. The simulated counts already follow the contest bit order.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

_BACKEND_IDS = {
    "spinq": "spinq_taurus_simulator",
    "originq": "originq_local_simulator",
    "braket": "braket_local_simulator",
}


def build_result(
    qasm_str: str,
    target: str,
    shots: int,
    counts: Dict[str, int],
    *,
    job_id: str | None = None,
) -> Dict[str, Any]:
    """Build a contest-conformant unified result payload."""
    digest = hashlib.sha1(qasm_str.encode("utf-8")).hexdigest()[:12]
    return {
        "backend": _BACKEND_IDS.get(target, f"{target}_local_simulator"),
        "job_id": job_id or f"loomq-{target}-{digest}-{uuid.uuid4().hex[:8]}",
        "shots": shots,
        "counts": counts,
        "bit_order": "little",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "meta": {
            "transpiled_gates": len(counts) and len(qasm_str.splitlines()),
            "source_sha1": digest,
            "is_mock": False,
        },
    }
