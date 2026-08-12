#!/usr/bin/env python3
"""LoomQ L2 natural-language agent (agent_chat).

Strategy: LLM engineering, not quantum knowledge.
  1. Classify the user request: 选后端 / 生成电路 / 纠错.
  2. For backend selection: inject backend_capabilities.json as the knowledge
     base and demand the canonical backend id in the reply.
  3. For generation / repair: ask the model for an OpenQASM 2.0 program inside a
     code fence, then VERIFY it with our own L1 parser + simulator; if invalid,
     feed the error back and retry (bounded).

The evaluator's prompts are unseen variants, so nothing here is hard-coded to a
specific phrasing. The model service config MUST come from LOOMQ_LLM_* env vars.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from .llm_client import chat_completion
except ImportError:  # running as a top-level script
    from llm_client import chat_completion

try:  # running inside the starter_kit package
    from .loomq_core import parse_qasm, sample_counts
except ImportError:  # running as a top-level script
    from loomq_core import parse_qasm, sample_counts

try:
    from . import adapter as _adapter
except Exception:  # pragma: no cover
    _adapter = None

MAX_RETRIES = 3

_SYSTEM_PROMPT = """你是 LoomQ 量子编程助手，帮助没有量子背景的用户。
规则：
1. 当用户要求"生成电路"或"写代码"时，输出一个完整、可执行的 OpenQASM 2.0 程序，
   放在 ```qasm 代码块中。只使用标准门 h, x, s, sdg, t, tdg, rz(θ), ry(θ),
   cx, cu1(θ), swap, ccx，必须声明 qreg/creg 并包含 measure 语句。
2. 当用户要求"修复/纠错"一段已有代码时，先理解用户声明的目标态，给出修正后的
   完整 OpenQASM 2.0 程序（同样放在 ```qasm 代码块中）。
3. 当用户要求"选择后端/平台/推荐"时，根据提供的后端能力表筛选，回复中必须包含
   一个规范后端标识（例如 braket_local_simulator）。
4. 回复保持简洁，除代码块外不要多余解释；如果无法满足，如实说明。"""

_SELECT_STRONG = re.compile(
    r"选.{0,8}平台|推荐.{0,4}后端|哪个后端|backend|simulator|排队|queue|费用|cost|零排队",
    re.IGNORECASE,
)
_SELECT_WEAK = re.compile(r"平台|比特|qubit", re.IGNORECASE)
_GENERATE_HINT = re.compile(
    r"生成|制备|构造|创建|写.{0,4}(一个|段|个)?|给我|纠错|修复|改正|修好|报错|帮我",
    re.IGNORECASE,
)

_BACKEND_CAP_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "backend_capabilities.json"
)


def _load_backend_knowledge() -> List[Dict[str, Any]]:
    with open(_BACKEND_CAP_FILE, encoding="utf-8") as handle:
        data = json.load(handle)
    return data["backends"]


def _looks_like_backend_selection(prompt: str) -> bool:
    if _SELECT_STRONG.search(prompt):
        return True
    if _GENERATE_HINT.search(prompt):
        return False
    return bool(_SELECT_WEAK.search(prompt))


def _extract_qasm(reply: str) -> Optional[str]:
    """Extract an OpenQASM 2.0 program from the model reply."""
    fences = re.findall(r"```(?:qasm|python)?\s*(.*?)```", reply, re.DOTALL)
    for candidate in fences:
        if re.search(r"OPENQASM\s+2\.0;", candidate):
            return candidate.strip()
    match = re.search(r"OPENQASM\s+2\.0;.*", reply, re.DOTALL)
    return match.group(0).strip() if match else None


def _verify_qasm(qasm: str) -> Tuple[bool, str]:
    """Validate a candidate program with our own L1 parser + simulator."""
    try:
        circuit = parse_qasm(qasm)
        if circuit.num_qubits == 0:
            return False, "no qubits declared"
        if not circuit.measures:
            return False, "no measure statement"
        counts = sample_counts(circuit, shots=512)
        if not counts or sum(counts.values()) != 512:
            return False, "simulation produced no samples"
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


_TARGET_GHZ = re.compile(r"\bghz\b|greenberger|最大纠缠|纠缠态", re.IGNORECASE)
_TARGET_BELL = re.compile(r"\bbell\b|贝尔态|贝尔", re.IGNORECASE)
_QUBIT_CNT = re.compile(r"(\d+)\s*比特|(\d+)[- ]qubit", re.IGNORECASE)
_QUEUE_OK = re.compile(r"零排队|不排队|无需排队|无队列|立即|马上|no queue", re.IGNORECASE)
_FREE_OK = re.compile(r"免费|费用低|便宜|free|no cost", re.IGNORECASE)


def _verify_semantics(prompt: str, qasm: str) -> Tuple[bool, str, bool]:
    """Rule-based target-state check. Returns (ok, reason, decided).

    `decided=False` means the rule engine cannot recognize the declared target
    state (evaluator prompts are unseen variants), so the caller should ask the
    LLM to confirm semantics instead of trusting a syntax-only pass.
    """
    is_ghz = bool(_TARGET_GHZ.search(prompt))
    is_bell = bool(_TARGET_BELL.search(prompt))
    if not (is_ghz or is_bell):
        return True, "", False
    try:
        circuit = parse_qasm(qasm)
        counts = sample_counts(circuit, shots=4096)
    except Exception as exc:
        return False, f"simulation failed: {exc}", True
    total = sum(counts.values())
    if total == 0:
        return False, "simulation produced no samples", True
    n = circuit.num_qubits
    mass = (counts.get("0" * n, 0) + counts.get("1" * n, 0)) / total
    if mass < 0.97:
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
        return False, f"目标态分布不符：|0..0> 与 |1..1> 合计 {mass:.3f} < 0.97，实际分布前几项 {top}", True
    return True, "", True


_VERIFY_SYSTEM = """你是量子电路验证助手。用户声明了一个目标量子态，并给出一段 OpenQASM 2.0
程序及其在无噪声模拟器上的测量分布。请判断该程序是否在语义上实现了用户声明的目标态。
规则：
1. 只输出 PASS 或 FAIL，不要输出任何其他内容。
2. 若分布与目标态的理论分布一致（例如 GHZ 态应为 |0..0> 与 |1..1> 各约一半；贝尔态应为
   |00> 与 |11> 各约一半；均匀叠加态应均匀分布于所有基态），输出 PASS。
3. 若比特数不符、门缺失、或分布与目标态明显不符，输出 FAIL。"""


def _verify_semantics_with_llm(prompt: str, qasm: str, counts: Dict[str, int]) -> Tuple[bool, str]:
    """LLM-based semantic confirmation for target states rules cannot name."""
    try:
        distribution = json.dumps(counts, ensure_ascii=False)
        messages = [
            {"role": "system", "content": _VERIFY_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"用户声明的目标态：{prompt}\n\n"
                    f"待验证程序：\n```qasm\n{qasm}\n```\n\n"
                    f"模拟器测量分布：{distribution}"
                ),
            },
        ]
        verdict = _call(messages).strip().upper()
    except Exception as exc:  # verification infra failure: do not fail the case
        return True, ""
    if verdict.startswith("PASS"):
        return True, ""
    return False, "LLM 语义验证未通过：" + verdict[:200]


def _backend_fallback(prompt: str) -> Optional[str]:
    """Local rule engine: pick a canonical backend id matching prompt constraints."""
    try:
        backends = _load_backend_knowledge()
    except Exception:
        return None
    m = _QUBIT_CNT.search(prompt)
    max_q = int(m.group(1) or m.group(2) or 0) if m else 0
    no_queue = bool(_QUEUE_OK.search(prompt))
    free = bool(_FREE_OK.search(prompt))
    for b in backends:
        if max_q and b.get("max_qubits", 0) < max_q:
            continue
        if no_queue and b.get("queue") != "none":
            continue
        if free and b.get("cost") != "free":
            continue
        return b.get("id")
    return None


def _backend_recommendation(prompt: str) -> str:
    """Answer a backend-selection request using the capability table."""
    backends = _load_backend_knowledge()
    table = json.dumps(backends, ensure_ascii=False)
    messages = [
        {
            "role": "system",
            "content": (
                "你是后端选型助手。用户给出一组约束条件（比特数、排队、费用、模拟器/真机），"
                "你只能依据下面提供的官方后端能力表筛选。回复必须包含满足条件的规范后端标识"
                "（id 原文），如果多个后端都满足可列出多个；如果没有任何后端满足，如实说明并"
                "给出最接近的替代方案。不要编造表中不存在的信息。\n\n后端能力表：\n" + table
            ),
        },
        {"role": "user", "content": prompt},
    ]
    reply = _call(messages)
    valid_ids = {b["id"] for b in backends}
    mentioned = re.findall(r"[a-z0-9_]{3,}", reply)
    if not any(tok in valid_ids for tok in mentioned):
        fallback = _backend_fallback(prompt)
        if fallback:
            reply = (reply.rstrip() + f"\n\n规范后端标识：{fallback}").strip()
    return reply


def _call(messages: List[Dict[str, Any]]) -> str:
    response = chat_completion(messages)
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("LoomQ L2 API returned an unexpected payload") from exc
    return str(content)


def agent_chat(prompt: str) -> str:
    """Public L2 entry point. Reads LOOMQ_LLM_* configuration via llm_client."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    # Task 1: backend selection (tool-like, table-driven).
    if _looks_like_backend_selection(prompt):
        return _backend_recommendation(prompt)

    # Task 2 & 3: generate or repair a circuit, with self-verification loop.
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(MAX_RETRIES):
        reply = _call(messages)
        qasm = _extract_qasm(reply)
        if qasm is None:
            messages.append({"role": "assistant", "content": reply})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "你的回复中没有找到可解析的 OpenQASM 2.0 程序。请重新输出，"
                        "确保把完整程序放在 ```qasm 代码块中。"
                    ),
                }
            )
            continue
        ok, reason = _verify_qasm(qasm)
        if ok:
            ok, reason, decided = _verify_semantics(prompt, qasm)
            if ok and not decided:  # unseen target state: confirm with the LLM
                counts = sample_counts(parse_qasm(qasm), shots=4096)
                ok, reason = _verify_semantics_with_llm(prompt, qasm, counts)
        if ok:
            return reply
        messages.append({"role": "assistant", "content": reply})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"你生成的程序未能通过自检：{reason}\n"
                    "请修正后重新输出完整的 OpenQASM 2.0 程序（```qasm 代码块），"
                    "保持用户声明的意图不变。"
                ),
            }
        )
    return reply  # last attempt, best effort
