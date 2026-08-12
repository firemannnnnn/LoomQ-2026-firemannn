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

_SELECT_HINT = re.compile(
    r"选|推荐|哪个|平台|后端|backend|simulator|queue|排队|比特|qubit|qubits|免费|费用|cost",
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
    return bool(_SELECT_HINT.search(prompt))


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
    return _call(messages)


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
