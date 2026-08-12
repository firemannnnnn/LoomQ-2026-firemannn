"""L2 变体压力测试：按 problem_statement 判定标准，用真实 LLM 覆盖三类任务的各种措辞/比特数变体。

生成/纠错 → 提取 QASM → 自研模拟器采样 → Hellinger Fidelity vs 理想分布（阈值 0.97）。
选后端    → 回复须含 backend_capabilities.json 中的规范后端 id。
"""
import math
import re
import sys
import time

sys.path.insert(0, ".")
from loomq_core import parse_qasm, sample_counts
import adapter

# 6 个规范后端 id
VALID_IDS = {
    "spinq_taurus_simulator", "spinq_cloud_qpu",
    "originq_local_simulator", "originq_wukong",
    "braket_local_simulator", "braket_cloud",
}

QASM_BLOCK = re.compile(r"OPENQASM[^`]*?measure[^`]*?;\s*}", re.DOTALL)
QASM_BLOCK2 = re.compile(r"(OPENQASM\s+2\.0;.*?)(?:\n\s*```|\Z)", re.DOTALL)


def extract_qasm(text: str):
    m = QASM_BLOCK.search(text) or QASM_BLOCK2.search(text)
    return m.group(1) if m else None


def hellinger_fidelity(p_counts, q_probs, shots):
    """官方公式：H = 1/√2·√(Σ(√p_i-√q_i)²)，Fidelity = 1 - H。"""
    h2 = 0.0
    for k, q in q_probs.items():
        p = p_counts.get(k, 0) / shots
        h2 += (math.sqrt(p) - math.sqrt(q)) ** 2
    for k, cnt in p_counts.items():
        if k not in q_probs:
            h2 += (math.sqrt(cnt / shots) - 0.0) ** 2
    h = math.sqrt(h2 / 2)
    return 1.0 - h


def ideal_ghz(n):
    return {"0" * n: 0.5, "1" * n: 0.5}


def ideal_bell():
    return {"00": 0.5, "11": 0.5}


def fidelity_of_qasm(qasm, ideal):
    circ = parse_qasm(qasm)
    counts = sample_counts(circ, 8192, seed=42)
    return hellinger_fidelity(counts, ideal, 8192), counts


def run_generate_case(name, prompt, ideal):
    t0 = time.time()
    reply = adapter.agent_chat(prompt)
    qasm = extract_qasm(reply)
    if not qasm:
        return {"name": name, "PASS": False, "why": "无 QASM", "t": round(time.time() - t0, 1)}
    try:
        fid, counts = fidelity_of_qasm(qasm, ideal)
    except Exception as e:
        return {"name": name, "PASS": False, "why": f"模拟失败 {e}", "t": round(time.time() - t0, 1)}
    pass_ = fid >= 0.97
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:2]
    return {"name": name, "PASS": pass_, "fid": round(fid, 4), "top": top,
            "t": round(time.time() - t0, 1)}


def run_backend_case(name, prompt, require_id=None):
    t0 = time.time()
    reply = adapter.agent_chat(prompt)
    found = [i for i in VALID_IDS if i in reply]
    pass_ = bool(found) and (not require_id or require_id in found)
    return {"name": name, "PASS": pass_, "ids": found, "t": round(time.time() - t0, 1)}


# ============ 用例集（三类任务 × 多种措辞/比特数变体） ============
cases = []

# --- 意图生成：GHZ 各种比特数 + 措辞 ---
for n in (3, 4, 5):
    cases.append(run_generate_case(
        f"GHZ-{n} 中文", f"生成一个 {n} 比特的最大纠缠态（GHZ 态），并进行全测量", ideal_ghz(n)))
cases.append(run_generate_case(
    "GHZ-3 口语", "帮我写个三比特的纠缠态电路，全部测量", ideal_ghz(3)))
cases.append(run_generate_case(
    "GHZ-3 English", "Create a 3-qubit GHZ state and measure all qubits", ideal_ghz(3)))
cases.append(run_generate_case(
    "Bell 中文", "帮我生成一个两比特的贝尔态电路并测量", ideal_bell()))
cases.append(run_generate_case(
    "Bell English", "Generate a Bell state circuit for 2 qubits", ideal_bell()))

# --- 代码纠错：目标态贝尔/GHZ + 多种错误 ---
cases.append(run_generate_case(
    "纠错-Bell-大小写", "我想制备一个贝尔态，但这段代码报错了，帮我修好：H q[0]; CX q[0] q[1]（未定义寄存器且门名大小写错误）", ideal_bell()))
cases.append(run_generate_case(
    "纠错-GHZ-漏测量", "请修复这段代码，我想要 GHZ 态：qreg q[3]; h q[0]; cx q[0], q[1]; cx q[0], q[2]", ideal_ghz(3)))
cases.append(run_generate_case(
    "纠错-缺qreg", "这段代码报错帮我修好，目标贝尔态：h q[0]; cx q[0], q[1]; measure q -> c", ideal_bell()))

# --- 智能选后端：比特数/排队/费用变体 ---
cases.append(run_backend_case(
    "选后端-15比特零排队", "我需要运行一个 15 比特电路，且零排队等待，选哪个平台？"))
cases.append(run_backend_case(
    "选后端-大比特", "我要跑 100 比特的电路，应该用哪个后端？"))
cases.append(run_backend_case(
    "选后端-免费本地", "我不想花钱也不需要账号，本地模拟器有哪几个？"))
cases.append(run_backend_case(
    "选后端-真机", "我想上真实量子计算机，哪个平台可以？"))

# ============ 汇总 ============
passed = sum(1 for c in cases if c["PASS"])
print(f"=== L2 变体压力测试：{len(cases)} 例，通过 {passed}/{len(cases)} ===\n")
for c in cases:
    mark = "PASS" if c["PASS"] else "FAIL"
    extra = c.get("fid", "")
    if extra:
        print(f"[{mark}] {c['name']:<22} Fidelity={extra} top={c['top']} t={c['t']}s")
    elif c.get("ids") is not None:
        print(f"[{mark}] {c['name']:<22} ids={c['ids']} t={c['t']}s")
    else:
        print(f"[{mark}] {c['name']:<22} {c.get('why')} t={c['t']}s")
print(f"\n结论: {'全部通过' if passed == len(cases) else '存在失败，需修复'}")
