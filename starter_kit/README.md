# LoomQ 2026 —— 量子接入平权计划（Team: firemannn）

> 一句话：**用自然语言驱动，把"写量子程序、选平台、看懂结果"这件事的门槛降到零。**
> L1 零依赖状态向量模拟器 + L2 自然语言智能体（LLM 生成/纠错/语义验证闭环）+ L3 混合编译，
> 并已在量旋云真机（triangulum_vp）跑出 GHZ-3 主峰 88.2% 的真机证据。

本目录 `starter_kit/` 是构建与评测根目录。**第一部分是本团队的实现说明（工程叙事），第二部分是官方提交协议**（contract v1.0），两者都以代码为准。

---

## Part 1 —— 本团队实现

### 1.1 为谁降低门槛（必答叙事）

量子计算目前被三道门槛挡在普通开发者门外：

| 门槛 | 传统解法 | 我们的解法 |
|---|---|---|
| **语言门槛**：没写过 QASM | 读手册、学语法 | 自然语言直接生成 + 自动纠错（L2 智能体） |
| **平台门槛**：不知道用哪家 | 挨个注册平台查文档 | 内置 6 后端能力表，按需求智能推荐（免账号/免费/零排队/比特数） |
| **结果门槛**：看不懂测量分布 | 手工数二进制 | 自研模拟器即时跑出分布柱状图 + 语义验证闭环（LLM 确认"代码真的实现了你的意图"） |

目标用户：**没写过一行量子代码的学生、想快速验证量子算法想法的研究者、想比较各平台的开发者**。

### 1.2 架构说明

```
┌────────────────────────────────────────────────────────────┐
│ L2 Web 控制台  web_agent.py  (仅 Python 标准库 http.server)  │
├────────────────────────────────────────────────────────────┤
│ L2 智能体  l2_agent.py                                      │
│   意图分类链 → LLM 生成/纠错 QASM → 自研模拟器语义验证闭环     │
│   (规则快路径 GHZ/Bell → LLM 语义确认 → 失败反馈重试)          │
├────────────────────────────────────────────────────────────┤
│ L1/L3 统一入口  adapter.py                                  │
│   transpile(qasm, target) / run(qasm, target, shots)       │
│   compile_hybrid(hybrid_qasm)                              │
├────────────────────────────────────────────────────────────┤
│ loomq_core/  零第三方依赖的中间层（评测隔离环境可直接运行）      │
│   parser.py      OpenQASM 2.0 → Circuit（12 门白名单）        │
│   simulator.py   状态向量演化 + 8192 次采样                   │
│   transpiler.py  Circuit → spinq(QASM2) / originq(OriginIR)  │
│                   / braket(QASM3) 三目标 IR 渲染             │
│   result.py      统一结果 Schema（backend/job_id/counts/…）  │
├────────────────────────────────────────────────────────────┤
│ L3  l3_compiler.py → riscv_emulator.py（Hybrid-QASM 编译执行）│
│ LLM  llm_client.py（urllib OpenAI-compatible 传输）          │
│ 选后端  backend_capabilities.json（6 后端能力表）              │
└────────────────────────────────────────────────────────────┘
```

**关键设计**：L1/L3 核心（解析、模拟、转译、编译）全部只依赖 Python 标准库，适配组织方隔离构建环境；LLM 只通过 `LOOMQ_LLM_*` 环境变量注入，无硬编码。

**两条数据流**：

```
L1: QASM ──parse──▶ Circuit ──transpile──▶ 目标 IR（spinq/originq/braket）
                        └──run──▶ 状态向量模拟 ──8192次采样──▶ 统一 result.json

L2: 用户提问 ──意图分类──▶ 生成/纠错/选后端/闲聊 四路分支
      生成分支：LLM 产出 QASM ──▶ 模拟器语义验证（规则快路径 / LLM 确认）──▶ 不通过则带报错重试
```

### 1.3 一键运行

干净环境（Python 3.10，无需安装任何第三方包即可运行 L1/L2 核心）：

```bash
# ① 官方公开契约自测（L1/L2/L3）
python3 evaluator.py --level l1 --target spinq,originq,braket
python3 evaluator.py --level l2
python3 evaluator.py --level l3

# ② 团队回归矩阵（全部应为 PASS）
python3 selfcheck_allgates.py        # 12 门白名单逐门保真度
python3 selfcheck_l1_hidden.py       # QFT-4 / Grover-3 / Random×3 隐藏电路（独立 NumPy 参考交叉验证）
python3 selfcheck_l2.py              # L2 六项回归（生成/纠错/选后端/追问/缺配置不泄露 Key）
python3 selfcheck_l2_variants.py     # L2 十四例变体压力测试
python3 selfcheck_l3.py              # L3 混合编译端到端

# ③ L2 交互 Web 控制台（浏览器打开 http://127.0.0.1:8787/）
export LOOMQ_LLM_BASE_URL=https://api.deepseek.com
export LOOMQ_LLM_API_KEY=<YOUR_OWN_KEY>     # 只走环境变量，不写入代码
export LOOMQ_LLM_MODEL=deepseek-v4-flash
python3 web_agent.py --host 0.0.0.0 --port 8787
```

> 未配置 `LOOMQ_LLM_*` 时 Web 页面会明确提示模型未就绪，基于本地能力表的后端推荐仍可离线工作。

真机证据复现（需要 Python 3.10 + 平台凭证，脚本均从环境变量读取凭证，不落盘不提交）：

```bash
# 本源悟空真机
export ORIGINQ_API_TOKEN=<YOUR_TOKEN>
python3 evidence_submit_originq.py

# 量旋云真机（RSA 签名，公钥上传 cloud.spinq.cn，本地用私钥）
export SPINQ_CLOUD_USERNAME=<YOUR_USERNAME>
export SPINQ_CLOUD_KEYFILE=<PATH_TO_PRIVATE_KEY>
python3 evidence_submit_spinq.py
```

### 1.4 现场体验流程（3 个用户任务）

1. **自然语言生成**：输入"生成一个 3 比特的 GHZ 态并进行全测量" → 智能体写出 QASM，自研模拟器实时跑出 000/111 各半的分布柱状图。
2. **代码纠错**：输入"修复报错的贝尔态代码 `H q[0]; CX q[0] q[1]`" → 智能体识别寄存器未定义/大小写错误，给出可运行程序并可视化 00/11。
3. **智能选后端**：输入"我需要运行一个 30 比特电路、免费且免账号" → 依据能力表推荐 `originq_local_simulator`（30q/免费/免账号）。

### 1.5 真机证据

- **量旋云 triangulum_vp（3 比特核磁真机）**：GHZ-3，主峰 000(899)+111(907)=**88.2%**，job `S-260812-0004` 可在量旋云控制台溯源。原始结果见 [`evidence/files/spinq-ghz3-result.json`](evidence/files/spinq-ghz3-result.json)。
- **本源悟空（72 比特超导）**：提交脚本就绪（`evidence_submit_originq.py`），悟空/悟源芯片近期处于平台维护，恢复后自动重试提交。

---

## Part 2 —— 官方提交协议（contract v1.0，保留原文）

### 提交结构

```text
starter_kit/
├── __init__.py
├── VERSION
├── CHANGELOG.md
├── submission.yaml
├── adapter.py
├── llm_client.py
├── l2_policy.json
├── evaluator.py
├── prepare_submission.py
├── riscv_emulator.py
├── backend_capabilities.md
├── backend_capabilities.json
├── QUANTUM_101.md
├── gate_identities.md
├── target_ir_contract.md
├── requirements.txt
├── Dockerfile
├── evidence/
│   ├── README.md
│   └── files/                # 可选附件
├── circuits/
│   ├── bell.qasm
│   └── ghz3.qasm
└── examples/
```

目录名使用下划线，因此从 fork 根目录编写测试时可以按标准 Python 包导入：

```python
from starter_kit import adapter
```

### 环境

公开 evaluator 只使用 Python 标准库，无需安装依赖。推荐 Python 3.10，与官方基础镜像一致（spinqit 最高只提供 cp310 wheel）：

```bash
python3 evaluator.py --level l1 --target spinq,originq --json-out report.json
```

参赛项目使用第三方 SDK 时，必须把依赖写入 `requirements.txt` 并精确锁定版本，例如 `package==1.2.3`。

也可以先验证基础容器：

```bash
docker build -t loomq-submission .
docker run --rm loomq-submission
```

### Adapter 契约

L1 必须实现：

```python
def transpile(qasm_str: str, target: str) -> str: ...
def run(qasm_str: str, target: str, shots: int) -> dict: ...
```

`transpile()` 的三个目标格式规范子集见 `target_ir_contract.md`。正式评测会由组织方解析并模拟返回的目标 IR。

L2、L3 为可选接口：

```python
def agent_chat(prompt: str) -> str: ...
def compile_hybrid(hybrid_qasm_str: str) -> tuple[list, str]: ...
```

### 公开自测

```bash
# 默认只测试 submission.yaml 中声明为 true 的 Level
python3 evaluator.py --json-out report.json
# 单独测试
python3 evaluator.py --level l1 --target spinq,originq,braket
python3 evaluator.py --level l2
python3 evaluator.py --level l3
```

退出码：全部公开测试通过为 `0`，存在失败为 `1`。`report.json` 只表示公开契约自测结果，不是正式分数。

### 最终提交

截止时间为 **2026-08-25 12:00 UTC+8**。先在 fork 根目录运行：

```bash
python3 starter_kit/prepare_submission.py --team-id <GITHUB_USERNAME>
```

预检通过后，在上游 `QAIDAO/LoomQ-2026` 的"LoomQ 最终提交" Issue Form 中填写输出的 fork 地址和 40 位 commit SHA。出现 `submission:accepted` 标签与归档哈希回执后才算提交成功。更新代码后必须新建 Issue，截止前最后一次有效提交生效。

如申报 L1 真机、L2 交互体验、工程与产品化或 Bonus，只需填写 [`evidence/README.md`](evidence/README.md)。证据必须随最终 commit 归档。

### L2 统一模型与环境变量

正式 L2 客观评测统一使用 DeepSeek `deepseek-v4-flash`，最终答案由确定性的官方测试判定，不使用 LLM 充当裁判。组委会在赛前不提供 API 地址、Key 或额度。`agent_chat` 实现不得硬编码 URL、Key 或模型名，必须读取：

| 环境变量 | 含义 |
|---|---|
| `LOOMQ_LLM_BASE_URL` | OpenAI-compatible API 根地址 |
| `LOOMQ_LLM_API_KEY` | 当前运行凭证 |
| `LOOMQ_LLM_MODEL` | 当前模型；正式评测为 `deepseek-v4-flash` |
| `LOOMQ_LLM_TIMEOUT_SECONDS` | 单次请求超时 |

正式限制为每个 case 时限 120 秒；两组固定私有种子共 12 个 case。机器可读版本见 `l2_policy.json`。缺少配置时应立即失败，错误信息不得包含任何 Key。

### 版本政策

合同版本为 `1.0`。开赛后，`1.x` 只允许增加向后兼容的文档、诊断信息和公开测试，不改变已有接口语义；破坏性修改必须发布新的合同版本并为旧版保留评测通道。
