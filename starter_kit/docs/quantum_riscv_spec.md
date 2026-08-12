# LoomQ Quantum RISC-V Extension (LQE) — 指令编码规格

> Bonus：自定义量子 RISC-V 扩展指令（+8）
> 配套实现：`starter_kit/quantum_riscv.py`（编码器/解码器）、`starter_kit/riscv_quantum_emulator.py`（官方模拟器 fork + 扩展）
> 端到端测试：`python3 starter_kit/selfcheck_bonus_riscv.py`（48 项，全部通过）

## 1. 设计动机

官方 L3 的 `riscv_emulator.py` 只覆盖经典控制流（`li/add/sub/addi/beq/bne/j`），量子部分以"指令列表字符串"形式与汇编分离。LQE 将量子操作**编码进标准 RISC-V 32 位指令字**，使量子与经典逻辑共用同一条指令流、同一个程序计数器、同一个执行器——量子电路从此可以像普通程序一样：编码、落盘、传输、按地址执行。

## 2. 指令格式

采用 RISC-V 保留的 **custom-0** opcode，不占用标准整数/控制流 opcode 空间：

```
 31        25 24        20 19        15 14        12 11        7 6        0
┌───────────┬────────────┬───────────┬───────────┬───────────┬───────────┐
│  funct7   │    rs2     │    rs1    │  funct3   │    rd     │  opcode   │
│   7 bit   │   5 bit    │   5 bit   │   3 bit   │   5 bit   │ 0b0001011 │
└───────────┴────────────┴───────────┴───────────┴───────────┴───────────┘
```

- `opcode[6:0] = 0b0001011`（custom-0）
- 量子比特索引放在标准 `rd/rs1/rs2` 寄存器字段，取值 x0–x31（本设计中仅用低 8 位）
- 经典测量位映射遵循 L3 规则：`c[k]` → 结果写入 `x10+k`

## 3. funct3 类别

| funct3 | 类别 | 门 |
|---|---|---|
| `000` | 单比特门 | `qh qx qs qsdg qt qtdg`（funct7 选择） |
| `001` | 双比特门 | `qcx qswap`（funct7 选择） |
| `010` | 三比特门 | `qccx` |
| `011` | 参数门 | `qrz`（funct7+rs2 = 12 位角度） |
| `100` | 测量 | `qmeas` |
| `101` | 参数门 | `qry` |
| `110` | 参数门 | `qcu1`（rd=目标, rs1=控制） |
| `111` | 保留 | — |

## 4. 指令编码表

| 助记符 | funct3 | funct7 | rd | rs1 | rs2 | 语义 |
|---|---|---|---|---|---|---|
| `qh xd` | 000 | 0 | qubit | 0 | 0 | H(q[d]) |
| `qx xd` | 000 | 1 | qubit | 0 | 0 | X(q[d]) |
| `qs xd` | 000 | 2 | qubit | 0 | 0 | S(q[d]) |
| `qsdg xd` | 000 | 3 | qubit | 0 | 0 | S†(q[d]) |
| `qt xd` | 000 | 4 | qubit | 0 | 0 | T(q[d]) |
| `qtdg xd` | 000 | 5 | qubit | 0 | 0 | T†(q[d]) |
| `qcx xd, xs` | 001 | 0 | 目标 | 控制 | 0 | CX(q[s] → q[d]) |
| `qswap xa, xb` | 001 | 1 | b | a | 0 | SWAP(q[a], q[b]) |
| `qccx xd, xa, xb` | 010 | 0 | 目标 | ctrl0 | ctrl1 | CCX(q[a], q[b] → q[d]) |
| `qrz xd, θ` | 011 | imm[11:5] | qubit | 0 | imm[4:0] | RZ(q[d], θ) |
| `qmeas xd, xs` | 100 | 0 | qubit | cbit | 0 | 测量 q[d] → 结果写 x10+s |
| `qry xd, θ` | 101 | imm[11:5] | qubit | 0 | imm[4:0] | RY(q[d], θ) |
| `qcu1 xd, xs, θ` | 110 | imm[11:5] | 目标 | 控制 | imm[4:0] | CU1(q[s] → q[d], θ) |

## 5. 角度编码（参数门）

- 角度以**弧度**在汇编层书写；编码为 12 位有符号整数：`imm = round(θ × 128 / π)`
- `imm` 的存放：`funct7 = imm[11:5]`（7 位）、`rs2 = imm[4:0]`（5 位）
- 解码：`θ = imm × π / 128`；量化步长 π/128 ≈ 1.4°，范围 ±16π
- 示例：`qrz x1, 1.5707963267948966`（π/2）→ `imm = 64` → word `0x0400308b`

## 6. 汇编语法示例

```asm
# GHZ-3：编码为 4 条 custom-0 指令
qh x0
qcx x1, x0
qcx x2, x1
qmeas x0, x0
qmeas x1, x1
qmeas x2, x2
```

## 7. 端到端使用流程

```bash
# ① 编码：L3 量子操作列表 → LQE 32 位字
python3 -c "
from l3_compiler import compile_hybrid
from quantum_riscv import encode_quantum_ops
ops, _ = compile_hybrid('''...Hybrid-QASM...''')
words = encode_quantum_ops(ops)
print([hex(w) for w in words])"

# ② 执行：LQE 扩展模拟器（load_program 汇编 / load_binary 二进制皆可）
python3 -c "
from riscv_quantum_emulator import TinyQuantumRISCVEmulator
emu = TinyQuantumRISCVEmulator()
emu.load_program('qh x0\nqcx x1, x0\nqmeas x0, x0\nqmeas x1, x1\n')
print(emu.execute())"

# ③ 完整回归（48 项，含编码规格 / 态演化参考验证 / 二进制加载 / 混合流水线 / 官方子集兼容）
python3 selfcheck_bonus_riscv.py
```

## 8. 兼容性与可复现性

- 扩展模拟器**完整保留官方 API**（`load_program/execute/set_register/get_register`）与官方指令子集，官方 L3 用例（`selfcheck_l3.py` 4 例）在扩展模拟器上全部通过
- 状态向量演化**确定性强**：量子门精确酉演化（无采样），仅 `qmeas` 引入采样且种子可固定（`set_quantum_seed`），测试可复现
- 量子态与门定义遵循 qelib1 标准位序（`q[k]` 对应状态索引的第 k 位，与大赛 `bit_order="little"` 一致）
