# LLaMA-Factory × QLoRA 微调实践总报告

> **执行环境**：vm-60（`simin@10.5.100.60`，局域网可直连）/ Ubuntu 22.04 / RTX 4090 D 24GB / 24 核 / 23GB 内存
> **执行日期**：2026-09-05（与 Task5/6 相邻的打卡轮）
> **配套教程**：本目录 `README.md`（启发式教程）+ `vm60_runbook.md`（实操手册，含全部踩坑记录）
> **一句话总览**：以 SwanLab 课程 QLoRA 章节为主线，在 24GB 卡上完成「安装排障 → 显存对照 → 人设微调 → 合并导出 → 本地 MoE 微调 → 真指令数据 → 混比扫描」全链路，全程 8 个模型×配置组合、24 份实测证据。

---

## 1. 环境与约束

| 维度 | 现状 | 对实践的影响 |
|---|---|---|
| 磁盘 | 137G 总量 / 初始仅剩 11G（**92% 已用**） | 第一约束 → 模型选型由磁盘而非显存决定 |
| 网络 | **huggingface.co / github.com 被墙**；hf-mirror / ghfast.top / pypi-tuna 可达 | 全部下载走镜像：`HF_ENDPOINT=https://hf-mirror.com`、`HF_HUB_DISABLE_XET=1`（新版 hub 的 Xet 通道镜像不代理）、克隆走 `ghfast.top` |
| Python | 3.10.12（无 sudo、无 python3-venv） | venv 用 `--without-pip` + get-pip.py 搭建；系统环境零污染 |
| 已有栈 | torch 2.9.1+cu128 / bnb 0.49.2 / peft 0.18.1 系统级 | venv `--system-site-packages` 复用 torch，省 ~5GB 磁盘 |
| 本地模型 | `~/local_models` 有 OLMoE-1B-7B（26G）、gpt2 | OLMoE 档实验零下载 |

**版本决策**：最新 LLaMA-Factory main 要求 **Python ≥3.11**，本机只有 3.10（3.11 venv 无法复用 cp310 的 torch，重拉 cu128 要 ~5GB，磁盘不允许）→ 回退 tag **v0.9.3**（`python_requires>=3.9`）。

**最终版本栈**（venv `~/venvs/llamafactory`，系统环境未动）：
LLaMA-Factory 0.9.3 | torch 2.9.1+cu128 | transformers 4.52.4 | bitsandbytes 0.49.2 | peft 0.15.2 | trl 0.9.6 | datasets 3.6.0 | accelerate 1.7.0 | swanlab 0.6.13 + swanboard 0.1.9b1 + peewee 3.19.0

---

## 2. 六阶段实录

### 阶段 A｜安装排障（解决 6 个真实障碍）

| # | 障碍 | 药方 |
|---|---|---|
| 1 | github 被墙 | `git clone https://ghfast.top/https://github.com/hiyouga/LLaMA-Factory.git` |
| 2 | 无 python3-venv | `python3 -m venv --without-pip --system-site-packages` + `curl get-pip.py | python` |
| 3 | main 要 Python≥3.11 | checkout v0.9.3 tag |
| 4 | 系统 `kernels` 包漏进 venv，与钉住版本的 hub 冲突 | venv 内 transformers 4.52.4 + API 兼容 `kernels.py` shim（Device/LayerRepository/…/get_kernel）+ hub==0.36.2 |
| 5 | transformers 内置 SwanLabCallback 与 swanlab≥0.6 契约冲突（可运行的） | 走 v0.9.3 原生集成：`use_swanlab: true` + `report_to: none` |
| 6 | swanlab local 模式 + swanboard/peewee | `pip install "swanlab[dashboard]" "peewee<4"` |

### 阶段 B｜冒烟：Qwen2.5-0.5B QLoRA（20 步，零下载）

train_loss 2.15 / 35s / **峰值 1738 MiB**——全链路（4bit 加载→LoRA→落盘）首次打通。

### 阶段 C｜对照实验：Qwen2.5-1.5B LoRA vs QLoRA（本报告核心数据）

同 identity 数据集（91 条，人设"小智 / hello-gpu 实验室"）、同 seed、同超参，唯一变量 = 量化开关：

| 指标 | LoRA (bf16) | QLoRA (NF4 4bit) | 结论 |
|---|---|---|---|
| **峰值显存** | 4052 MiB | **2350 MiB** | **省 42%** |
| train_loss | 1.6719 | 1.6892 | 量化误差方差内 |
| eval_loss | 1.2001 | 1.2747 | 4bit 代价 ≈ +0.075 |
| 训练耗时 | 74.6s | 109.7s | 反量化开销 +47% |

**关键方法论发现**（本实践最值钱的教训）：
- **静默失败检测**：第一轮"QLoRA"与 LoRA 的 loss 逐位相同（1.6718808376427852）——顺藤摸瓜发现 v0.9.3 的枚举是 `QuantizationMethod.BNB="bnb"`，yaml 写新版文档的 `quantization_method: bitsandbytes` 会导致 **量化分支无声跳过**（HfArgumentParser 不校验该枚举）。判据：日志必须出现 `Quantizing model to 4 bit with bitsandbytes.`。修复后才是真 QLoRA（2350 MiB）。
- **同 seed 确定性复现**：两组失效 run 逐位相同的 loss，反过来证明了该栈下训练是 deterministic 的——对照实验成立的前提。
- **checkpoint 残留陷阱**：output_dir 残留旧 checkpoint 会让 trainer 恢复后 0 步"假完成"（train_loss=0.0、19.5 步/秒）——对照实验前必须清盘。

### 阶段 D｜人设验收 + 合并导出

- 验收："你是谁？"→"我是 小智…" ✅；"你由谁开发？"→"由智源机器智能…" ⚠️（1.5B × 91 条只能"染"不能"洗"——低秩增量与预训练先验互相妥协）
- 合并：fp16 下 `W'=W+BA`（merge 配置故意无 quantization_bit），产出 2.9G 完整模型 + **自动生成的 ollama Modelfile**

### 阶段 E｜OLMoE-1B-7B：微调本地模型（零下载）

64 专家 / top-8 / 16 层 / **fp32 权重 26G**（`~/local_models/allenai`）：

| 组 | 峰值显存 | train_loss | eval_loss | 耗时 |
|---|---|---|---|---|
| QLoRA NF4 (batch 4×2) | **10306 MiB** | 2.1303 | 2.2537 | 209s |
| LoRA pure_bf16 (batch 2×4) | **14602 MiB** | 2.1606 | 2.1438 | 154s |

MoE 守则：① `lora_target` 别用 `all`（3072 个专家线性层会被全插适配器）；② **bnb 无 fused MoE**，batch 1→4 提速 **3.7×**（23.5→6.3 s/step），小 batch 时吞吐被 launch 开销吃掉；③ `pure_bf16: true` 是 fp32 config 的保命开关（否则 28G OOM）；④ v0.9.3 无 olmoe 模板 → `template: default`。

**P3 终答（跨尺度）**：1.5B 显存差 1.7GB（42%）→ 7B-MoE 差 4.3GB（29%）——QLoRA 红利随规模放大，8B 稠密模型上是"24G 卡分水岭"。

### 阶段 F｜真指令数据 + 混比扫描

**F1 base vs alpaca_gpt4_zh SFT**（42677 条中取 1000 条 + identity 91 条 × 2 epochs，260 步 / 29:41）：
train_loss 2.13 → **1.19**（identity-only 是 2.13）。四问对比：**格式层变化巨大**（base 连对话格式都不知道，输出里自己打出 `User:` 标签；SFT 后学会编号列表）；**知识层变化有限**（"春联是节日"仍然错）——SFT 是行为克隆/格式教学，不是知识注入。

**F2 混比扫描**（固定预算 ≈900 条 × 1 epoch，唯一变量 identity 占比 10/30/50%）：

| 提问 | mix10 | mix30 | mix50 |
|---|---|---|---|
| 你是谁 | ✗ 纯幻觉 | ⚠️ 有影子 | ✓✓ "我是小智…" |
| 谁开发 | ✗ 但**句式对了** | ⚠️ 泛化 | ✓✓ "小智团队开发" |
| 三条建议 | ✓ 编号在 | ✓✓ **最佳** | ✗ 空输出 |
| 解释微调 | ✓✓ **最佳** | ✓✓ | ✗ 空输出 |

1. **甜蜜点在 30%**：50% 时小智高保真但知识任务直接崩（重复 5 遍把人设"背死" + alpaca 只剩 455 条）；
2. **eval_loss 陷阱**：eval_loss 随 identity 占比**单调下降**（1.34→1.21）看似"越高越好"，但 val 集从混比数据切出、重复样本占比同步升高——**横比 eval_loss 被数据构成污染，对话实测才是真相**；
3. **句式迁移先于事实**：10% 档答错但句式已学会（§13 结论三次确认）。

---

## 3. 显存全景表（全部实测）

| 模型 | 规模 | 模式 | 峰值显存 | 备注 |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.5B | QLoRA | 1738 MiB | 冒烟 |
| Qwen2.5-1.5B | 1.5B | QLoRA | **2350 MiB** | 正式 |
| Qwen2.5-1.5B | 1.5B | LoRA bf16 | 4052 MiB | 对照 |
| OLMoE | 6.9B MoE | QLoRA | **10306 MiB** | batch4 |
| OLMoE | 6.9B MoE | LoRA pure_bf16 | 14602 MiB | 对照 |
| OLMoE | 6.9B MoE | QLoRA | 10378 MiB | alpaca SFT |

---

## 4. 方法论收获（可复制到后续任务）

1. **对照实验 = 单一变量 + 证据链**：所有失败/成功都留下日志与显存采样，可回溯
2. **异常值即线索**：逐位相同的 loss 抓住了静默跳过；train_loss=0.0 抓住了残留恢复
3. **评估污染意识**：val 集构成绑定实验设计时，横比指标会撒谎
4. **版本契约三层检查**：框架↔库↔插件（swanlab 案例）——混合环境下优先用框架原生集成
5. **预算思维**：磁盘/显存/时间三本账在动手前算清，全程按预算决断（这正是"账本启发式"的核心）

---

## 5. 证据清单（24 份，全部回传至 `evidence/`）

```
train_smoke.log / mem_smoke.log                        （0.5B 冒烟）
train_qlora.log / mem_qlora.log / train_lora.log / mem_lora.log   （1.5B 对照 ×2）
chat_qlora.txt / export.log                            （验收 + 合并）
train_olmoe_qlora.log / mem_olmoe_qlora.log / train_olmoe_lora.log / mem_olmoe_lora.log  （7B 对照 ×2）
train_olmoe_alpaca.log / mem_olmoe_alpaca.log           （alpaca SFT）
chat_olmoe_compare.txt                                 （base vs SFT 四问）
train_olmoe_mix10/30/50.log ×3 + mem ×3 + chat_mix10/30/50.txt ×3  （混比扫描）
```

配套：`vm60_runbook.md`（§1-§14 全部踩坑与结论）、4 组 yaml 配置、18 个执行/排障脚本（`t*.sh`、`probe*.sh`）全部保留可复现。

---

## 6. 复现索引（一句话版）

```bash
# 0. 环境（vm-60）
bash /tmp/t1a2.sh && bash /tmp/t1b_pip_install.sh && bash /tmp/t1e_fix.sh && bash /tmp/t1h_fix3.sh && bash /tmp/t1i_fix4.sh
# 1. 冒烟 / 2. 对照（Qwen 1.5B）
bash /tmp/t2c_smoke2.sh; bash run_train.sh config/vm60_qlora_qwen2p5_1p5b.yaml mem_qlora.log train_qlora.log
# 3. OLMoE 对照
bash /tmp/t6_smoke_olmoe.sh; bash run_train.sh config/olmoe_qlora.yaml mem_olmoe_qlora.log train_olmoe_qlora.log
# 4. alpaca SFT + 混比
bash /tmp/t8a_prep_alpaca.sh; bash /tmp/t9b_mix_chain.sh
# 5. 验收（对话 + 合并）
bash /tmp/t5_chat.sh; bash /tmp/t5_merge.sh; bash /tmp/t8c_chatcmp.sh; bash /tmp/t10a_chat_mix.sh
```

---

*报告完。全部数字可溯源于 `evidence/` 原始日志，曲线可经 `swanlab watch ~/projects/LLaMA-Factory/swanlog` 复盘。*
