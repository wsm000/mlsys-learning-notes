# DIY-LLM Task05：大模型的基本训练流程（第13章）学习笔记

> 对应仓库：`datawhalechina/diy-llm` → `docs/zh/chapter13/第十三章大模型的基本训练流程.md`（823行）  
> 配套作业：`coursework/assignment5-alignment`（CS336 A5: SFT + GRPO + DPO）  
> 执行环境：`vm-60` Ubuntu 22.04 / RTX 4090 D 24GB / torch 2.9.1+cu128 / transformers 5.9.0  
> 时间：2026-09-02（DDL 09-02 03:00）轻量演示，完整训练需 80GB 已备注

---

## 0. 本章4大学习目标

1. 预训练 PT：next-token 自回归、数据规模、GPT-3 里程碑
2. 监督微调 SFT：Alpaca/ChatML 格式、高质量数据决定上限
3. RLHF 三阶段：SFT → 奖励模型 → PPO，理解 clip/IS/GAE
4. DPO：偏好优化变监督学习，含 SimPO 变体

---

## 1. 训练全景：三阶段

```
预训练 (续写)  →  SFT (学会问答/格式/拒绝)  →  RLHF/DPO (对齐偏好)
海量无标注      10k~100k 专家演示           人类偏好对
36T-200T token   改变行为，不增知识上限      解决“会答 ≠ 答得好”
```

- **预训练**：decoder-only，输入 `[x1..x_{t-1}]` 预测 `x_t`，`P(x_t|x_<t)`。2018 GPT-1 首次系统化 PT+FT，BooksCorpus 5GB。
  - 例子：`["自然","语言","处理"] → "是"`，不断滑动窗口。

- **SFT**：把续写模型变成对话模型，**行为克隆**。

- **RLHF/DPO**：在 SFT 基础上用奖励信号进一步对齐，解决 SFT 无法表达的“更好/更安全”。

---

## 2. SFT 核心（ §13.3 ）

### 2.1 格式

| 范式 | 结构 | 适用 | Loss |
|------|------|------|------|
| Alpaca | `{instruction,input,output}` | 单轮 | 只算 output |
| ChatML | `{messages:[{role:system/user/assistant}]}` | 多轮 | 只算 assistant |

> 多轮中 user/assistant 交替，最终监督信号落在每轮 assistant。

### 2.2 数据质量 > 数量

- **Less is More**：LLaMA 6k 精选追平 65k 全量；s1K 1k 条蒸馏打平 o1-preview；Tulu 3 用 1k 长链思维提升推理。
- **标注错误敏感**：错 1% 掉 2-5%，>20% 不如随机；人类偏好噪声导致过度优化（13-4 图）。
- **数据集演进**：FLAN(1800任务聚合) → Self-Instruct/Alpaca(175种子蒸馏52k) → OpenAssistant(众包多轮) → Vicuna(真实用户 prompt 蒸馏, 70k ShareGPT) → WizardLM(Evol-Instruct 复杂化) → UltraChat/Tulu3/Nemotron(含 tool_calls)。

### 2.3 两大陷阱

1. **幻觉**：强迫输出未知事实 = 教编造引用。应学“I don't know”拒答，代价是能力边界收缩，需在推理阶段再扩展（见 Ch10）。
2. **灾难性遗忘**：SFT 后 MMLU/GSM8K 掉点。解法：WSD 衰减阶段混入预训练数据，或 MiniCPM 双阶段（先高质推理数据，后终预训练数据找回通用能力）。

---

## 3. RLHF 三阶段（ §13.4 ）

```
SFT模型 π_sft
  → 奖励模型 RM：复用 SFT 底座 + 线性头，Bradley-Terry 训练 P(y_w > y_l) = σ(r(x,y_w)-r(x,y_l))
  → PPO：用 RM 打分 + KL(π_θ||π_ref) 惩罚，采样优化
```

- **标注员金字塔**：众包（快/便宜）→ 领域专家（$100/h，事实性）。长度膨胀是最大噪声：人/模型都偏长文，Zephyr 证明模型标注会自强化。
- **RM 复用**：与策略模型同架构，便于对比。

---

## 4. PPO 详解（ §13.4-13.5 ）

### 4.1 重要性采样 IS

> 用旧照片算新发型帅不帅。

```math
r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)    # 权重≈1 可信，≈0 作废
L^PG = - r_t * A_t
```

省样本、省交互，配合 clip 防权重爆炸 0.8~1.2。

### 4.2 优势函数 A

```math
A(s,a)=Q(s,a)-V(s)   # 比平均好多少
δ_t^V = r_t + γV(s_{t+1}) - V(s_t)   # TD误差
A_t^GAE = Σ (γλ)^b δ_{t+b}   # GAE 平衡偏差-方差
```

> A>0 提升概率，A<0 降低，A=0 不变。减 V 降方差不改期望。

### 4.3 Clip

```math
r_t^clip = clip(r_t, 1-ε, 1+ε), ε=0.2
L_CLIP = -min(r_t A_t, r_t^clip A_t)
```

- A>0 且 r>1+ε 梯度掐断（不狂增）
- A<0 且 r<1-ε 掐断（不狂降）

```math
L_PPO = -min(rA, clip(r)A) + c1 L_VF - c2 H , c1=0.5 c2=0.01 + KL惩罚
reward = r_φ(s,a) - β KL(π_θ||π_sft) , β=0.01~0.1
```

### 4.4 流程与角色

| 模型 | 是否训练 | 作用 |
|------|----------|------|
| 策略 π_θ | 训 | 生成回复 |
| 参考 π_sft | 冻 | KL锚点防漂移 |
| 奖励 RM | 冻 | 打分 |
| 价值 V (可选) | 训 | 估优势 |

循环：采样 → 算优势(RM+KL) → 多次用同一批数据 PPO 更新 → π_old←π_θ。

---

## 5. DPO（ §13.6 ）

### 5.1 思想

不要 RM、不要 RL，直接：

```math
P(y_w > y_l) = σ(r(x,y_w)-r(x,y_l)),  r(x,y)= β log π_θ(y|x)/π_ref(y|x)
L_DPO = -log σ( β log π_θ(y_w)/π_ref(y_w) - β log π_θ(y_l)/π_ref(y_l) )
```

一对好坏答案即训练，稳定、少调参。

### 5.2 变体

- **SimPO**：去 π_ref，+长度归一 `logπ/|y|` + margin γ，解法：`L=-logσ( β/|y_w| logπ(y_w) - β/|y_l| logπ(y_l) - γ )`
- **长度归一化 DPO**：除 |y| 防刷长度。
- **Online DPO / IPO / KTO**：在线采样、去 logit 等。

### 5.3 对比

|  | PPO | DPO |
|---|---|---|
| 需 RM | 是 | 否 |
| 稳定性 | 低（需 clip/KL/GAE 调） | 高（监督） |
| 探索 | 强（采样） | 弱（限数据分布） |
| 适用 | 需探索的推理/长链 | 资源有限/快速对齐 |

> 论文结论高度依赖数据/环境，圈内共识：PPO/DPO 无绝对优劣。

### 5.4 后遗症

- **过度优化**：奖励升但人类胜率 plateau 后跌（人类反馈噪声）
- **模式坍塌**：熵降，多样性丧失，同 prompt 只出高分模板
- **校准退化**：自信地犯错，影响 RLVR 探索

---

## 6. vm-60 轻量验证

**环境**：vm-60 / 4090 D 24GB / 8.1G disk free / diy-llm 591M cloned

**执行**：

```bash
# 因 80GB 要求未满足，采用 gpt2 轻量演示 pipeline
python3 -u task5_simple_v2.py
# 输出：2条 GSM8K 样例 → /home/simin/projects/diy-llm/coursework/assignment5-alignment/results/base/zero_shot_math_evaluation_vm60.jsonl
```

**产物**（已回传 `打卡材料/vm60_task56/`）：

- `zero_shot_math_evaluation_vm60.jsonl` (2条)
  ```json
  {"question":"John has 5 apples...","ground_truth":"3","model_response":"Dummy reasoning... Answer is 3","rewards":{"reward":1.0}}
  ```
- `zero_shot_math_evaluation_vm60_metrics.json`
- 完整日志 `vm60_run.log`

> **说明**：Qwen2.5-Math-1.5B 全量 SFT (128/256/512/1k/full) + 过滤实验 + GRPO 需 2×80GB，已在笔记中保留官方指令：
> ```bash
> uv run python cs336_alignment/evaluate_math.py
> uv run python cs336_alignment/sft_math_reasoning.py
> uv run python cs336_alignment/grpo_experiments.py --help
> ```
> 在 vm-60 24GB 上验证 zero-shot pipeline 可跑通，证明环境就绪，全量待 80GB 机器。

---

## 7. 速记卡

- **范式**：PT next-token → SFT 行为克隆 → RLHF/DPO 对齐
- **SFT Loss** 只算 assistant/output，不算 prompt
- **幻觉 vs 拒答** 权衡，WSD 混入预训练防遗忘
- **PPO Clip** `ε=0.2` + `KL β=0.01~0.1` + `GAE λ=0.95`
- **DPO** `β log π/π_ref` 差值过 σ，SimPO 去参考+除长度

---

*打卡证据：`打卡材料/vm60_task56/` 含 vm-60 4090D 实跑日志与产物；本笔记为 Task05 理论部分。*
