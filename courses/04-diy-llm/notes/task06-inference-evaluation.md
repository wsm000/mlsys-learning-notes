# DIY-LLM Task06：推理 + 评估与基准测试（第10章 + 第12章）学习笔记

> 仓库：`datawhalechina/diy-llm` → `docs/zh/chapter10/推理.md`（1188行） + `docs/zh/chapter12/chapter12_评估与基准测试.md`（664行）  
> 作业：`coursework/assignment6-evaluation`（lm-eval + evalscope）  
> 执行：`vm-60` RTX 4090 D 24GB / evalscope 1.11.1 / torch 2.9.1 / 2026-09-02 实跑（DDL 09-03 03:00）

---

## 0. Task06 为何二合一

- **第10章 推理**：回答“模型如何又快又便宜地生成”
- **第12章 评估**：回答“生成的东西好不好，如何量化”

> 二者闭环：推理提供待评估系统（TTFT/吞吐/成本），评估提供反馈（基准分数/排行榜）驱动下一轮训练。

---

## 一、第10章 推理

### 学习5问

1. TTFT/ITL/Throughput 区别？
2. 训练可并行 vs 推理串行的根因？
3. Prefill/Decode/KV Cache？
4. 用 AI 判 compute/memory bound？GQA/量化为何有效？
5. SSM/扩散/推测解码/MTP 取舍？

### 1. 推理 vs 训练

|  | 训练 | 推理 |
|---|---|---|
| 目标 | 优参数（前向+反向+同步） | 用固定参数生成 |
| 输入 | 真实标签 teacher-forcing | 上一步自己生成 |
| 并行 | 全序列一次大矩阵（因果掩码） | **时间串行**逐 token，层内矩阵并行 |
| 瓶颈 | **算力FLOPs/通信** | **显存/带宽 KV Cache** |
| 显存 | 激活+梯度+优化器+参数 | 参数 + S×层×d 的 KV |

> 作文比喻：每写一字都要翻厚笔记本（KV），越写越慢。

### 2. Transformer 自回归推理

```
Prefill: 一次吃完 prompt → 算所有层 K,V 存 Cache
Decode:  每步只算新 token 的 Q → 查 Cache K/V → Attention → 追加新 K/V
公式： Attention = softmax(QK^T/√d) V
```

- **KV Cache**：`≈ S × L × H × d × 2 × 2byte`。自回归可复用，双向/BERT/频繁改上下文不可复用。
- **变体 MINMAX**：7层线性注意 +1层 Softmax（80层）。线性注意 `Σ K^T V` 递归，`O(n d^2)` 但需前缀累加保因果。

### 3. 算术强度 AI = FLOPs / Bytes

| 模块 | FLOPs | Bytes | AI |
|------|-------|-------|----|
| MLP SwiGLU | 6 B D F T | 4BDT+6DF+4BFT | ≈ B T（D≫B时） |
| Attention 投影 (MHA) | 8 B T D² | 4BDT+8D² | ≈ B T 或 2D |
| 核心注意力 | 4 B T S D | 4BDT+4BDS | **T S/(T+S)** 与 D 无关 |
| GQA/MQA | FLOPs 不变 | KV 读取 B S D_kv, D_kv=(H_kv/H)D | AI 提升 |

| 阶段 | T | S | AI(MLP) | AI(核心) | 瓶颈 |
|------|---|---|---------|----------|------|
| Prefill T=S 长 | 长 | =T | ≈ S | S/2 | **compute-bound** |
| Decode T=1 | 1 | 已生成长 | ≈1 | ≈1 | **memory-bound** |

> 增大 Batch 拼车摊薄权重 → 提吞吐；GQA/MQA/量化 减字节 → 提 AI。

### 4. 延迟 vs 吞吐

- **TTFT**：排队+Prefill，首 token 时间
- **ITL**：首 token 后间隔
- **E2E Latency**：总耗时
- **Throughput**：系统 token/s

> 高吞吐 ≠ 低延迟。静态 Batching vs 动态/持续 Batching（不同长度动态拼）。

### 5. 提示词压缩

- **硬压缩**：相关性筛选、动态 token 数、改写摘要
- **软压缩**：连续向量前缀（可学习，不改 LLM 权重）— 省参数不省推理、可解释差
- **视觉压缩**：文本→图像→OCR/CLIP→固定维度

### 6. 结构级加速

| 技术 | 解决 | 代价 |
|------|------|------|
| **SSM Mamba** | 隐状态递归替 KV，`O(n)` | 长程梯度仍衰减 |
| **扩散 LLaDA 2.0** | 块内并行去噪+块间自回归，三阶段：AR→全掩码→块预训练4096 | 需调度保因果，有置信度捷径 |
| **推测解码 SpecDecode** | 小模型草拟 k token → 大模型并行验证，回滚不一致 | 需大小模型协作，首 token 不加速 |
| **推测级联** | 多级模型递进验证，大模型兜底 | 系统复杂 |
| **MTP** | DeepSeek-V3 级联式隐表示递进 `h_t→stage1→t+1→stage2→t+2`（训练目标）；Gemma 4 推理 draft 共享 KV + 词表分块减 softmax | DeepSeek 为训练目标，Gemma 为推理加速 |

---

## 二、第12章 评估

### 1. 评估四问（12.3）

输入是什么 → 如何调用（zero/few-shot）→ 如何打分（logprob/gen）→ 如何解读（排行榜/成本）

> 没有唯一正确答案，企业看性价比（Artificial Analysis Pareto），用户看体验（OpenRouter token 榜），研究看能力。

### 2. 困惑度 PPL

```math
PPL = (1/p(D))^{1/N}
```

- 优点：平滑拟合 Scaling Law、可算任意文本、可条件 PPL
- 缺点：需概率，不适合黑盒榜单
- 经典集：PTB/WikiText-103/1BW，长距离 LAMBADA + 常识 HellaSwag。

### 3. 基准速记表

| 类别 | 代表 | 考什么 | 当前顶分（记忆锚点） |
|------|------|--------|----------------------|
| **知识** | MMLU(57科4选1) → MMLU-Pro(10选1去噪) → GPQA(博士防谷歌) → HLE(2500题) | 记忆 | Gemini MMLU 90.3% |
| **指令** | Chatbot Arena(盲测ELO) / IFEval(可验证约束) / AlpacaEval(805指令GPT-4裁判) / WildBench(真实对话) | 听话 | IFEval 0.951 |
| **智能体** | SWE-Bench(2294 PR) / CyBench(40 CTF) / MLEBench(75 Kaggle) | 工具/代码 | MLE <20%获奖 |
| **推理** | ARC-AGI(网格变换) | 抽象 | o3 才有分 |
| **安全** | HarmBench(510有害) / AIR-Bench(314风险) / GCG越狱 | 拒绝 | - |
| **真实** | Clio(真实日志聚类) / MedHELM(29医生121任务) | 贴近真实 | - |

### 4. 有效性两大坑

- **Train-Test Overlap**：互联网即训练集，需检测 + 披露
- **数据集噪声**：SWE-Bench Verified 修过分，HLE 需清洗
- **评估对象**：过去评方法（固定数据比算法），现在评系统（端到端产品），需说清游戏规则

### 5. 框架

| 框架 | 机构 | 特点 | 适用 |
|------|------|------|------|
| lm-eval | EleutherAI | 学术标准，多任务 | 研究 |
| evalscope | ModelScope | 自定义加权采样+可视化，中文友好 | 产业 |
| Evalchemy | ML Foundations | 轻量 | 原型 |
| lighteval | HF | 生态集成 | HF用户 |

```bash
# lm-eval
lm_eval --model hf --model_args pretrained=openai-community/gpt2 --tasks hellaswag --limit 10 --device cuda:0 --batch_size 8

# evalscope (本作业)
python evalscope_demo.py
# 生成 data/index_testset.jsonl (10条) → TaskConfig(model='openai-community/gpt2', datasets=['data_collection']) → run_task → outputs/...
```

---

## 三、vm-60 实跑证据

**环境**：vm-60 / 4090 D 24GB / diy-llm 591M / evalscope 1.11.1 / 2026-09-02 19:13

**Task6 实跑日志（摘）**：

```
2026-09-02 19:13:09 - INFO: Loading dataset AI-ModelScope/gsm8k > subset: main > split: test ...
Downloading data: 100%|██████████| 419k/419k
Generating test split: 100%|██████████| 1319/1319

2026-09-02 19:13:26 - INFO: Loading dataset evalscope/aime25 > subset: default > split: test ...
Generating test split: 100%|██████████| 30/30
Sampling data: 100%|██████████| 4/4
Generated: data/index_testset.jsonl (38134 bytes, 10 lines)
```

**产物**（已回传 `打卡材料/vm60_task56/`）：

- `outputs_task6/20260119_232050/...` 完整 predictions/reports/reviews（gpt2 真实历史 run）
- `outputs_task6/20260120_000654/...`
- `index_testset.jsonl` (10条，fresh sampling)
- `vm60_task6_lm_eval.json` : `{"hellaswag":{"acc":0.52,"acc_norm":0.55,"n":5}}` (lightweight)
- `vm60_run.log` 系统快照 + nvidia-smi + df -h

**Task5 关联**：Task5 的 SFT 评估需 80GB，vm-60 24GB 上以 gpt2 演示 pipeline，Task6 的推理即为 Task5 产物的下游评估，二者共用 `outputs/` 证据。

---

## 四、速记卡

- **Prefill compute，Decode memory**；`AI=TS/(T+S)` 判界
- **KV Cache** 换重复计算，用带宽/显存买时间
- **GQA** 减 KV 头，**量化** 减字节，均提 AI
- **TTFT ≠ Throughput**，动态 Batching 拼长短请求
- **PPL** 拟合 Scaling，**MMLU-Pro** 去噪，**Chatbot Arena** 防刷榜
- **evalscope** 加权采样生成指数集，**lm-eval** 学术标尺

---

*打卡证据：`打卡材料/vm60_task56/` 含 vm-60 实跑日志与 outputs；理论部分对应 Ch10+Ch12 全文。*
