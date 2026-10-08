# 打卡文案 · Issue #178（Task5 · 量化与显存容量扩展）

> 提交方式：把下面的正文粘贴到 https://github.com/datawhalechina/llm-algo-leetcode/issues/178 的评论框，
> 并上传"测试通过截图"（推荐 evidence/task5/task5_runshot.png 或任一证据图 + 任一 solved notebook 输出页）。
> 微信群截图需包含：issue 标题、GitHub ID、微信群昵称。
> 注意：issue 截止时间为 09-28 12:00，本帖为截止后补交（issue 已被官方标记 completed）。

---

### 打卡：llm-algo-leetcode 显存优化 Task5 · 量化与显存容量扩展

- GitHub ID：wsm000 ｜ 微信群昵称：（填写）
- 打卡选择：**4.1 + 4.2 + 4.3（全量）**
- 环境：vm-60 · RTX 4090 D 24 GB · torch 2.9.1+cu128 · transformers 5.16.1 · torchao 0.15.0 · Python 3.10.12
- 学习材料：Part01·21、Part02·25、Part02·40（最小）+ topic05、Part02·41、Part02·66、Part02·67（扩展），六个上游 notebook（commit e518638）均补全 TODO 并在 vm-60 执行通过
- 笔记与证据：见下方"笔记与证据"节

#### 4.1 最小打卡（Part01·21 / Part02·25 / Part02·40）

1. **对象与 workload**：权重、激活、KV Cache 是三种生命周期不同的内存，量化时机与验证指标都不同。实测（Qwen2.5-0.5B 全部 169 矩阵 / 494M 参数）：FP16 942.16 MiB → INT8 per-tensor 471.08（×2.000，scale 仅 676 B）→ INT8 per-channel 472.82（×1.993，MSE 低 18 倍）→ INT4 g128 打包 250.26（×3.765，scale 占 5.9%——名义 4× 实测打折）。KV 每 token：MHA-32kv 512 / GQA-8kv 128 / MQA 16 KiB（fp16）。**不能只看参数量**：同一 7B，短上下文单并发时权重占 99.6%，32K×B32 时 KV 占 90.0%；权重 INT8 后 KV 占比反而升到 94.7%——量化转移瓶颈而非消除。
2. **W8A16 三角色**：INT8 权重=存储本体（1,605,632→802,820 B，-50.0%）；scale=动态范围映射元数据（per-tensor 4 B，per-channel 每行一个也才 +0.4%）；FP16/BF16 激活=计算路径，不量化。反量化发生在进入 matmul 之前，MACs 与 FP16 完全相同——收益在存储与权重读取带宽，不在算力。
3. **为什么不解决 KV OOM**：W8A16 只动静态权重行且省的是固定字节；KV 是随 token 线性增长的另一行。实测：toy-12L 权重省 209.5 MiB 只够多 4,470 token KV；全模型 per-tensor 教学实现 forward 峰值反而 976→1970 MiB（反量化副本），logits MSE 1.84。
4. **GPTQ vs AWQ**：GPTQ 用层输出二阶信息逐列补偿，AWQ 用激活幅度保护显著通道（实测保护 56 个）。上游 40 real_gpu（真实 q_proj + 32 校准 prompt）：权重 MSE 只差 28%，**校准输出 MSE 差 3.8 倍**（GPTQ 5.2352e-04 vs AWQ 1.3635e-04）。校准数据的职责是估计量化参数（激活分布决定哪条通道误差最致命），不是证明质量没掉——它同时是评测集，自评。
5. **压缩比之外**：同 workload 四候选实测（toy-12L GQA-8kv，B=4/prompt 512/gen 64，teacher-forced 同权重）：W8A16 peak -8.0% 但吞吐 -26% → tune；INT4 峰值 +10.5% 且余弦 0.960 → reject（误差预算硬门槛）；W8A16+KV-INT8 KV 存储对半、TPOT +75% → tune。选型 = 误差门槛 → 峰值 → TTFT/TPOT/吞吐 → 元数据 → backend 路径五类证据。

#### 4.2 增项 1（topic05 / Part02·41）

1. **FP8 计算路径 vs KV 量化路径**：前者压前向 GEMM 的权重/激活（误差当步影响输出），后者压跨 step 驻留的 K/V（误差每步被重读累积、无离线校准阶段）。真实 KV 五档实测（layer0，[1,2,13,64]）：INT8-pt ×1.995（MSE_K 7.6e-02）/ INT8-per-row ×1.882 / INT4-per-row ×3.556（MSE_K 4.2、max_err 4.34）/ FP8-e4m3 ×2.00（MSE_K 0.55、max_err 4.0）。K 比 V 难量化 5 个数量级（离群 token），**粒度不是越细越好**（per-row 对 K 几乎无改善）。
2. **weight-only 探针 ≠ KV serving**：torchao FP8 weight-only 探针跑通、shape 正确，但 decision 字段自己写着 `weight_only_api_path_executed`——没碰过 KV Cache。真实 KV 量化需要 backend 在 attention kernel 内完成读量化 KV、按组反量化、动态增长驱逐与 PagedAttention 兼容。
3. **容量与并发**：KV 池 = 24G×0.9 − 权重 419 MiB − 缓冲 75 MiB − 256 MiB = 20.44 GiB；每 token 减半 → 容量精确翻倍（GQA-8kv：327→654 并发@512；167,452→334,904 tokens）。构造性准入至 OOM 实测：FP16 123 请求（20.35 GiB）→ INT8 246 请求，**×2.00**。另需预留激活峰值 243 MiB、分配器 gap 72 MiB 与 scale 元数据。
4. **权重已量化仍 OOM**：按 OOM 随什么增长分线——随累计 token 增长查 KV（dtype/单价/block/前缀驻留/未释放），只在长 prefill 炸查激活峰值与 scheduler token budget。账本侧佐证：INT8 权重后 KV 占比 94.7%，继续压权重几乎无贡献。
5. **backend 无 kernel 的风险**：静默回退浮点（67 契约四态测试：absent/legacy/mismatch/compatible，adapter 不能替代 artifact）、反量化副本推高峰值（976→1970 MiB 实测）、吞吐倒退（TPOT +34%/+75%）。67 Step5 真实 backend（vLLM）未装，`decision=pending`，不报告 serving 数字。

#### 4.3 增项 2（Part02·66 / Part02·67）

1. **66 浮点 baseline**：唯一比较基准 + 契约来源（模型/backend/workload/dtype/batch/并发/warmup/重复/口径九类字段固定）。我固定 toy-12L GQA-8kv、B=4、prompt 512、gen 64、fp16、teacher-forced 固定续续序列、同 seed 同权重、同预热，只改量化一个变量族。
2. **指标分工**：peak memory=装不装得下；KV 占用=并发/上下文上限的直接决定项；TTFT=prefill/加载健康度；TPOT=decode 每步成本（KV 量化代价出现在读取路径）；吞吐=容量收益是否兑现的综合指标；质量=门槛不是排序项（INT4 候选被 0.960 余弦直接 reject）。
3. **峰值降但并发不涨**：实验4 的 W8A16 候选就是活例——peak -8.0% 但 KV 单价没变 → 容量边界没动，吞吐还被反量化拖累。若 KV 也降了仍不涨，限制在准入策略、调度波次、paged 粒度、峰值与驻留之差、backend 是否真走量化路径。**先判断容量瓶颈类型再选机制。**

#### 测试通过截图

<img src="evidence/task5/task5_runshot.png" width="700" />

（另附四张证据图、4 个实验 JSON/日志、六份已执行 notebook 及 GPU 步骤 JSON，均在下方仓库链接）

#### 笔记与证据

- 笔记：`courses/06-llm-algo-leetcode-memory/notes/task05-quantization-and-memory-capacity.md`
- 证据目录：`courses/06-llm-algo-leetcode-memory/evidence/task5/`（4 实验 JSON+日志、6 已执行 notebook + notebook GPU 步骤 JSON、5 张证据图、README）
- 代码：`courses/06-llm-algo-leetcode-memory/code/task5_*.py`（4090D 实测，`ALL_TESTS_PASS`）
- 社区与教程：<https://github.com/datawhalechina/llm-algo-leetcode> ｜ <https://github.com/datawhalechina>

> 本篇为原创整理，未照抄教程内容；所有实测数字注明环境与统计口径，可在证据 JSON/日志复核。为跑通上游 real_gpu 路径修复了 5 处执行路径问题（deepcopy 污染 baseline / del batch 后引用 / WARMUP-ITERS / DynamicCache 下标 / inference_mode version_counter），均不改量化数学，详见笔记第 4 节。
