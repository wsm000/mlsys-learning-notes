# 打卡文案 · Issue #177（Task4 · 推理侧 KV Cache 与显存容量）

> 提交方式：把下面的正文粘贴到 https://github.com/datawhalechina/llm-algo-leetcode/issues/177 的评论框，
> 并上传"测试通过截图"（推荐 evidence/task4/task4_runshot.png 或任一证据图 + 任一 solved notebook 输出页）。
> 微信群截图需包含：issue 标题、GitHub ID、微信群昵称。
> 注意：issue 截止时间为 09-26 12:00，本帖为截止后补交（issue 仍 open）。

---

### 打卡：llm-algo-leetcode 显存优化 Task4 · 推理侧 KV Cache 与显存容量

- GitHub ID：wsm000 ｜ 微信群昵称：（填写）
- 打卡选择：**4.1 + 4.2 + 4.3（全量）**
- 环境：vm-60 · RTX 4090 D 24 GB · torch 2.9.1+cu128 · transformers 5.16.1
- 学习材料：Part01·11、Part02·22、Part02·34（最小）+ topic04、Part02·24、Part02·66、Part02·69（扩展），六个上游 notebook（commit e518638）均补全 TODO 并在 vm-60 执行通过
- 笔记与证据：见下方"笔记与证据"节

#### 4.1 最小打卡（Part01·11 / Part02·22 / Part02·34）

1. **增长关系**：`KV bytes = 2 × L × H_kv × D × tokens × dtype_bytes × B`。batch、上下文长度、层数、KV heads、dtype 字节数**全部严格线性**（倍增实验比率恰为 2.0；4 组配置实测与公式逐字节相等；真实模型 MHA 96.0 / GQA-4 48.0 / GQA-2 24.0 / MQA 12.0 MiB，B=4/S=512/12L/fp16）。KV heads 是唯一可"改结构线性压缩"的维度。
2. **连续预分配 OOM**：按 max_len 预留连续段，同时产生预留尾部、外部碎片、不可共享三重浪费。真机构造性实验：交错释放后空闲 9.35 GiB，申请 4.63 GiB **单连续块失败**，按 8 MiB 逐块分配同样总量成功。同一请求分布下分页（bs=16）4101 MiB vs 连续预留 38390.6 MiB（省 89.3% token，尾块浪费仅 1.7%）。PagedAttention 用"逻辑连续、物理离散、按需增长 + 块表引用计数"缓解，gather 回逻辑序列后注意力输出与连续存放逐位一致（max_diff=0）。
3. **block size**：扫描 1→256：bs=1 块表 86,040 条/零碎片（元数据与间接寻址爆炸），bs=256 仅 466 条但内部碎片 38.7%（退化回连续预留）。bs=16 是折中（5,468 条 / 1.7% 碎片）。不能无限小（管理开销压倒收益）也不能无限大（碎片与调度粒度恶化）。
4. **Prefix Cache**：复用命中前缀**所有层**已算好的 K/V 及物理块映射，跳过其 QKV 投影/attention/MLP/logits。prompt=4096 实测整段 prefill 22.8 ms → suffix-only 12.0 ms（TTFT -47.6%），末位 logits 最大差 0.005（fp16 噪声）；6 请求共享 512 token prompt：KV 存储 180 → 60 MiB（-66.7%，若 backend 支持块共享）。**必须连续前缀**：首段不同命中 0、中段不同只命中分叉前共享段——因果注意力和 RoPE 位置绑定决定链式哈希从分叉点即失效。

#### 4.2 增项 1（topic04 / Part02·24）

1. **容量规划**：`KV 池 = 总显存×0.9 − 上下文 − 权重 − 运行缓冲 − 256MiB 余量`，`T = 池÷每 token 字节`，`B_max = T÷S`。4090D 实测：权重 3.09 GiB（真实 1.5B 模型）、运行缓冲 189 MiB → KV 池 17.65 GiB。MHA(kv=16) 192 KiB/token→23 并发@4096；GQA-8→47；MQA→376；MLA→167；GQA-8+FP8→94。8B 账本：MHA 512 KiB→8、GQA-8 128 KiB→35。
2. **RadixAttention**：压缩基数树按最长公共前缀分裂边，多分支对话只存一份共享 stem；固定 Prefix Cache 每条前缀独立登记，公共部分重复。代价：树分裂/匹配/驱逐的维护与引用管理。
3. **命中率不够**：同机制两种规模结论相反——prompt=512 命中率 66.7% 但 TTFT 零收益（11.96 vs 12.18 ms，launch-bound），prompt=4096 收益 -47.6%（compute-bound）。还必须看峰值显存（缓存常驻 +24 MiB）、维护开销（隔离 2.9 ms/请求）与 P95/P99（无界缓存 P99 15.55→31.52 ms）。

#### 4.3 增项 2（Part02·66 / Part02·69）

1. **共同口径**：66 的 workload 契约（模型/backend、batch、prompt/gen、dtype、cache policy、并发、预热与重复、超时）原样固定，69 只改 cache policy 一个变量；我固定 toy-12L、12 请求、512 prompt/128 suffix/24 gen、fp16、sequential、同 seed 集。
2. **TTFT 改善但 peak 升、P99 差的候选**：实测无界 prefix cache 命中率 1.0，但 TTFT +0.93 ms、P99 +15.97 ms、峰值 +24 MiB → **reject**；LRU 封顶版 TTFT -0.71 ms、P99 最优、峰值仍 +24 MiB → **tune**。规则：先质量门槛，再峰值预算，再 TTFT/吞吐，最后 P95/P99。
3. **四机制分工**：PagedAttention 治分配浪费（省 89.3% token），Prefix Cache 治重复（TTFT -47.6%/存储 -66.7%），KV 量化与 GQA/MQA/MLA 治每 token 单价（192→12 KiB；FP8 减半、K 最大误差 0.123，INT8 误差 0.012）。分页与复用/量化**可组合**（GQA-8+FP8=48 KiB，容量×4），但**互不替代**——先判断容量瓶颈类型再选机制。

#### 测试通过截图

<img src="evidence/task4/task4_runshot.png" width="700" />

（另附六个 solved notebook 执行输出、四张证据图、四个实验 JSON/日志，均在本帖附件/下方仓库链接）

#### 笔记与证据

- 笔记：`courses/06-llm-algo-leetcode-memory/notes/task04-inference-kv-cache-and-capacity.md`
- 证据目录：`courses/06-llm-algo-leetcode-memory/evidence/task4/`（4 个实验 JSON + 日志、6 个已执行 notebook、5 张证据图）
- 代码：`courses/06-llm-algo-leetcode-memory/code/task4_*.py`（4090D 实测，`ALL_TESTS_PASS`）
- 社区与教程：<https://github.com/datawhalechina/llm-algo-leetcode> ｜ <https://github.com/datawhalechina>

> 本篇为原创整理，未照抄教程内容；所有实测数字注明环境与统计口径，可在证据 JSON/日志复核。
