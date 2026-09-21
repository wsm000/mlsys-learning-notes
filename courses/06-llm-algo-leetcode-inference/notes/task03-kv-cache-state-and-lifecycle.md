# llm-algo-leetcode 推理优化 | 202609 Task3 · KV Cache 状态与生命周期 —— 学习笔记

- GitHub ID：wsm000 ｜ 微信群昵称：（填写）
- 打卡选择：**(3) 4.1 + 4.2 + 4.3**
- 运行环境：vm-60 · NVIDIA RTX 4090 D（Ada sm89，24 GB）· torch 2.9.1+cu128 · transformers 5.16.1 · Python 3.10.12
- 教程与社区：[datawhalechina/llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode) ｜ [DataWhale 社区](https://github.com/datawhalechina)
- 对应节：Part 02 · 22（vLLM PagedAttention）、Part 02 · 24（SGLang RadixAttention）、04（KV Cache 生命周期与复用）、Part 02 · 34（Prefix Caching 与 Chunked Prefill）、Part 02 · 66（推理性能比较）、Part 02 · 69（Prefix Cache）

> 本篇为个人学习笔记，代码与实验均在 vm-60 实测；观点为原创整理，非教程原文照抄。所有"实测"数字可在 [../evidence/task3](../evidence/task3) 的日志与 JSON 中复核。

---

## 0. 一句话总览

KV Cache 的完整故事是四层递进：**状态保存**（每个 token 的 K/V 常驻显存，Task2 已立账）→ **分页分配**（PagedAttention：逻辑连续、物理离散，按需增长）→ **前缀复用**（RadixAttention / Prefix Cache：相同前缀只算一次）→ **容量治理**（命中、驱逐、重算、量化之间的取舍）。22 和 24 是两种**不同维度**的机制：前者改"怎么放"，后者改"怎么找回来"。

---

## 1. 4.1 最小打卡（Part 02 · 22、Part 02 · 24）

### 1.1 PagedAttention：为什么拆 block？拆出了什么？

**（1）为什么要把连续 KV Cache 拆成固定大小的 block**

连续分配的问题不是"放不下"，而是**放不齐**。自回归生成的长度事先不可知，如果每个请求按最大长度预留一段连续显存，会同时产生三种浪费：

- **预留尾部**：请求实际只生成了 300 token，却按 4096 预留；
- **外部碎片**：每段预留都是整块，池子里空闲总量够，但找不到一段足够大的连续空间；
- **不可共享**：连续段是请求私有的，即使两个请求开头完全相同，也没法共用同一段显存。

PagedAttention 借操作系统虚拟内存的思路：把 KV Cache 切成固定大小的 **block**（16 token/block 是常见值），请求按实际长度**按需申请 block**，物理块可以来自池里任意位置。我的理解：这不是"压缩"，而是把"预留策略"换成"分配策略"——连续分配问"最多要多少"，分页问"现在要多少"。

**（2）解决了连续显存分配中的什么问题（实测）**

我在 vm-60 上用 LLaMA-7B 量级账本（32 层 / 32 KV 头 / head_dim 128 / fp16，每 token 0.5 MiB）实测了同一请求分布（长度 17…2047，共 4080 token）下两种策略的真实 GPU 分配（实验 1，task3_exp1_paged_fragmentation.json）：

| 策略 | 分配 token | 峰值显存 | 利用率 |
|---|---:|---:|---:|
| 分页（block=16） | 4144 | **2072.0 MiB** | 98.5% |
| 连续预留（4096/请求） | 32768 | **16384.0 MiB** | 12.5% |

节省 14312 MiB（87.4%），同等显存预算下**并发数 ×7.9**。注意分页并不是零浪费：每个请求的尾块平均浪费约 block_size/2 个 token（本例共 64 token），这是用"小块内碎片"换掉"整段外碎片 + 预留尾部"的交易。block_size 扫描也印证了权衡：block 从 8 增到 128，尾块浪费从 32 增到 400 token，但块表条目从 514 降到 35——块越小访存越碎、块表越大，块越大碎片越多，工程上 16 是常见折中。

**（3）逻辑 token 位置、物理 KV block 与 block table 的关系**

三层视角要分开看：

- **逻辑层**：请求看到的是一维连续序列 t0 … t4095，attention 按这个顺序算，与显存地址无关；
- **物理层**：显存池是 [num_blocks, block_size, head_dim] 的大张量，块与块之间物理上不连续、可任意分配给任意请求；
- **映射层**：block_table[i] = p 表示"逻辑第 i 个 block 存在物理块 p"，请求的 Request.block_table 就是这个页表。

我在 22 节的实现里把这三层落成了可测代码：allocate_for_prefill 按 ceil(seq_len/block_size) 一次性提交块表（资源不足抛 OOM 且**不留下半分配状态**——测试里的原子 OOM 用例钉住了这一点）；allocate_for_decode 只在 (seq_len+1) % block_size == 1（刚好跨过块边界）时 +1 块，体现"按需增长"；get_physical_cache 按块表顺序拼接物理块并截断到 seq_len，恢复逻辑连续视图。测试还专门造了物理块 [3, 1] 不连续的散乱场景，验证恢复结果遵循**逻辑顺序**而非物理索引排序。

**（4）为什么逻辑连续不要求物理地址连续**

因为 attention 的计算只依赖 token 的**顺序**，不依赖它们在显存里的地址。kernel 只要按块表顺序把 K/V 取出来参与计算，结果与连续存放完全一致。物理不连续换来的是按需分配、请求间共享池、前缀按块共享（引用计数）三种能力；代价是块表寻址、非连续访存和 kernel 复杂度。22 节参考解析也点破了一个细节：真实 PagedAttention 的 kernel 直接在非连续块上算，并不需要先拼成连续张量——拼装只是教学视角。

### 1.2 RadixAttention：前缀树与最长前缀匹配

**（1）如何通过前缀树组织多个请求的 KV Cache**

RadixAttention 把"已算过的 token 前缀"组织成一棵**压缩基数树**（radix tree）：每条边保存一段 token 序列（不是单个 token，所以叫压缩），从根到某节点的路径就是一串已登记的请求前缀；terminal 标记完整请求路径在此结束。插入时沿首 token 找子边、计算最长公共前缀（lcp）：完全匹配就继续向下；部分匹配就把旧边**分裂**成 shared（公共段）+ old_suffix（旧后缀，继承 terminal 与子节点）+ new_suffix（新路径）。这样 [0,1,2,3] 和 [0,1,2,3,4] 只保留一条共享边 [0,1,2,3]，[4] 挂在其下——测试断言了共享边的 token 序列、terminal 标记和子节点挂接。

**（2）最长前缀匹配在请求处理过程中起什么作用**

新请求到来时 match_prefix 从 prompt 开头沿完整共享边向下，返回可复用的最长前缀长度 hit_len；split_prompt 据此把 prompt 切成 hit_prefix（复用，跳过 prefill）和 miss_suffix（重算，并作为新分支登记）。它的作用是把"重复 prefill"变成"一次计算、多次检索"：多轮对话第 k 轮的输入天然包含前 k-1 轮的前缀，树匹配让历史部分的 KV 直接复用，只有新增部分进入计算。

这里有一个**口径细节**值得强调：24 题的 match_prefix 只把 **terminal 节点**（已登记的完整路径）计入可复用长度。我在自加实验 2a 中因此观察到一个现象：共享 256 token system prompt 的 146 个请求里，多轮对话请求命中最长（max 373 token），而只共享 system prompt 的独立请求，除非落在别人已登记的完整路径上，命中为 0。真实 SGLang 的 RadixCache 里每个节点都携带 KV（可 LRU 驱逐），匹配不要求 terminal——这是教学简化与工程实现的差别，读代码时不能把两者混为一谈。

**（3）RadixAttention 与 PagedAttention 解决什么不同**

| 维度 | PagedAttention（vLLM） | RadixAttention（SGLang） |
|---|---|---|
| 核心问题 | KV Cache **怎么放**：连续预留的浪费与碎片 | 已算出的 KV **怎么找回来**：重复前缀的重复计算 |
| 抽象对象 | 物理 block + block table（逻辑↔物理映射） | token 前缀树 + 边共享（公共前缀索引） |
| 关键操作 | allocate / 扩容 / 释放 / 按块表拼装 | insert / split / match_prefix / split_prompt |
| 主要收益 | 容量与并发（实测 ×7.9） | 计算与 TTFT（跳过重复 prefill） |
| 主要代价 | 块表管理、非连续访存、尾块碎片 | 树维护、匹配开销、驱逐策略 |

一句话区分：**PagedAttention 是内存分配管理问题，RadixAttention 是前缀复用问题**。两者正交且互补——真实系统（vLLM 的 automatic prefix caching、SGLang 的 RadixCache）都是"分页 block 做物理管理 + 前缀索引做复用"的组合：block 决定"存哪"，树/索引决定"能不能不算"。

---

## 2. 4.2 增项 1（04 KV Cache 生命周期与复用、Part 02 · 34）

### 2.1 KV Cache 的状态机与触发事件

从请求进入到释放，我把 KV Cache 的生命周期划成五个状态（对应 04 节的"建立、追加、复用、释放"）：

| 状态 | 进入事件 | 离开事件 | 关键量 |
|---|---|---|---|
| ① 未建立 | 请求到达 | prefill 开始 | — |
| ② 建立中（prefill） | 分配初始 block / 前缀命中 | prefill 完成 | prompt tokens、初始 block 数、hit_len |
| ③ 追加中（decode） | 每生成 1 token | 跨过 block 边界时 +1 block；或 EOS/长度上限 | TPOT、扩容次数、峰值显存 |
| ④ 可复用（驻留） | 请求完成 / 前缀被登记 | 被命中（refcount+1）、被驱逐、被清理 | hit_rate、reused tokens |
| ⑤ 释放/驱逐 | 引用归零 / LRU 淘汰 / 显存压力 | 物理块归还空闲池 | 驱逐次数、重算 tokens |

触发关系可以概括为四类事件：**创建**（prefill 分配、decode 跨块扩容）、**命中**（前缀匹配成功，refcount 增加，跳过对应 prefill）、**更新**（写入新 K/V、LRU 时钟刷新）、**驱逐/释放**（容量超限按 LRU 驱逐、请求结束归还）。22 题里的 acquire_prefix/release_prefix 就是这套引用计数的缩小版：命中只加引用不申请新块（测试断言第二次 acquire 不消耗 free_blocks），只有最后一个引用释放才归还物理块。

容量治理的四个动作（保留/驱逐/重算/压缩）是一组显式取舍：驱逐释放容量但下次要重算；重算降低驻留但增加计算；量化降低字节但要验质量。04 节还强调了一个容易忽略的正确性条件：**相同文本 ≠ 可安全共享**——模型 revision、tokenizer、LoRA adapter、RoPE 位置、租户隔离任一变化，缓存都必须失效重算。这一点在 3.2 展开。

### 2.2 Prefix Cache：跳过哪些计算？Chunked Prefill 改什么？

**（1）最长公共前缀如何减少重复 Prefill**

机制链是：登记已算前缀 → 新请求从开头做最长匹配 → 命中的 hit_len 部分直接复用 KV → 未命中 suffix 才进 prefill → 计算结果作为新前缀登记。34 题的 PrefixCacheManager 把这套逻辑做成可测账本：match_prefix 只允许从 prompt 开头连续命中（中间偶然相同的子串不算），cache_stats 给出 hit_tokens / uncached_tokens / reuse_ratio，chunked_suffix_prefill_plan 保证**命中前缀不再进入执行计划**。

**（2）命中后哪些计算可以跳过**

命中长度为 H 的前缀，意味着这 H 个 token 在**所有层**的前向都被跳过：每层的 Q/K/V 投影、attention 打分与加权、MLP，以及最终 logits——因为这些 token 只是"历史"，不影响本次要生成的下一个 token 的分布（它们的影响已经体现在被复用的 K/V 里）。换句话说，prefill 从"整段 prompt"缩成"suffix 一段"，34 题的对照就是 chunked_prefill_plan(全量) vs chunked_suffix_prefill_plan(只含 suffix)。

我在真实模型上验证了这条链路（实验 3，gpt2 124M / fp16 / 4090 D，8 请求共享 800 token system prompt + 200 token 后缀）：开启前缀复用后 **prefill 处理 token 数从 8000 降到 2400（-70%）**，且 G1 与 G0 的 logit 最大差异 0.188、top1-top2 最小边距 0.094——差异同量级于近 tie 边距，个别 argmax 翻转是 fp 噪声而非逻辑错误。**跳过的计算是真实的，输出分布不变也是真实的。**

**（3）Chunked Prefill 为什么拆区段、如何影响调度**

长 suffix 一次性 prefill 会产生很大的单次工作集（激活 + 中间状态），把 decode 请求堵在后面。Chunked Prefill 按固定 block_size 把 suffix 切成多个 chunk，每轮只算一段：单次峰值下降、调度粒度变细、decode 有机会插进相邻 chunk 之间。34 题的 GPU 探针（real_gpu）直接看到了峰值差异：4096 token suffix 一次性处理 peak 16.0 MiB，按 512 切 8 段后 peak 2.0 MiB（1/8）。

对调度关系的影响：prefill 是 compute-bound（并行 token），decode 是 memory-bound（读 KV 与权重），chunk 化后两类任务可以混排进同一 batch，提高 GPU 利用率；代价是调度复杂度、块边界效率损失（尾块不满）和 TTFT 不变（总算量相同，只是切开）。它改变的是 **TPOT 抖动与系统吞吐**，不是 prefill 的总计算量。

---

## 3. 4.3 增项 2（Part 02 · 66、Part 02 · 69）

### 3.1 五个指标各反映什么

| 指标 | 链路位置 | 回答的问题 | 典型恶化原因 |
|---|---|---|---|
| TTFT | 排队 + prefill | 用户等多长时间看到第一个字 | 长 prompt、排队、无前缀复用 |
| TPOT | decode 稳态 | 生成流不流畅、每 token 多慢 | KV 读取带宽、batch 过大 |
| 吞吐 | 系统级 token/s | 单位算力服务多少请求 | batch 不足、调度空转 |
| 峰值显存 | 容量边界 | 能开多少并发、会不会 OOM | 权重 + KV + 激活同时膨胀 |
| P99 | 长尾延迟 | SLO 是否达标（线上 Usually 看它） | 少数长请求、抢占/驱逐重算 |

它们相互冲突：batch 升高吞吐上升但 TTFT、显存同步上升；prefill 变长只推 TTFT；上下文变长三者全推。所以 66 的设计是"固定 workload → 单变量 G0/G1 对照 → 按目标选型"，而不是追单指标最优。我在 vm-60 复跑了 66 解答版（Task2 已补全，本次复跑通过）：CPU 模拟器给出 decode-bound 的判定顺序（decode 占比高 → 先查 KV 读写/调度/投机解码），并把 prefill_share/decode_share 作为 bottleneck 分类依据。

### 3.2 什么是一次"有效"Prefix Cache 命中

有效命中要同时满足三个一致性，缺一个就是错误复用：

1. **token 内容一致**：相同字符串经同一 tokenizer 必须得到相同 token 序列；词表/特殊 token 变了，前缀就不再是同一序列（69 题函数文档原话："相同字符串但 token 序列不同不视为命中"）。
2. **位置一致**：缓存的 K/V 是按位置算的（RoPE 把位置编进 K），position offset 或上下文边界不同，复用出来的 K/V 与真实前向不等价。我的实验 3 用 past_key_values 续接后缀时位置是自动接续的（DynamicCache 长度决定 position_ids），这正是"位置一致"的实现前提。
3. **模型配置一致**：模型 revision、dtype、量化配置、LoRA adapter、采样上下文任一不同，缓存都必须失效。04 节还补了租户隔离：跨用户/权限域禁止共享前缀。

反过来说，**"看起来命中"的三种伪命中**：中间子串相同（不从开头连续匹配）、位置错位后的相同 token 块、换了 tokenizer 的相同文本。69 题的缓存 key 用 (block_index, block) 把**位置**编进 key，就是为了防第二种。

### 3.3 命中率、复用 token、TTFT 改善与维护成本怎么一起算

四个数必须同时记录，因为它们回答不同问题：

- **命中率**（请求口径）：hit_requests / total_requests，回答"多少请求碰到了缓存"；
- **复用 token 数 / token 复用率**（token 口径）：reused_tokens / total_prompt_tokens，回答"省了多少 prefill 计算"——它才与 FLOPs 成正比；
- **TTFT 改善**：ttft_delta_ms = candidate - baseline（负才好），回答"用户是否真的等得更短"；
- **维护成本**：overhead_delta + eviction_count + 缓存常驻显存，回答"这份收益拿什么换的"。

69 题的四个函数就是这条流水线：simulate_prefix_cache（LRU block 模拟，吐出全部字段）→ summarize_prefix_cache（多运行汇总）→ compare_prefix_cache_to_baseline（同 workload_id 才比较）→ recommend_prefix_cache_run（三个门槛全过才 accept）。我的自加实验把这条流水线跑在了真实分布上：

- **实验 2b（LRU 容量扫描，block=16 token）**：容量 128 token 时命中率 0%（连 system prompt 都放不下）；512 token 时命中率 99.3%、token 复用 89.0%，但驱逐 209 次；4096 token 时命中率封顶且驱逐归零。**容量必须大于工作集，收益才存在；驱逐次数就是维护成本的直接度量。**
- **实验 3a（真实模型串行）**：prefill token 省 44~70%，但 TTFT 只快 1.05~1.07 倍——gpt2 124M 的 prefill 被约 9ms 固定开销（kernel 启动、权重读取）摊薄；**实验 3b（批量 8 请求）**：TTFT 38.66 → 11.96 ms（×3.23），峰值显存 1321.5 → 779.5 MiB。批量摊薄固定开销后，prefill 计算占比上升，prefix cache 的 TTFT 收益才直接显形。

**为什么命中率提升不一定值得上线**：命中率是 token/请求口径的"潜力"，不是"到手收益"。① 命中率高的前缀可能很短（省的是小头）；② 缓存要常驻显存，挤压并发预算；③ 驱逐把收益还回去（2b 的容量不足区间）；④ 小请求场景 TTFT 被固定开销主导（3a）；⑤ 错误复用的代价是输出错误，比慢更严重。所以 69 的决策函数要求**命中率增益、TTFT 收益、维护开销三个门槛同时满足**才 accept，且明确"命中率不够时不能仅因 TTFT 改善就 accept"。

**与 66 baseline 的对照方式**：66 提供"固定 workload、固定模型/backend/并发/生成长度，唯一变量换策略"的 G0/G1 框架与 TTFT/TPOT/吞吐/显存/P99 口径；69 把这个框架的唯一变量换成 cache policy（G0 cache off vs G1 prefix cache），并用同一套指标 + 命中证据（backend metrics/日志，**不能只用 TTFT 反推命中率**）输出 accept/tune/reject。两者是同一实验方法在不同策略上的实例化。

---

## 4. 架构变化会改变什么（扩展视角）

- **换 block_size**：碎片与块表开销互换（实验 1 扫描：util 99.2% → 91.1%，块数 514 → 35）。
- **换注意力表示（MHA→GQA→MLA）**：Task2 已实测每 token KV 字节 8:2:1；KV Cache 的全部四层问题（状态、分页、复用、容量）都按同比例缩放。
- **开前缀缓存**：从"每请求独立状态"变成"跨请求共享状态"，容量治理从"归还"变成"驱逐/保留"的动态策略（2b）。
- **PD 分离 / chunked prefill**：改变的是 prefill 与 decode 的调度混排关系，不改变 KV 的状态语义。

## 5. 证据与边界

- **CPU 证据**（22/24/34/69 解答版全部测试通过，vm-60）：block 分配原子性、OOM 回滚、逻辑顺序恢复、前缀树分裂与 terminal、命中拆分、chunk 边界、LRU 驱逐、决策逻辑。
- **GPU 证据**（4090 D 实测）：分页 vs 连续预留分配（×7.9 并发）、chunk 峰值（1/8）、真实模型 prefix cache（-70% prefill token、批量 TTFT ×3.23、峰值 -542 MiB、输出分布不变）。
- **不能推出的结论**：CPU 模拟推不出真实 TTFT/吞吐；合成 GPU 探针（[token, head_dim] 张量）不是真实 vLLM/SGLang kernel；vLLM 未安装，真实 backend 命中率指标留待 serving 环境（69 的 RUN_REAL_BACKEND=False 已如实保留）。
- 复现入口：[../evidence/task3](../evidence/task3)（notebook、日志、JSON、截图）+ [../code](../code)（解答版与三个自加实验脚本）。

---

*学习来源：[DataWhale 社区](https://github.com/datawhalechina) / [llm-algo-leetcode 教程](https://github.com/datawhalechina/llm-algo-leetcode)（Issue #166）。笔记为原创整理，实验均在 vm-60 (RTX 4090 D) 实测。*
