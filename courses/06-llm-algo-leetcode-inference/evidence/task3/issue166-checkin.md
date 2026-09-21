<!-- 使用说明：
1. 若 mlsys-learning-notes 仓库已推送（origin/main 含本次提交），文中 raw.githubusercontent.com 截图链接可直接渲染；
2. 若尚未推送，请在 GitHub issue 页面直接拖拽上传 evidence/task3/ 下的 6 张 PNG（task3_22/24/34/66/69_*_runshot.png + task3_exp_prefix_reuse_runshot.png），
   并将正文中的 ![..](raw链接) 替换为上传后生成的图片；
3. 微信群昵称一处需填写后提交。
-->
### llm-algo-leetcode 推理优化 | 202609 Task3 · KV Cache 状态与生命周期

微信群昵称：（填写）
GitHub ID：wsm000
打卡选择：**(3) 4.1 + 4.2 + 4.3**
运行环境：vm-60 · NVIDIA RTX 4090 D（Ada sm89，24 GB）· torch 2.9.1+cu128 · transformers 5.16.1 · Python 3.10.12

学习笔记：https://github.com/wsm000/mlsys-learning-notes/blob/main/courses/06-llm-algo-leetcode-inference/notes/task03-kv-cache-state-and-lifecycle.md
补全并跑通的 notebook：[Part 02 · 22](https://github.com/wsm000/mlsys-learning-notes/blob/main/courses/06-llm-algo-leetcode-inference/code/22_solved.ipynb) · [Part 02 · 24](https://github.com/wsm000/mlsys-learning-notes/blob/main/courses/06-llm-algo-leetcode-inference/code/24_solved.ipynb) · [Part 02 · 34](https://github.com/wsm000/mlsys-learning-notes/blob/main/courses/06-llm-algo-leetcode-inference/code/34_solved.ipynb) · [Part 02 · 66](https://github.com/wsm000/mlsys-learning-notes/blob/main/courses/06-llm-algo-leetcode-inference/code/66_solved.ipynb) · [Part 02 · 69](https://github.com/wsm000/mlsys-learning-notes/blob/main/courses/06-llm-algo-leetcode-inference/code/69_solved.ipynb)
额外实验：分页碎片账本（GPU 实测）× 前缀复用与 LRU 容量扫描 × 真实模型 Prefix Cache 对照（脚本与 JSON 见 [evidence/task3](https://github.com/wsm000/mlsys-learning-notes/tree/main/courses/06-llm-algo-leetcode-inference/evidence/task3)）

---

#### 4.1 最小打卡（Part 02 · 22、Part 02 · 24）

**（1）PagedAttention 为什么拆 block、逻辑/物理/块表关系、为什么逻辑连续不要求物理连续**

连续分配的问题是"放不齐"：预留尾部、外部碎片、请求间不可共享。拆成固定大小 block 后按需分配，把"最多要多少"换成"现在要多少"。vm-60 实测（LLaMA-7B 账本 0.5 MiB/token，8 请求共 4080 token）：分页 4144 token / 2072 MiB vs 连续预留 32768 token / 16384 MiB——省 87.4%，同等预算并发 ×7.9。三层视角：逻辑层是一维连续序列（attention 只认顺序）；物理层是 [num_blocks, block_size, head_dim] 池；block_table[i]=p 是逻辑第 i 块到物理块 p 的页表。逻辑连续不要求物理连续，因为计算只依赖 token 顺序，kernel 按块表寻址即得正确结果；换来按需分配、共享池、前缀按块共享，代价是块表管理与非连续访存。

**（2）RadixAttention 的前缀树组织与最长前缀匹配的作用**

已算前缀组织成压缩基数树：每条边存一段 token，terminal 标记完整路径；插入时部分重叠就分裂成 shared/old_suffix/new_suffix，公共前缀只留一条共享边。新请求 match_prefix 从开头沿完整共享边向下得 hit_len，split_prompt 切出复用前缀与重算后缀——把"重复 prefill"变成"一次计算、多次检索"，多轮对话的历史前缀直接复用。实测 146 请求（共享 256 token system prompt）：多轮请求命中最长 373 token；注意题目口径只有 terminal 节点计入命中，真实 SGLang RadixCache 每节点携 KV、可 LRU 驱逐。

**（3）RadixAttention 与 PagedAttention 的区别**

PagedAttention 是**内存分配管理**问题（怎么放：块表 + 按需分配，收益是容量/并发 ×7.9）；RadixAttention 是**前缀复用**问题（怎么找回来：前缀树 + 最长匹配，收益是计算/TTFT）。两者正交互补，真实系统都是"分页 block 管物理 + 前缀索引管复用"。

![Part 02 · 22 PagedAttention 运行截图](https://raw.githubusercontent.com/wsm000/mlsys-learning-notes/main/courses/06-llm-algo-leetcode-inference/evidence/task3/task3_22_paged_attention_runshot.png)
![Part 02 · 24 RadixAttention 运行截图](https://raw.githubusercontent.com/wsm000/mlsys-learning-notes/main/courses/06-llm-algo-leetcode-inference/evidence/task3/task3_24_radix_attention_runshot.png)

---

#### 4.2 增项 1（04 生命周期、Part 02 · 34）

**（1）KV Cache 的状态与触发事件**

五状态：未建立 → prefill 建立（分配初始块/前缀命中）→ decode 追加（跨块边界 +1 块）→ 可复用驻留（被命中 refcount+1）→ 释放/驱逐（LRU 淘汰或请求结束归还）。四类事件：创建（prefill 分配、跨块扩容）、命中（前缀匹配、跳 prefill）、更新（写 K/V、刷新 LRU）、驱逐/释放。容量治理是显式取舍：保留换命中、驱逐换容量（下次重算）、重算换驻留、量化换字节；正确性前提是模型 revision/tokenizer/位置/租户一致，否则必须失效。

**（2）Prefix Cache 跳过什么、Chunked Prefill 为什么拆**

命中 H 长度前缀 = 这 H 个 token 在**所有层**的 Q/K/V 投影、attention、MLP、logits 全跳过（历史 token 不影响下一个 token 的分布，影响已在复用的 K/V 里）。真实模型实测（gpt2，8 请求共享 800 token prompt）：prefill token 8000→2400（-70%），logit 最大差 0.188 与 top1-top2 边距 0.094 同量级（fp 近 tie 噪声，非逻辑错误）。Chunked Prefill 把长 suffix 按 block_size 拆段：单次工作集峰值下降（实测 4096 token 一次性 16 MiB vs 512×8 段 2 MiB）、调度粒度变细，使 compute-bound 的 prefill 与 memory-bound 的 decode 可混排进同一 batch；TTFT 不变（总算量相同），改善的是 TPOT 抖动与吞吐。

![Part 02 · 34 Prefix Caching 运行截图](https://raw.githubusercontent.com/wsm000/mlsys-learning-notes/main/courses/06-llm-algo-leetcode-inference/evidence/task3/task3_34_prefix_chunked_prefill_runshot.png)

---

#### 4.3 增项 2（Part 02 · 66、Part 02 · 69）

**（1）五个指标反映什么**

TTFT=排队+prefill（交互体感第一关）；TPOT=decode 稳态每 token 时间；吞吐=系统级 token/s；峰值显存=容量边界（并发与 OOM）；P99=长尾延迟（SLO  Usually 看它）。它们相互冲突（batch 升 → 吞吐升但 TTFT/显存升），所以 66 用固定 workload 的单变量 G0/G1 对照 + bottleneck 分类（prefill/decode/memory-bound）代替单指标最优。

**（2）有效命中的三个一致性**

token 内容一致（同 tokenizer 同序列）、位置一致（RoPE 把位置编进 K，position offset 必须接续）、模型配置一致（revision/dtype/量化/LoRA/租户）。三种伪命中：中间子串相同、位置错位的相同 token 块、换 tokenizer 的相同文本——69 的 key=(block_index, block) 就是把位置编进 key 防第二种。

**（3）四个数一起算 + 为什么命中率≠值得上线**

命中率（请求口径）、复用 token 率（token 口径，与 FLOPs 成正比）、TTFT 改善（candidate-baseline，负才好）、维护成本（overhead + eviction_count + 常驻显存）必须同时记录。实测：LRU 容量 128 token 命中率 0%、512 token 99.3% 但驱逐 209 次、4096 token 封顶且零驱逐——容量必须大于工作集；真实模型串行场景 prefill 省 70% 但 TTFT 只快 1.05~1.07×（固定开销摊薄），批量 8 请求才 ×3.23。命中率是"潜力"不是"到手收益"：短前缀、显存挤压、驱逐回吐、固定开销、错误复用代价都会让高命中率不值上线；69 的决策因此要求命中率增益、TTFT 收益、维护开销三门槛同时满足，与 66 的 G0/G1 baseline 同口径对照，命中证据必须来自 backend metrics/日志而非 TTFT 反推。

![Part 02 · 66 推理性能比较 运行截图](https://raw.githubusercontent.com/wsm000/mlsys-learning-notes/main/courses/06-llm-algo-leetcode-inference/evidence/task3/task3_66_inference_metrics_runshot.png)
![Part 02 · 69 Prefix Cache Benchmark 运行截图](https://raw.githubusercontent.com/wsm000/mlsys-learning-notes/main/courses/06-llm-algo-leetcode-inference/evidence/task3/task3_69_prefix_cache_benchmark_runshot.png)
![自加实验：前缀复用/容量治理/真实模型 Prefix Cache](https://raw.githubusercontent.com/wsm000/mlsys-learning-notes/main/courses/06-llm-algo-leetcode-inference/evidence/task3/task3_exp_prefix_reuse_runshot.png)

---

学习来源：[DataWhale 社区](https://github.com/datawhalechina) / [llm-algo-leetcode 教程](https://github.com/datawhalechina/llm-algo-leetcode)（Issue #166）。笔记与实验为原创整理，均在 vm-60（RTX 4090 D）实测。
