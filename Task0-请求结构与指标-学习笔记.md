# Task 0/4 学习笔记：请求结构与推理优化

> 日期：2026-09 · 课程：llm-algo-leetcode 推理优化 · Issue [#146](https://github.com/datawhalechina/llm-algo-leetcode/issues/146)
> 实测环境：CPU 前向 + KV Cache 对比实验（GQA 配置：32 query 头 → 8 KV 头）

---

## 4.1 Attention 与 Transformer 的关系；MHA / GQA / MLA

### Attention 与 Transformer 的关系

Transformer = Attention 堆叠出来的架构（自注意力 + FFN，重复 N 层）。Attention 是其中唯一**需要跨 token 看历史**的部件——FFN、LayerNorm、残差都是逐 token 独立计算的。

正因为历史必须被保存和检索，推理时产生了 KV Cache；KV Cache 太大，才有了 MHA → GQA → MQA → MLA 的演进。**推理优化的主战场就是 Attention。**

### 三种 Attention 方案

| 方案 | 结构 | KV Cache | 质量代价 |
|---|---|---|---|
| **MHA** | 每个 query 头配专属 K/V 头 | 最大（∝ 头数） | 无，检索能力最强 |
| **GQA** | 一组 query 头共享一套 K/V（本实验 32→8，即 4:1） | 缩到 1/4 | 同组头共用检索钥匙，略降 |
| **MLA** | K/V 压缩进低秩 latent 向量，只存压缩态 | 压缩率比 GQA 更高 | 靠低秩投影补表达力 |

- MHA 与 MQA 是 GQA 的两个极端（每组 1 个头 = MHA；所有头共享 1 套 = MQA）
- 理解差异的比喻：GQA 是"减少钥匙数量"，MLA 是"把钥匙库压缩打包"
- **演进主线：检索质量 ↔ KV Cache 体积的交换**。MHA 最贵最好，MQA 最省最损，GQA 是甜点，MLA 换压缩思路图两者兼得

---

## 4.2 一个请求的生命周期：Prefill / Decode；TTFT / TPOT

### 全过程

```
prompt (L 个 token) ──► [Prefill] ──► 第1个输出token ──► [Decode × N] ──► EOS
                          一次整段前向              逐步生成
```

- **Prefill**：整段 prompt 一次过前向，所有 token 并行算（算力密集），同时把每层 K/V 写入 KV Cache
- **Decode**：每步只输入 1 个新 token——从 Cache **读**历史 K/V，现场算自己的 K/V 并**存入**，输出下一个 token

### 两个指标

| 指标 | 定义 | 主导因素 |
|---|---|---|
| **TTFT**（Time To First Token） | 发出请求 → 第一个 token 出现 | Prefill 时间 ∝ prompt 长度（计算） |
| **TPOT**（Time Per Output Token） | 每个后续 token 的平均间隔 | 每步读 Cache 的体积（显存带宽）+ 前向 1 个 token |

### 疑难点（本节最大的困惑，已澄清）

**困惑**：Decoder 生成下一个 token，难道不需要前面完整信息吗？为什么 TPOT 看起来不随长度变？

**澄清**：Decode 每步**当然需要完整历史**，但历史的形态变了——不是不需要历史，而是**不需要重算历史**：

| | 不用 Cache（每步重算全部） | 用 Cache（每步） |
|---|---|---|
| 历史从哪来 | 前面 L 个 token 重新过一遍前向 | 从显存**读** L 份现成 K/V |
| 当前 token | 现场算（它的 K/V 不在 Cache 里） | 现场算，算完存入 Cache |
| 成本类型 | **计算**主导（贵） | **搬运**主导（便宜） |

- 因果掩码只挡"未来"不挡"自己"：第 5 个 token 做 attention 时能看到全部 5 个位置，含它自己
- **历史靠存，当下靠算**：Cache 里存的是过去的 K/V，当前 token 永远现场算
- Cache 一致性测试失败 → 只能是 Cache 存的历史不一致（存错/拼反/漏），不可能是"现场算错了"（两条路用同一套权重，Prefill 会同样错）
- **结论**：KV Cache 把"每步重算整段"（昂贵的计算）降级为"每步读一遍历史"（便宜的搬运）。TPOT 随长度**线性但缓慢**增长——短中长度被淹没在噪声里，128K 长上下文才明显爬升（长文推理慢的主因）

---

## 4.3 Roofline 模型看 Prefill 与 Decode

### Roofline 模型

性能上限 = min(算力， 带宽 × 算术强度)，其中算术强度 = FLOPs / Bytes（每读 1 字节能做多少次浮点运算）：

- **计算受限**：算术强度高 → GPU 算力跑满 → 时间 ∝ 计算量
- **带宽受限**：算术强度低 → 内存喂不饱算力 → 时间 ∝ 搬运量

### 两个阶段恰好落在 Roofline 两端

| | 算术强度 | 瓶颈 | 时间公式 | 优化方向 |
|---|---|---|---|---|
| **Prefill** | 高（L 个 token 并行，一次权重读取被 L 个 token 摊薄） | 计算受限 | ∝ FLOPs ∝ prompt 长度 | 拼算力：更强 GPU、并行、算子融合 |
| **Decode** | 低（每步只算 1 个 token，却要搬全部权重 + 全部 KV） | 带宽受限 | ∝ 每步搬运字节数 | 减搬运：GQA/MLA、量化、PagedAttention |

### 用 Roofline 解释三个现象

1. **GQA 为什么让 Decode 提速**：Decode 是带宽受限，时间 ∝ 搬运字节。KV Cache 缩到 1/4 → 搬运缩到 1/4 → TPOT 近似缩到 1/4。Prefill 几乎不受影响（本就不是带宽受限）
2. **为什么大 batch 对 Decode 收益大**：多请求共享一次权重搬运（权重只读一遍服务所有请求），算术强度被拉高，Decode 向计算区靠拢；Prefill 算术强度本来就高，batch 提升有限
3. **两阶段优化方向相反**：Prefill 要"算得快"，Decode 要"搬得少"。vLLM 这类系统要同时优化两端并处理资源争抢（chunked prefill 把 Prefill 分片插进 Decode 间隙，平衡 TTFT 与 TPOT）

**一句话总结**：Prefill 在计算象限拼算力，Decode 在带宽象限拼搬运。所有推理加速技术，本质都是回答"卡在哪条 roof 上，怎么把那条 roof 抬上去"。

---

## 实测踩坑记录

| 坑 | 现象 | 解法 |
|---|---|---|
| `enable_cache` 参数名传错 | 不报错但行为不对 | 查 API，参数名写对才生效 |
| 改 config 字典被覆盖 | 配置不生效 | 直接改 config 对象本身 |
| hidden 维度报错 | shape mismatch | 先打印 shape 定位是哪一层（hidden 2048 vs KV 投影维度 512） |

## 主线速查

| 主线 | 核心结论 |
|---|---|
| KV Cache | 历史靠存，当下靠算；只省计算，不改 logits（一致性） |
| GQA | 4:1 共享 → Cache 1/4 → Decode 提速；代价是共用检索钥匙 |
| TTFT/TPOT | 长 prompt 代价集中到 TTFT 一次付清；TPOT 线性缓慢增长，近似恒定 |
