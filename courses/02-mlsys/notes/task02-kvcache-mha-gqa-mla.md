# KV cache 三机制专题：MHA / GQA / MLA

> 配套笔记：[task02-rope-attention.md](task02-rope-attention.md)（RoPE 数学与 MHA/GQA 基础）·
> 配套代码：`molecular_gpt_KVCacheBuf.py`（预分配 KV buffer 版，本篇所有改动锚点均指向该文件行号）·
> 实测背景：Day5 推理加速报告（vm-60）。

## 0. 先统一账本：KV cache 存什么

预分配 buffer 版每层有两个 buffer：

```python
k_buf = torch.zeros(B, n_heads, max_len, head_dim)   # (B, nh, max_len, hs)
v_buf = torch.zeros_like(k_buf)
```

每生成 1 个 token，每层写入 1 个位置，decode 全程驻留。所以：

```text
KV cache 大小 = 层数 × 2（K和V） × 每位置元素数
```

三种机制的差异全部集中在「**每位置存多少、存的是原始 K/V 还是别的表示**」。

## 1. MHA：每位置存全量 K/V

Query 头 = KV 头 = `n_heads`。每位置每层存 **2 × H × D** 个元素（K 一份 + V 一份，各 H×D）。

以 demo 配置（`d_model=128, n_heads=4, head_dim=32`）：

| 配置 | 每位置每层元素 | max_len=512, 2层, B=1 的 buffer |
|:---|---:|---:|
| MHA（已实现） | 2×4×32 = 256 | 256×512×2层 = 256K 元素（fp32 约 1 MiB） |

特点：cache 最大，但每个 KV 头独立、表达力最强。这正是 Day5 实验里「批量生成时显存随 batch × 上下文线性增长」的那一项。

## 2. GQA：buffer 里少存几个头，读取时临时扩展

只存 `H_kv` 份 K/V，attention 前临时 `repeat_kv` 扩到 H 份。落到 buffer 版代码，改动只有三处：

```python
# __init__ (对照原 :58):
self.n_kv_heads = n_kv_heads          # 例如 1
self.n_rep = n_heads // n_kv_heads
self.qkv = nn.Linear(d_model, d_model + 2 * n_kv_heads * self.head_dim)
#                      Q 全量       K/V 只投影 H_kv 头

# _ensure_buf (对照原 :67): buffer 也只有 H_kv 头
k_buf = torch.zeros(B, self.n_kv_heads, self.max_len, self.head_dim)

# forward (对照原 :83-86): 写入逻辑完全不变（本来就只写 H_kv 份），
# 读出来之后、进入 QK^T 之前扩展:
k_full = repeat_kv(k_buf[:, :, :total], self.n_rep)
v_full = repeat_kv(v_buf[:, :, :total], self.n_rep)
```

关键点（顺序）在 buffer 版同样成立：**cache 里存的必须是 repeat 之前的原始 H_kv 份**。若误把扩展后的副本写进 buffer，cache 退化回 MHA 大小，GQA 白做。

正确顺序：

```text
计算原始 K/V -> 与历史原始 Cache 拼接 -> 保存新 Cache -> 临时 repeat_kv -> Attention
```

账本对比（同上配置）：

| 结构 | 每位置每层元素 | 相对 MHA |
|:---|---:|---:|
| MHA（H_kv=4） | 256 | 1× |
| GQA（H_kv=2） | 128 | 2× |
| GQA（H_kv=1，即 MQA） | 64 | 4× |

真实模型里收益更大：Llama-3-70B 用 GQA（H=64, H_kv=8），KV cache 直接砍到 MHA 的 1/8——长上下文 + 多并发下最重要的省显存旋钮之一。

## 3. MLA（DeepSeek-V2/V3）：存的根本不是 K/V，而是低秩压缩后的隐向量

三种里差异最大的。MLA 不再问「存几份 K/V」，而是问「**能不能把 K/V 压缩后再存**」。它把每层每 token 的 KV 信息压缩成一个隐向量 $c_{KV} \in \mathbb{R}^{d_c}$（如 512 维），cache 只存 $c_{KV}$ 加一个解耦 RoPE 键 $k_R$：

```text
cache 每位置每层 = c_KV(d_c) + k_R(d_R)    vs    MHA 的 2 × H × D
```

用 DeepSeek-V2 的真实配置算账（H=128, D=128, fp16）：

| 机制 | 每位置每层字节 | 相对 MHA |
|:---|---:|---:|
| MHA | 2×128×128×2B = **64 KiB** | 1× |
| MLA（d_c=512, d_R=64） | (512+64)×2B = **1.125 KiB** | **≈ 1/57** |

两个核心设计：

### ① 权重吸收

K 和 V 每步本来要用上投影恢复：$K_i = W_{UK} c_{KV}$、$V_i = W_{UV} c_{KV}$。但 attention score 是点积：

```text
q^T k = (W_UQ c_Q)^T (W_UK c_KV) = c_Q^T (W_UQ^T W_UK) c_KV
```

投影矩阵可以互相吸收。**输出侧同理**：`probs · V` 可以先算 `probs · c_KV`（对整个历史做一次），最后过一个 $W_{UV}$。所以**全程不必为历史 token 物化 K/V**，cache 里存压缩态就够了。

### ② 解耦 RoPE

RoPE 必须作用在「点积所在的空间」里；如果先旋转 $k$ 再投影 $W_{UK}$，线性投影会破坏旋转的相对位置性质。所以 RoPE 键走独立通路：

```text
k_R = RoPE(W_KR h_t)     # 投影后不再变换，直接存进 cache
score 拆成两项: q_c^T k_c + q_R^T k_R
```

对应到 buffer 版，`k_buf/v_buf` 两个 buffer 合并成一个低维 buffer：

```python
# 每层预分配（对照原 :67）: 不再是 (B, H, max_len, D)，而是
kv_buf = torch.zeros(B, max_len, d_c + d_R)      # 每位置只存 576 维

# 每步写入（对照原 :79-80）:
c_kv = W_DKV(h_t)                                 # 压缩投影
k_R  = rope(W_KR(h_t))                            # 解耦 RoPE 键
kv_buf[:, kv_offset:kv_offset+T] = torch.cat([c_kv, k_R], -1)

# attention（对照原 :86-93）:
k_c = W_UK(c_kv)                                  # 或吸收进 Q 侧，逐头恢复
att = q_c @ k_c^T + q_R @ k_R^T                   # 两项 score
out = (att @ c_kv_hist) @ W_UV                    # V 吸收：历史 V 不物化
```

顺带澄清一个常见误解：DeepSeek 把 Q 也低秩压缩（$c_Q$），但 **Q 侧的压缩省的不是 cache**——Q 每步重算、从不驻留，省的是激活和计算；真正落到 KV cache 上的只有 $c_{KV} + k_R$。

## 4. 三机制一张表总结

| | MHA（已实现） | GQA | MLA |
|:---|:---|:---|:---|
| cache 内容 | 原始 K/V，H 份 | 原始 K/V，**H_kv 份**（H_kv<H） | **压缩隐向量** $c_{KV}$ + 解耦 $k_R$ |
| 每位置每层元素 | 2×H×D | 2×H_kv×D | d_c + d_R（与 H 无关！） |
| buffer 形状 | (B, H, max_len, D) | (B, H_kv, max_len, D) | (B, max_len, d_c+d_R) |
| attention 时恢复 | 不需要 | repeat_kv 临时扩到 H | W_UK/W_UV 恢复（或吸收） |
| RoPE | 旋转后直接进 cache | 同 MHA | K 侧必须**解耦**：旋转后的 k_R 单独存 |
| 质量 | 基线 | 轻微折损 | ≥ MHA（DeepSeek 论文口径） |
| 代表 | GPT-2/3、Day5 的模型 | Llama-2/3、Mistral、Qwen | DeepSeek-V2/V3 |

## 5. 关键差异的深层理解

1. **GQA 是"存得少、算得对"**：恢复靠复制，数学上等价于少几个头——多个 Query head 共用一组 KV head 本身就是模型结构的一部分（训练时就如此），不是推理期的近似。
2. **MLA 是"存的是信息瓶颈"**：恢复靠学习到的投影，H 个头的信息被压进一个 d_c 维瓶颈（d_c=512 比单头 D=128 大、比 H×D=16384 小）。压缩是有损的吗？——论文口径下质量不掉，因为瓶颈是**训练出来的**：模型从头就学会把 KV 信息集中进低秩子空间，推理时 cache 存的就是这个子空间的坐标。
3. **MLA 的 cache 大小与头数解耦**：这让「加大头数增强表达」不再付出 cache 代价——GQA 做不到这点。
4. **RoPE 与投影的顺序约束**是三机制里最容易写错的地方：MHA/GQA 旋转后的 K 直接进 cache；MLA 中 $W_{UK}$ 与 RoPE 不交换（解耦通路），$c_{KV}$ 与 RoPE 不相交。

## 6. 延伸思考（可做实验）

- 照 `molecular_gpt_KVCacheBuf.py` 写 GQA buffer 版（三处改动），对照同 batch 下 buffer 显存与吞吐 vs MHA。
- 写 MLA 最小玩具版，数值验证权重吸收的等价性：`(q@W_UK^T)@k` vs `q@(W_UK@k)`，以及 `probs@V` vs `(probs@c_kv)@W_UV`。

---
*整理自 2026-09 学习对话；机制差异锚点对应 `molecular_gpt_KVCacheBuf.py`（Day5 加分挑战：预分配 KV buffer 对照实验）。*
