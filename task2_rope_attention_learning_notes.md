# Task2 学习笔记：RoPE 与 Attention（MHA/GQA）

## 一、RoPE：把相对位置信息写入 Q 和 K

### 1. 为什么 Attention 需要位置编码

自注意力本身只根据 Query 和 Key 的相似度计算分数，无法分辨 token 的先后顺序。若没有位置编码，打乱 token 顺序不会改变注意力机制对内容的基本处理方式。

RoPE（Rotary Position Embedding，旋转位置编码）不再把一个位置向量直接加到 embedding 上，而是对 Query 和 Key 的每一对特征维度施加与位置有关的二维旋转。这样，Q 和 K 的点积会自然包含它们的相对位置差。

```text
hidden states
  |- q_proj -> Q -> RoPE rotate -|
  |- k_proj -> K -> RoPE rotate -|-> QK^T / sqrt(d) -> attention scores
  `- v_proj -> V -----------------|
```

关键点：RoPE 只作用于 Q 和 K，不旋转 V；它改变的是注意力打分时的方向关系，而不是 Value 中保存的内容。

### 2. 数学直觉

将 head dimension 中的相邻两个实数看作一个复数：

```text
x = x_real + i * x_imag
```

在位置 `m`、第 `j` 个维度对上，将它乘以复数旋转因子：

```text
exp(i * m * theta_j)
theta_j = base^(-2j / d)
```

其中 `d` 是 `head_dim`，`base` 通常取 `10000`。不同维度对具有不同旋转频率：低维变化较快，适合表达局部位置关系；高维变化较慢，适合表达更长距离的位置关系。

因为旋转满足复数乘法规律，位置为 `m` 的 Query 和位置为 `n` 的 Key 在点积中留下的是 `m - n` 的关系。这是 RoPE 能表达相对位置的根本原因。

### 3. PyTorch 实现路径

RoPE 的实现可分为三步：

```python
inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
positions = torch.arange(seq_len, dtype=torch.float32)
angles = torch.outer(positions, inv_freq)
freqs_cis = torch.polar(torch.ones_like(angles), angles)
```

上面的 `freqs_cis` 形状为 `[seq_len, head_dim // 2]`。应用时，先把 Q/K 的最后一维变成实部和虚部两两配对，再解释成复数：

```python
xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

freqs_cis = freqs_cis.view(1, seq_len, 1, head_dim // 2)
xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2).type_as(xq)
xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2).type_as(xk)
```

这里有三个实现细节：

1. `head_dim` 必须是偶数，否则无法组成实部/虚部对。
2. 频率张量要 reshape 为 `[1, seq_len, 1, head_dim // 2]`，从而在 batch 和 head 维度上自动广播。
3. 旋转计算先提升到 FP32，降低 FP16/BF16 下复数计算产生数值误差或异常的风险，最后再恢复输入 dtype。

### 4. RoPE 的不变量

二维旋转不会改变向量模长。因此完成 RoPE 后，Q/K 的形状和每个向量的 L2 norm 应保持不变；改变的是它们之间用于 attention score 的夹角。这个性质也是验证实现是否正确的一个直接方法。

## 二、MHA：多头注意力的计算链路

### 1. 从单头到多头

给定输入 `x`，先通过三个线性层得到 Query、Key 和 Value：

```text
Q = x Wq
K = x Wk
V = x Wv
```

多头注意力会把 hidden dimension 切成多个 head。若输入形状为 `[B, S, hidden_dim]`，且 `hidden_dim = num_heads * head_dim`，则典型变形是：

```python
q = q.reshape(B, S, num_heads, head_dim).transpose(1, 2)
```

得到 `[B, H, S, D]`，其中：

- `B`：batch size
- `H`：attention head 数
- `S`：sequence length
- `D`：每个 head 的维度

不同 head 可以学习不同类型的关联模式。各 head 的输出在最后拼接，并通过输出投影 `o_proj` 回到 `hidden_dim`。

### 2. Scaled Dot-Product Attention

每个 head 内的核心公式是：

```text
scores = Q K^T / sqrt(head_dim)
probs = softmax(scores + mask)
output = probs V
```

在代码中：

```python
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
scores = scores + attention_mask
probs = F.softmax(scores, dim=-1)
output = torch.matmul(probs, v)
```

除以 `sqrt(head_dim)` 的原因是：当特征维度增大时，QK 点积的数值范围也会变大。若不缩放，softmax 容易过于尖锐，进而使梯度不稳定。

因果语言模型还需要 causal mask。它会把未来 token 对应的位置加上极小值（通常是 `-inf`），保证当前位置只能关注自身及历史 token。

### 3. 合并多头输出

attention 输出的形状为 `[B, H, S, D]`。需要先转为 `[B, S, H, D]`，再将 head 维度与每头维度拼回：

```python
output = output.transpose(1, 2).reshape(B, S, hidden_dim)
output = self.o_proj(output)
```

使用 `reshape` 通常比 `view` 更稳妥，因为转置后的张量不一定连续；`reshape` 会在必要时处理连续性。

## 三、GQA：用更少的 KV 头降低推理成本

### 1. MHA、MQA 与 GQA 的区别

| 结构 | Query 头 | Key/Value 头 | 特点 |
| --- | ---: | ---: | --- |
| MHA | `H` | `H` | 表达能力强，但 KV Cache 最大 |
| MQA | `H` | `1` | Cache 最小，但所有 Query 共用一组 KV |
| GQA | `H` | `H_kv`，且 `1 < H_kv < H` | 在质量与缓存成本之间折中 |

GQA（Grouped-Query Attention）让多个 Query head 共用一个 KV head。若 `num_heads=32`、`num_kv_heads=8`，则每个 KV head 对应 4 个 Query head：

```python
num_queries_per_kv = num_heads // num_kv_heads
```

前向计算时，需要将 KV 头临时扩展到与 Query 头数量一致：

```python
def repeat_kv(x, repeats):
    return x.repeat_interleave(repeats, dim=1)
```

GQA 的核心收益在于 KV Cache 仅储存原始 `num_kv_heads` 份 K/V，而不储存扩展后的副本。因此它显著降低长上下文和多并发推理的显存压力。

## 四、KV Cache：自回归推理中的时间换空间

### 1. 为什么需要缓存

生成第 `t` 个 token 时，新 token 的 Query 需要与前 `t` 个 token 的 Key/Value 交互。若每一步都重新计算历史 token 的 K/V，推理会重复做大量工作。

KV Cache 会把过去 token 已算出的 K/V 保存下来。当前步只计算新 token 的 K/V，然后沿 sequence 维拼接：

```python
if kv_cache is not None:
    k_cache, v_cache = kv_cache
    k = torch.cat([k_cache, k], dim=2)
    v = torch.cat([v_cache, v], dim=2)
new_kv_cache = (k, v)
```

在 `[B, H, S, D]` 布局中，`dim=2` 是 sequence length 维度。

### 2. 与 GQA 的正确组合顺序

缓存必须在 `repeat_kv` 之前进行：

```text
计算原始 K/V -> 与历史原始 Cache 拼接 -> 保存新 Cache -> 临时 repeat_kv -> Attention
```

若把扩展后的 KV 写入缓存，显存占用会退化为 MHA 水平，失去 GQA 的主要价值。

## 五、Task2 关键收获

1. RoPE 通过旋转 Q/K 的二维特征对表达相对位置，既保持向量模长，也不改变 V。
2. 多头注意力的本质不仅是矩阵乘法，还包括投影、切头、缩放、mask、softmax、拼头与输出投影等严格的形状链路。
3. `sqrt(head_dim)` 缩放和 causal mask 分别保证数值稳定与自回归信息约束。
4. GQA 通过减少 KV 头数降低 KV Cache 成本，并以临时扩展 KV 头保持与多个 Query head 的兼容。
5. KV Cache 消除了历史 K/V 的重复计算，是 LLM 自回归生成加速的基础；其代价是显存随上下文长度增长。
