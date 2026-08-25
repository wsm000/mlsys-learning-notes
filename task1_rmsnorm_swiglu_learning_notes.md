# Task1 学习笔记：归一化与 MLP 入口

## 一、RMSNorm

### 1. 核心思想

LayerNorm 会先减去均值，再根据方差进行归一化；RMSNorm 不减均值，只计算输入的均方根（Root Mean Square），因此计算更简单、同步开销更低。

对于最后一维大小为 `d` 的输入向量 `x`：

```text
RMS(x) = sqrt(mean(x²) + eps)
y = x / RMS(x) * weight
```

其中 `weight` 是可学习的缩放参数，形状为 `[hidden_size]`，初始化为全 1。RMSNorm 通常没有 bias。

### 2. 维度与作用范围

输入形状可以是 `[B, T, D]`：

- `B`：batch size
- `T`：token 数量
- `D`：hidden size

归一化只发生在最后一维 `D` 上，每个 token 独立计算 RMS，不会混合不同 token 的信息，输出形状与输入保持一致。

### 3. 数值稳定性

FP16 的表示范围有限，直接计算 `x.pow(2)` 可能溢出。因此实现时先将输入转换为 `float32`，完成平方、求均值和 `rsqrt`，最后再转换回输入 dtype：

```python
x_fp32 = x.float()
variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
normalized = x_fp32 * torch.rsqrt(variance + eps)
output = (weight.to(x.dtype) * normalized).to(x.dtype)
```

`keepdim=True` 可以保留最后一维，方便与原张量广播相乘；`torch.rsqrt` 直接计算倒数平方根，适合这个公式。

## 二、SwiGLU

### 1. 核心结构

普通 MLP 通常是：

```text
down(activation(up(x)))
```

SwiGLU 使用两条升维分支：一条经过 SiLU 作为门控，另一条保留线性信息，之后逐元素相乘，再通过 down projection 降维：

```text
gate = gate_proj(x)
up = up_proj(x)
output = down_proj(SiLU(gate) * up)
```

SiLU 的公式为 `x * sigmoid(x)`。相比 Sigmoid 门控，SiLU 在负值区域仍保留一定梯度，信息传递更平滑。

### 2. 中间层维度为什么是 `8/3 * hidden_size`

标准 MLP 使用两个矩阵，隐藏层为 `4d` 时参数量约为：

```text
2 * d * 4d = 8d²
```

SwiGLU 有两个升维矩阵和一个降维矩阵，参数量约为：

```text
3 * d * h
```

令两者参数量相近：`3dh = 8d²`，得到：

```text
h = 8d / 3
```

实际实现中先取整，再向上对齐到 `multiple_of`（常见为 256）：

```python
intermediate_size = int(hidden_size * 8 / 3)
aligned_size = ((intermediate_size + multiple_of - 1) // multiple_of) * multiple_of
```

例如 `hidden_size=4096` 时，中间层维度为 `11008`。

### 3. 融合投影

为了避免对同一个输入执行两次独立的矩阵乘法，可以把 gate 和 up 投影合并：

```python
self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

gate_up = self.gate_up_proj(x)
gate, up = torch.chunk(gate_up, 2, dim=-1)
output = self.down_proj(F.silu(gate) * up)
```

这样只需读取一次输入并完成一次融合的升维投影，有助于减少显存带宽压力。

## 三、两节内容的联系

RMSNorm 负责在 Attention 或 MLP 前稳定每个 token 的 hidden 向量尺度；SwiGLU 则负责在 MLP 中进行更有选择性的非线性特征变换。二者都是现代 LLaMA 类 Transformer Block 的重要组成部分：

```text
x -> RMSNorm -> Attention -> residual add
h -> RMSNorm -> SwiGLU MLP -> residual add
```

## 四、实践收获

1. 归一化算子不仅要关注数学公式，还要关注混合精度下的溢出问题。
2. `dim=-1` 和 `keepdim=True` 是处理 Transformer hidden states 时非常常见的维度写法。
3. SwiGLU 的额外门控分支会增加参数量，因此需要通过 `8/3` 规则调整中间层维度。
4. 将共享输入的投影合并，是从“能运行”走向“工程高效”的重要优化。
5. RMSNorm 与 SwiGLU 结合后，分别解决了激活尺度稳定和信息选择性传递问题。
