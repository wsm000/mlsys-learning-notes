# 学习笔记：模型结构主线 05–08（LLaMA Block → MoE → 架构技巧）

> **日期**：2026-08-25
> **材料**：《02_PyTorch_Algorithms》05–08 四个 notebook（CPU-first）
> **学习方式**：苏格拉底式——先自答热身题，再填空写代码，最后对参考解析
> **状态**：四节全部通关 ✅

---

## 0. 总览地图

| 节 | 主题 | 难度 | 核心 TODO | 一句话带走 |
|---|---|---|---|---|
| 05 | LLaMA3 Block 组装 | Medium | SwiGLU MLP、Pre-Norm 残差 | 窄主干 + 宽支路："存旧的、norm 新的、加回去" |
| 06 | MoE Router | Medium | 全局 softmax → top-k → 重归一化 | 顺序决定梯度流向 |
| 07 | 负载均衡损失 | Hard | P̄ᵢ、fᵢ、aux loss | 一硬一软相乘，均匀时 loss = α |
| 08 | 架构技巧 | Easy | Gemma `(1+w)`、Qwen 绑定 | Trick 不是花活，是工程取舍 |

**主线暗线**：05 把零件拼成稠密 Block → 06/07 把稠密 MLP 换成"专家 + 路由 + 约束" → 08 展示真实模型在同一骨架上做微创新。学完可读懂 LLaMA / Mixtral / Qwen / Gemma 的 config 与建模骨架。

---

## 1. 第 05 节：LLaMA3 Block 组装

### 1.1 数据流（背下来）

```text
x [B, T, D]
  ├─► RMSNorm ─► Attention(RoPE+GQA) ─┐
  │                                   ⊕ ← 残差
  └───────────────────────────────────┘
  │ h
  ├─► RMSNorm ─► SwiGLU MLP ──────────┐
  │                                   ⊕ ← 残差
  └───────────────────────────────────┘
  ▼ output [B, T, D]
```

判据：主干张量永远是 `[B, T, D]`；归一化和变换都发生在"支路"上。

### 1.2 LLaMA 相对 GPT-2 的五大改动

1. **Pre-Norm**（归一化放子层之前）→ 深层训练更稳
2. **RMSNorm** 替代 LayerNorm（不减均值、无偏置）→ 快 10–15%
3. **SwiGLU** 替代 ReLU/GELU → 门控提升表达力
4. **RoPE** 彻底替代绝对位置编码 → 支持长度外推
5. **GQA**（LLaMA-2 起）→ KV Cache 显存降数倍，性能接近 MHA

### 1.3 参考代码

```python
class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class LlamaDecoderLayer(nn.Module):
    def forward(self, hidden_states):
        # --- Attention Block ---
        residual = hidden_states                        # 存旧的（norm 之前！）
        hidden_states = self.input_layernorm(hidden_states)   # norm 新的
        hidden_states = self.self_attn(hidden_states)         # 变换
        hidden_states = residual + hidden_states              # 加回去
        # --- MLP Block（完全对称）---
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```

维度账（hidden=512, intermediate=1376）：

| 投影层 | 输入 | 输出 | 权重形状 |
|---|---|---|---|
| gate_proj | 512 | 1376 | [1376, 512] |
| up_proj | 512 | 1376 | [1376, 512] |
| down_proj | 1376 | 512 | [512, 1376] |

### 1.4 为什么 intermediate ≈ 8/3 × hidden？

- 传统 FFN：2 个矩阵 × d×4d = **8d²**
- SwiGLU：3 个矩阵 × d×(8/3)d = **8d²**

8/3 是"加了第三个矩阵还想保持参数预算不变"解出来的数。理论值向上取整成硬件友好的倍数。

### 1.5 ⭐ 重点问答：gate_proj 和 up_proj 都是从 hidden 到 intermediate，为什么要两个独立投影？

**1. 功能差异**
- **`gate_proj`**：输出经过 **SiLU** 激活，产生**门控系数**（gating signal），像"软开关"决定哪些信息向后传递。SiLU 值域约 `[-0.278, +∞)`：既能抑制（≈0 或负）也能增强（正）。
- **`up_proj`**：不经过激活，直接作为**候选特征**（candidate features），是被门控筛选的原始内容。
- 结合方式：`h = F.silu(gate_proj(x)) * up_proj(x)`，逐元素调制——门控值大的位置保留信息，小的位置抑制。

**2. 参数独立性**
两者是独立线性层，训练中学到不同功能：
- `gate_proj` 学**"哪些信息重要"**（选择/过滤）
- `up_proj` 学**"被选择的信息是什么"**（内容/表示）

若共享同一投影，门控与候选内容强耦合，表达能力大幅下降。

**3. 数学形式**

$$\text{SwiGLU}(x) = \big(\text{SiLU}(xW_g) \otimes xW_u\big)\, W_d$$

$W_g/W_u/W_d$ 分别对应三个投影权重；输入相同、权重不同，分别服务"门控"与"内容"。

**总结**：gate → 门控信号（过 SiLU）；up → 候选特征（不过激活）；独立学习，共同实现门控线性单元效果。

### 1.6 为什么先升维再降维？

一句话：**升维给非线性加工腾工位，降维把结论打包回统一宽度以便堆叠。**

1. 中间维度 = 并行"特征检测器"数量（1376 个工位各自做 SiLU），决定非线性变换容量。
2. 高维空间里特征摊开，门控才有得选：摊开 → 挑选 → 打包。
3. 降回 D 让**残差主干始终保持窄宽度**，重活只在支路临时展开——Transformer 的经济结构。

类比：升维=摊草稿纸演算；激活+门控=试各种解法；降维=把答案誊回一页宽答题卡。
（延伸：Geva et al. 2021《Transformer Feed-Forward Layers Are Key-Value Memories》——FFN 中间层即 key-value 记忆库，FFN 参数约占全模型 2/3。）

### 1.7 本节易错点

- ❌ 在 norm 之后才存 residual → 主路丢失原始信号，梯度高速路被破坏
- ❌ `silu(...)` 裸调用 → 必须写 `F.silu`
- ❌ Linear 加 bias → LLaMA 标配 `bias=False`

---

## 2. 第 06 节：MoE Router

### 2.1 核心思想

稠密模型每个 token 过全部参数；MoE 把大 MLP 拆成 E 个专家，Router 为每个 token 只挑 K 个（通常 K=2）。**56B 总参数、每 token 只走 14B** —— 大容量、低激活。

### 2.2 顺序铁律

```
logits → 全局 softmax(fp32) → top-k → 重归一化 → 加权聚合
```

**为什么必须先全局 softmax 再 top-k？** softmax 是相对性运算（分母含全体专家）。工程上三条硬理由：

1. **梯度流向所有专家**：全局分母让冷门专家也有梯度，才能翻身；局部 softmax 下未选中者永远零梯度。
2. **第 07 节 aux loss 需要全量概率分布**（P̄ᵢ 对全部 E 个专家统计）。
3. **数值稳定**：`.float()` 转 FP32 在全维度做一次。

> 🤓 诚实注记：重归一化后，"全局 softmax→top-k→renorm"与"top-k logits→局部 softmax"数值完全相同（分母抵消）。差异真正体现在训练动力学（上面三条），这也是工业实现清一色全局先行的原因。

重归一化的必要性：截走 K 个后其和 ≤ 1（当且仅当落选者全为 0 时 = 1），需按比例放大回 1 以稳定梯度尺度。

### 2.3 参考代码

```python
router_logits = self.gate(hidden_states)                       # [tokens, E]
routing_probs = F.softmax(router_logits.float(), dim=-1)       # [tokens, E]
routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
routing_weights = routing_weights.to(hidden_states.dtype)
```

⚠️ 变量名陷阱：类里是 `router_logits` 和 `self.top_k`，写成 `logits`/`k` 会 NameError。

### 2.4 SparseMoEBlock 分发-聚合（torch.where 技巧）

以 3 token、K=2 为例：

```text
selected_experts = [[2,5],[1,2],[5,7]]   routing_weights = [[.7,.3],[.6,.4],[.9,.1]]
```

轮到 expert 2 时：

```python
token_idx, kth = torch.where(selected_experts == 2)
# token_idx=[0,1] 哪些 token 选了它；kth=[0,1] 各在自己第几个槽位选的
current_state = flat_hidden_states[token_idx]          # 取出这两个 token
out = expert(current_state)
w = routing_weights[token_idx, kth].unsqueeze(-1)      # 成对索引取对应槽位权重
final_hidden_states[token_idx] += out * w              # += 因为 K>1 时一个 token 多专家累加
```

两个精妙点：
- **kth 的作用**：`routing_weights[token_idx, kth]` 成对索引，取的是"该 token 在那个槽位的权重"，不是整行——这就是 topk 必须同时返回 values+indices 的原因。
- **循环专家而非循环 token**：GPU 喜欢同质大批量计算（工业界还会先按专家排序拼批，vLLM/Megatron 的 Token Sorting）。

---

## 3. 第 07 节：负载均衡损失（本主线最难）

### 3.1 问题：Router 会偏科

训练早期 Router 发现"全发给 3 号最省事"→ 少数专家过载、其余闲置，MoE 白背参数退化成稠密模型。对策：主损失外加小辅助项逼 Router 摊匀。

$$L_{aux} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot P_i$$

### 3.2 两个统计量（灵魂）

| 符号 | 问的问题 | 数据来源 | 归一化分母 |
|---|---|---|---|
| fᵢ | 专家 i **实际**接到多少分配？（结果，one-hot 硬统计） | top-k indices | T×K（总分配次数） |
| P̄ᵢ | Router **想**给专家 i 多少概率？（意图，softmax 软统计） | 全量概率 | T（token 数） |

一硬一软相乘的妙处：只有当 Router **既想给、又真给了**同一个专家时 fᵢ·P̄ᵢ 才同时偏大，惩罚打得准。

### 3.3 最小值推导（E=8 具体数字版）

均匀时 fᵢ = P̄ᵢ = 1/E：

$$\sum_{i=1}^{8} \frac{1}{8}\cdot\frac{1}{8} = 8\times\frac{1}{64} = \frac{1}{8}
\;\Rightarrow\; L_{aux}=\alpha\cdot 8\cdot\frac{1}{8}=\alpha$$

求和号内有 **E 项**（每项 1/E²），和是 1/E 而非 1/E²。**完全均匀 ⇔ L_aux = α**，测试断言验证的就是这个。

**为什么乘 E？** 锚定尺度：不乘的话均匀时只有 α/E，专家越多均衡信号越弱；乘 E 后任何配置下完美均衡都恰等于 α，超参含义一致。

记忆钩子：$\sum f_iP_i$ 的最小值 $= \frac{(\sum f)(\sum P)}{E} = \frac{1}{E}$，当且仅当两者都均匀时取到（均值不等式）。

### 3.4 scatter_add_ 速成

`self.scatter_add_(dim, index, src)`：按 index 把 src 的值撒进格子并累加。

```python
buf = torch.zeros(4)
buf.scatter_add_(0, index=torch.tensor([2,0,2]), src=torch.tensor([0.7,0.6,0.4]))
# buf = [0.6, 0, 1.1, 0]   格子0←0.6；格子2←0.7+0.4
```

### 3.5 参考代码

```python
# P̄ᵢ（软视角）：把每个 token 的权重累加到选中专家名下，再按 token 数平均
P_i = torch.zeros(num_experts, dtype=routing_weights.dtype, device=routing_weights.device)
P_i.scatter_add_(0, selected_experts.flatten(), routing_weights.flatten())
P_i = P_i / total_tokens

# fᵢ（硬视角）：one-hot 数次数，按总分配次数 T×K 平均
expert_mask = F.one_hot(selected_experts, num_classes=num_experts)   # [T, K, E]
tokens_per_expert = expert_mask.sum(dim=(0, 1)).float()
f_i = tokens_per_expert / (total_tokens * top_k)

# 组装
aux_loss = alpha * num_experts * (f_i * P_i).sum()
```

易错点：
- ❌ 分母写成 E —— E 只出现在公式末尾的乘法里
- ❌ scatter_add_ 参数混淆 —— index=撒到哪(专家编号)，src=撒什么(权重)
- 实际训练：`total_loss = ce_loss + aux_loss`；α 通常 0.01，过大伤主任务，过小压不住塌缩

---

## 4. 第 08 节：架构技巧

### 4.1 Gemma 的 `(1+w)` 缩放

```text
标准 RMSNorm:  y = x/RMS(x) · w         # w 初始化为全 1
Gemma 版:      y = x/RMS(x) · (1 + w)   # w 初始化为全 0
```

⚠️ 关键澄清：w=0 **不等于零函数**！代入公式 y = x_norm × (1+0) = x_norm，输出是**纯归一化结果**。"稳定"体现在：缩放从精确中性点 1 出发，早期输出尺度完全可控、无随机缩放噪声，之后 w 从 0 被梯度一点点养大。

```python
x_f32 = x.float()                          # FP32 保数值稳定
variance = x_f32.pow(2).mean(-1, keepdim=True)
x_norm = x_f32 * torch.rsqrt(variance + self.eps)
output = x_norm * (1 + self.weight)        # ← Gemma 的全部秘密
return output.type_as(x)                   # 转回原精度
```

### 4.2 Qwen 权重绑定（Weight Tying）

```python
self.lm_head.weight = self.embed_tokens.weight   # 内存级指向，不是复制
```

- **验证**：`data_ptr()` 地址相同即绑定成功
- **梯度**：共享权重的 `.grad` 累加**两路**——Embedding 查表一路 + LM Head 矩阵乘一路；step 一次两头生效，Embedding 直接收到输出侧监督
- **参数账**：Qwen 词表 V=151,936 × d=4,096 ≈ **6.22 亿参数**，FP16 ≈ **1.24 GB**
- **HF 真实代码层级**：顶层 `XxxForCausalLM` 里套 `.model`，那边写的是
  `self.lm_head.weight = self.model.embed_tokens.weight`——迷你实现里少一层，别照抄错对象

对照表：

| 模型 | Embedding/LM Head | Norm |
|---|---|---|
| GPT-2 | 共享 | LayerNorm |
| LLaMA3 | 不绑定 | 标准 RMSNorm |
| Qwen | 共享 | RMSNorm |
| Gemma | 不绑定 | `(1+w)` RMSNorm |

---

## 5. 个人错误档案（本次学习中实际踩过的坑）

1. **残差时机**：知道存 norm 前，理由没说透——本质是残差必须承载原始信号，norm 只属于支路。
2. **裸函数调用**：`silu(...)` → 必须 `F.silu`（住在 `torch.nn.functional`）。
3. **变量名照抄**：算法对了但 `logits`/`k` 应为 `router_logits`/`self.top_k`。
4. **求和项数**：∑fᵢP̄ᵢ 均匀时算成 1/E²——忘了求和号里有 E 项，正确是 1/E。
5. **分母张冠李戴**：把"公式末尾乘 E"错搬到 P̄ᵢ/fᵢ 的除法分母上。
6. **Gemma 误解**：以为 w=0 层变零函数——实际是纯归一化，(1+w) 只是钉死恒等缩放起点。
7. **HF 层级迁移**：真实代码的 `self.model.embed_tokens` 不能照抄进迷你类。

## 6. 隔天自测清单（合上笔记回答）

1. 为什么 Pre-Norm 存残差要在 norm 之前？
2. 先 top-k 再 softmax 错在哪？（提示：从梯度流向和 aux loss 两方面答）
3. torch.where(selected_experts == e) 返回的两个东西各是什么用？
4. fᵢ 和 P̄ᵢ 的分母分别是什么？为什么不同？
5. L_aux 在完全均匀时等于多少？为什么公式要乘 E？
6. 权重绑定后梯度来自哪几路？省多少参数怎么算？

## 7. 下一步路线

- [ ] 补动手：装 CPU 版 PyTorch，把四个 notebook 自己跑通（看懂 ≠ 手撕）
- [ ] 前置查漏：01 RMSNorm / 02 SwiGLU / 03 RoPE / 04 GQA 若有薄弱回炉
- [ ] 硬件主线：GPU 架构与内存 → 通信拓扑 → 显存与 ZeRO → Profiling
