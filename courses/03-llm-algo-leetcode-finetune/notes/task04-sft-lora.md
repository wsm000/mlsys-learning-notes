# Task4 学习笔记：SFT 与 LoRA

对应 notebook：

- [09_SFT_Training_Loop.ipynb](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/09_SFT_Training_Loop.ipynb)
- [10_LoRA_Tutorial.ipynb](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/10_LoRA_Tutorial.ipynb)

## 一、先分清两件事

SFT 和 LoRA 不是同一层面的东西。

- **SFT** 解决的是“模型学什么”
- **LoRA** 解决的是“模型怎么少参数地学”

所以常见做法其实是：**用 LoRA 来做 SFT**。

## 二、SFT：让模型只为 response 负责

### 1. 数据构造的目标

`build_sft_data(...)` 要做三件事：

1. 把 `prompt_ids`、`response_ids`、`eos` 拼成一个完整序列
2. 让 `prompt` 部分不参与 loss
3. 处理截断、padding 和 mask

最关键的一行是：

```python
labels = [-100] * len(prompt_ids) + response_with_eos
```

这里把 `prompt` 全部设成 `-100`，因为 `CrossEntropyLoss(ignore_index=-100)` 会跳过这些位置。  
也就是说，**SFT 训练的是 response，不是 prompt**。

### 2. 为什么要保留 EOS

`response_with_eos = response_ids + [eos_id]`

这样做是为了让模型学会“什么时候结束回答”。  
没有 `eos`，模型更容易一直往下生成。

### 3. 截断和有效监督检查

```python
input_ids = input_ids[:max_len]
labels = labels[:max_len]
valid_supervised = sum(label != -100 for label in labels)
```

这一步有两个目的：

- 控制序列长度
- 防止 response 被截没后，样本失去训练信号

如果截断后没有任何有效监督 token，就应该直接报错，而不是悄悄训练一条“空样本”。

### 4. Padding 和 attention mask

```python
attention_mask = [1] * len(input_ids)
input_ids += [pad_id] * pad_len
attention_mask += [0] * pad_len
labels += [-100] * pad_len
```

这里三者分工明确：

- `input_ids`：模型真正看到什么
- `attention_mask`：哪些位置是有效上下文
- `labels`：哪些位置参与监督

padding 位置要同时被“看不见”和“不学习”。

### 5. next-token 对齐

```python
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
```

这是自回归训练的标准写法。  
第 `t` 个位置的输出，要去预测第 `t+1` 个 token。

如果传入了 `attention_mask`，还会再做一次保护：

```python
shift_labels = shift_labels.masked_fill(shift_attention_mask == 0, -100)
```

这是为了避免 padding 位置误进 loss。

### 6. loss 为什么这样算

```python
loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
loss = loss_fct(
    shift_logits.reshape(-1, shift_logits.size(-1)),
    shift_labels.reshape(-1),
)
```

原因很简单：

- 交叉熵是 next-token prediction 的标准损失
- 展平后才能对 batch 里的每个 token 统一计算
- `ignore_index=-100` 保证 prompt 和 padding 不参与训练

## 三、LoRA：只训练低秩旁路

### 1. 为什么需要 LoRA

全参微调要保存完整权重、梯度和优化器状态，显存压力很大。  
LoRA 的思路是：

- 冻结原始模型
- 只在旁边加一条低秩的可训练分支

这让微调更省参数，也更方便保存和部署。

### 2. `LoRALinear` 的结构

LoRA 里有三部分：

```python
self.linear = nn.Linear(in_features, out_features, bias=False)
self.linear.weight.requires_grad = False
self.lora_A = nn.Parameter(torch.empty(r, in_features))
self.lora_B = nn.Parameter(torch.empty(out_features, r))
```

对应的维度是：

- `A`: `[r, in_features]`
- `B`: `[out_features, r]`

它们组合成低秩更新：

```text
ΔW = B A
```

### 3. 为什么 `B` 要初始化成 0

```python
nn.init.zeros_(self.lora_B)
```

这样一开始 `ΔW = BA = 0`，模型输出和原始基座完全一致。  
这能保证训练从“什么都不改”的状态开始，更稳。

### 4. 前向传播怎么写

```python
result = self.linear(x)
dropped = self.lora_dropout(x)
lora_out = (dropped @ self.lora_A.T) @ self.lora_B.T * self.scaling
result += lora_out
```

可以拆成两路看：

- 主分支：冻结的原始线性层
- 旁路：低秩更新分支

`scaling = lora_alpha / r`，用来控制 LoRA 更新强度。

### 5. 为什么可以合并权重

```python
self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
```

因为数学上：

```text
Wx + BAx = (W + BA)x
```

所以训练结束后可以把 LoRA 直接合并回主权重，推理时就不用额外算旁路了。

## 四、参数量为什么会少很多

普通线性层参数量是：

```text
out_features * in_features
```

LoRA 新增参数量是：

```text
r * (in_features + out_features)
```

当 `r` 远小于输入输出维度时，参数量会少很多。  
这也是 LoRA 的核心优势。

## 五、target_modules 怎么选

最常见的起点是：

```python
["q_proj", "v_proj"]
```

原因是：

- `q_proj` 决定去哪里找信息
- `v_proj` 决定把什么信息传出去

如果效果不够，再扩到：

- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

原则是：**先少后多，先便宜后全面**。

## 六、SFT + LoRA 的完整链路

```text
raw prompt/response
-> build_sft_data
-> DataLoader / collator
-> model forward
-> compute_sft_loss
-> backward / step
```

如果用了 LoRA，只是把“更新谁”换掉了：

- base model 冻结
- 只更新 LoRA 参数

数据流和损失函数的逻辑不变。

## 七、最容易踩的坑

1. `prompt` 没有 mask 成 `-100`
2. 忘了做 `shift`
3. padding 进了 loss
4. `lora_B` 不是 0 初始化
5. 主模型参数没有冻结
6. optimizer 里混进了冻结参数
7. merge 后重复计算 LoRA 分支

## 八、我的学习收获

1. SFT 的关键不在“把序列喂进去”，而在“哪些位置该学”。
2. `attention_mask` 和 `labels=-100` 不是一回事，前者管可见性，后者管监督。
3. LoRA 的核心不是“少写几个参数”，而是把更新预算集中到低秩方向上。
4. `B=0` 这个初始化很重要，它保证了训练初始状态不扰动基座模型。
5. SFT 和 LoRA 可以很好地搭配：一个负责监督目标，一个负责参数高效更新。
