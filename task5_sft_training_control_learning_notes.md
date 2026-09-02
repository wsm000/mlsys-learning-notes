# Task5 学习笔记：项目准备、训练控制与端到端 SFT

> 学习来源：Datawhale `llm-algo-leetcode` 监督微调专题 Task 5
>
> 学习方式：助产婆学习法。先通过问题暴露自己的理解，再根据反馈修正概念，最后用计算题和实验判断题验证是否真正掌握。

## 一、学习路线

Task 5 的核心路线是：

```text
32 数据工程
  -> 33 Fine-Tuning Readiness
  -> 12 梯度累积
  -> 11 学习率调度
  -> 13 端到端微调实验
```

这一条路线解决的不是“如何把训练代码启动起来”，而是：

```text
数据是否可靠
  -> 训练是否值得启动
  -> 更新节奏是否正确
  -> 学习率是否与更新节奏一致
  -> 训练结果是否可信、可解释、可交付
```

## 二、32：SFT 数据工程

### 2.1 数据工程先解决什么问题

SFT 训练的输入不是未经处理的原始记录，而是结构稳定的 `prompt + response` 训练样本。

原始数据至少应该统一成三个字段：

```text
instruction / input / response
```

清洗时需要：

- 对字段进行字符串化和 `strip()`；
- 缺少的字段补为空字符串或按规则剔除；
- 检查 `response` 是否为空；
- 检查完全重复样本；
- 检查 prompt 和 response 的总长度；
- 按统一模板把 instruction 和可选 input 组织成 prompt。

数据处理流程应该是：

```text
原始记录
  -> 字段清洗
  -> 数据审计
  -> 按规则过滤/去重
  -> 构造 prompt + response
  -> tokenizer
  -> SFT batch
```

### 2.2 数据审计和数据过滤不是一回事

审计负责回答“数据有什么问题”，过滤负责回答“发现问题后怎么处理”。

仓库中的 `audit_sft_dataset` 默认会统计异常，但不会自动删除记录。其主要统计项包括：

```text
total_samples
empty_response_count
duplicate_count
over_length_count
avg_total_chars
```

`duplicate_count` 的实现口径是：第一条重复记录保留，之后每出现一条相同记录就加一。因此：

```python
records = [A, A, A]
# duplicate_count = 2
```

如果使用 `pandas.duplicated(keep=False)`，三条都会被标记为重复，统计结果会不同。写实验报告时必须说明统计口径。

### 2.3 为什么空 response 是严重问题

在 causal language model 的 SFT 中，通常只让 response 部分参与监督：

```text
prompt   -> label = -100
response -> label = 真实 token
```

`CrossEntropyLoss(ignore_index=-100)` 会跳过 label 为 `-100` 的位置。因此 response 为空时：

- 可能没有任何有效监督 token；
- 这条样本不能提供有意义的训练信号；
- 某些 `mean` reduction 的 loss 可能因为有效元素为零而产生 `NaN`；
- 如果模板仍然追加 EOS，模型甚至可能被训练成“看到问题就直接结束”；
- 即使不报错，也会浪费训练预算。

因此空 response 至少要被标记，并通常应该在正式训练前过滤。不能只因为数据结构能被代码读取，就认为它是有效训练样本。

### 2.4 格式质量和语义质量

数据检查有两个层面：

| 层面 | 典型检查 |
|---|---|
| 结构/格式 | 字段是否存在、类型是否正确、模板是否统一、是否超长 |
| 语义质量 | response 是否正确、是否完整、是否真正回答 instruction |

`32` 主要提供最小结构审计。回答是否正确通常还要依靠：

- 人工抽检；
- 规则检查；
- 独立验证集；
- 任务指标和样例回归测试。

“是否需要数学公式”也属于任务相关的格式规则，不能默认所有样本都必须包含公式。

## 三、33：Fine-Tuning Readiness

### 3.1 Readiness 检查的目的

数据通过审计，只能说明“样本大致可用”，不能说明“这次微调已经值得启动”。

Readiness 进一步检查：

```text
训练计划是否明确
资源预算是否够用
评测目标是否可计算
是否值得升级成完整项目
```

仓库模板至少汇总：

```python
{
    "goal": ...,
    "dataset_size": ...,
    "total_steps": ...,
    "eval_every": ...,
}
```

预算检查至少分为两部分：

```python
time_ok = estimated_hours <= available_hours
memory_ok = peak_memory_gb <= available_memory_gb
budget_ok = time_ok and memory_ok
```

只有目标存在、数据规模大于零且预算通过时，最小模板才会建议：

```python
promote_to_project = True
```

### 3.2 模板逻辑和工程判断要分开

例如：

```python
config = {
    "goal": "让模型更聪明",
    "dataset_size": 20000,
    "total_steps": 1200,
    "eval_every": 100,
    "estimated_hours": 6.0,
    "peak_memory_gb": 14.0,
}
```

可用时间为 8 小时、显存预算为 16 GB 时，模板代码会返回 `promote_to_project=True`，因为目标是非空字符串，时间和显存也满足预算。

但工程上这个目标仍然不合格，因为它没有说明：

- 提升什么能力；
- 用什么指标测量；
- 与什么 baseline 比较；
- 达到什么阈值才算成功。

一个更好的目标是：

```python
config = {
    "goal": "在固定验证集上提升指令格式遵循能力",
    "metric": "Format Accuracy",
    "baseline": 0.72,
    "target_threshold": 0.85,
    "dataset_size": 20000,
    "total_steps": 1200,
    "eval_every": 100,
    "estimated_hours": 6.0,
    "peak_memory_gb": 14.0,
}
```

其中格式通过率应有明确计算方式：

```text
Format Accuracy = 通过格式检查的样本数 / 验证集总样本数
```

baseline 和微调模型必须使用相同的验证集、prompt 模板、生成参数和格式判定器，否则无法把指标变化可靠地归因于微调。

### 3.3 Readiness 还要考虑交付条件

除了时间和显存，还应提前确认：

- adapter、checkpoint 和 tokenizer 是否有保存位置；
- metrics 和 report 是否能落盘；
- 是否需要保存最佳 checkpoint；
- 磁盘空间是否足够；
- 实验是否能用相同配置复现。

这些属于更完整的项目交付检查，不等同于模板中的 `peak_memory_gb`。

## 四、12：梯度累积

### 4.1 核心概念

显存不足时，可以把一个逻辑 batch 拆成多个 micro-batch：

```text
多个 micro-batch
  -> 多次 forward/backward
  -> 累积梯度
  -> 一次 optimizer.step()
```

有效 batch size 为：

```text
effective_batch_size = micro_batch_size * accum_steps
```

在单卡、没有额外并行因素时，如果：

```text
micro_batch_size = 2
accum_steps = 8
```

则：

```text
effective_batch_size = 16
```

梯度累积减少的是每个 micro-batch 的 activation 峰值，不会减少模型参数、梯度和优化器状态的长期占用。

### 4.2 为什么 loss 要除以 accum_steps

对于 8 个 micro-batch，正确的平均梯度近似为：

```text
g_correct = (g1 + g2 + ... + g8) / 8
```

因此每个 micro-batch 反向传播前应执行：

```python
loss = micro_loss / accum_steps
loss.backward()
```

如果忘记除以 `accum_steps`：

```text
g_wrong ≈ 8 * g_correct
```

可以近似理解为学习率被放大 8 倍，可能导致：

- loss 震荡；
- 训练发散；
- 参数更新过大；
- 收敛变慢或不稳定。

使用 AdamW、梯度裁剪或混合精度时，最终参数变化不一定严格是 8 倍，但更新幅度被显著放大这一判断仍然成立。

### 4.3 一次计算题

配置：

```text
micro_batch_size = 2
accum_steps = 8
total_steps = 1200
```

这里的 `total_steps` 指 `optimizer update` 次数，而不是 micro-batch 次数。

结果是：

```text
effective_batch_size = 2 * 8 = 16
loss.backward() = 1200 * 8 = 9600 次
optimizer.step() = 1200 次
```

如果每次有效更新处理 16 个样本，则总处理量为：

```text
1200 * 16 = 19200 个样本
```

对于 20000 条数据，相当于：

```text
19200 / 20000 = 0.96 epoch
```

## 五、11：学习率调度与 WSD

### 5.1 三个阶段

WSD（Warmup-Stable-Decay）把学习率分成三段：

```text
Warmup -> Stable -> Decay
```

例如：

```text
num_warmup_steps = 120
num_stable_steps = 840
num_decay_steps = 240
base_lr = 2e-4
min_lr_ratio = 0.1
```

总更新数为：

```text
120 + 840 + 240 = 1200
```

阶段边界是：

```text
Warmup: 0 <= step < 120
Stable: 120 <= step < 960
Decay: 960 <= step < 1200
```

Warmup 阶段线性升高：

```text
lr = base_lr * step / num_warmup_steps
```

Decay 阶段采用余弦衰减：

```text
decay_step = step - warmup_steps - stable_steps
decay_ratio = decay_step / decay_steps
cosine = 0.5 * (1 + cos(pi * decay_ratio))
lr = min_lr + (base_lr - min_lr) * cosine
```

最小学习率为：

```text
min_lr = base_lr * min_lr_ratio
       = 2e-4 * 0.1
       = 2e-5
```

### 5.2 调度器的步数必须和 optimizer 对齐

正确顺序通常是：

```python
loss.backward()
optimizer.step()
scheduler.step()
optimizer.zero_grad()
```

使用梯度累积时，多个 micro-batch 只对应一次有效更新，所以 scheduler 也只推进一次：

```text
micro-batch 次数 != scheduler step 次数
optimizer update 次数 = scheduler step 次数
```

对于上面的 WSD 配置：

- `step=60`：Warmup，学习率为 `1e-4`；
- `step=500`：Stable，学习率为 `2e-4`；
- `step=1080`：Decay 中点附近，学习率约为 `1.1e-4`；
- Decay 结束时：学习率约为 `2e-5`。

如果错误地每个 micro-batch 都执行一次 `scheduler.step()`，每个 optimizer update 会推进 8 次调度，总计划会在：

```text
1200 / 8 = 150 个 optimizer update 左右
```

提前走完。之后的训练大部分时间都会停留在最小学习率附近，导致学习率计划与实际训练长度不匹配。

## 六、13：端到端 SFT 实验

### 6.1 SFT label 的构造

假设 tokenizer 后：

```text
prompt:   [P1, P2]
response: [R1, R2, EOS]
```

合并输入：

```text
input_ids = [P1, P2, R1, R2, EOS]
```

为了只训练 response：

```text
labels = [-100, -100, R1, R2, EOS]
```

prompt 仍然作为上下文输入模型，但不作为监督目标。否则 loss 会混入“复现用户输入”的任务，模型可能过度学习 prompt 模板，且 prompt 长度会改变 loss 权重。

### 6.2 next-token shift

因果语言模型第 `t` 个位置的 logits 预测第 `t+1` 个 token，因此需要：

```python
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
```

对应关系为：

```text
logits at P1 -> label P2，忽略
logits at P2 -> label R1，计入 loss
logits at R1 -> label R2，计入 loss
logits at R2 -> label EOS，计入 loss
```

如果忘记 shift，就会错误地把当前位置输出与当前位置 label 比较，训练目标发生错位，loss 不再表示正确的 next-token prediction。

如果一个 batch 的所有 labels 都是 `-100`，则没有有效监督目标。这个 batch 没有训练意义，还可能因为交叉熵没有有效元素而产生 `NaN`，应该在 loss 函数中防御性报错。

### 6.3 端到端实验检查什么

第 13 节把以下组件串起来：

```text
数据构造
  -> input_ids / attention_mask / labels
  -> model forward
  -> next-token SFT loss
  -> backward
  -> gradient accumulation
  -> optimizer.step
  -> 周期性 train/val 评估
  -> 实验报告
```

不能只看最终 train loss，还要记录：

```text
initial_train_loss
initial_val_loss
history
final_train_loss
final_val_loss
```

同时检查：

- train/val 是否分开；
- 评估是否按固定间隔触发；
- train 和 val 是否使用相同 loss 口径；
- 参数是否真的更新；
- 重复样本或极小数据能否快速 overfit；
- 生成样例是否发生合理变化；
- 任务指标是否达到阈值；
- 显存、步时和吞吐是否在预算范围内；
- 配置、模型、数据、seed 和环境是否保存。

重复样本 overfit 只是 sanity check，不代表真实泛化能力。它用于验证数据、loss、梯度和参数更新链路是否接通。

## 七、实验结论：accept、tune、reject

统一判断可以写成：

| 决策 | 含义 |
|---|---|
| `accept` | 质量达到目标，资源可接受，artifact 和报告完整 |
| `tune` | 方向可能成立，但质量、配置、训练控制或证据链还需要补齐 |
| `reject` | 多次同口径实验仍未证明优于 baseline，或无法复现、无法交付 |

例如：

```text
initial_train_loss = 1.90
final_train_loss   = 0.40
initial_val_loss   = 1.80
final_val_loss     = 1.75
baseline_format_accuracy  = 0.72
finetuned_format_accuracy = 0.73
target_threshold = 0.85
```

不能 `accept`，因为任务指标远低于 `0.85`。但单次实验还不能立即证明整个方向必须放弃。更合适的阶段性结论是：

```text
decision = "tune"
```

优先检查：

1. train/val 是否重复或存在数据泄漏；
2. labels 是否只监督 response；
3. next-token shift 是否正确；
4. tokenizer 和 prompt 模板是否一致；
5. 训练和评测的生成参数、格式判定器是否一致；
6. 数据质量和格式规则是否真的匹配目标；
7. 是否存在过拟合、学习率过大或训练步数不合适。

如果经过控制变量、多次复现实验后仍只有 `0.73`，且无法接近目标或超过 baseline，再把这个候选方案标记为 `reject`。

## 八、我的最终理解

```text
32：先证明数据结构稳定、监督目标存在、异常可控。

33：再证明目标可测量、计划清楚、训练预算可承受，值得启动项目。

12：用多次 backward、一次 optimizer.step，把 micro-batch 合成有效 batch。

11：学习率和 scheduler 按 optimizer update 推进，不能按 micro-batch 误推进。

13：把数据、loss、更新和 train/val 评估接成闭环，并用实验报告判断结果是否可信。
```

在实际微调中，不能因为“训练跑完了”就宣布成功。可信结论至少需要同时回答：

```text
数据可靠吗？
训练目标正确吗？
更新节奏一致吗？
验证集是否改善？
任务指标是否达标？
资源成本能接受吗？
结果能复现和交付吗？
```

## 九、参考链接

- [监督微调专题总览](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/topic_discussion/fine_tuning_training/intro.md)
- [32 Data Engineering for SFT](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.ipynb)
- [33 Fine-Tuning Readiness](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/33_Fine_Tuning_Readiness.ipynb)
- [12 Gradient Accumulation](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/12_Gradient_Accumulation.ipynb)
- [11 LR Schedulers WSD Cosine](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/11_LR_Schedulers_WSD_Cosine.ipynb)
- [13 End-to-End Fine-Tuning Experiment](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.ipynb)
- [训练微调项目验证清单](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/docs/verification/fine_tuning_projects.md)
