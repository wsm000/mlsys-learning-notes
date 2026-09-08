# Task6 综合项目学习笔记：60 LoRA 微调项目 + 61 架构验证

对应 notebook：

- [60_LoRA_Fine_Tuning_Project.ipynb](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.ipynb)
- [61_Model_Architecture_Exploration.ipynb](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/61_Model_Architecture_Exploration.ipynb)（任务页链接省略了 `.ipynb` 后缀，实为同目录 notebook）

本地实测（CPU-first，本机证据）：

- `task60_lora_project.py` → `PASS test_lora_project_template`，见 `打卡材料/task6_60_61/60_run.log`
- `task61_arch_exploration.py` → `PASS test_architecture_project_template`，见 `打卡材料/task6_60_61/61_run.log`
- 报告：`60_lora_project_report.json`（`fine-tuning-project/v1`）、`61_architecture_report.json`
- 环境：`LAPTOP-K5HQA0TQ Windows-10-10.0.19045-SP0 python=3.10.2 device=cpu`，stdlib only，无 torch/numpy
- Step6 真实模型训练保持 `RUN_REAL_TRAINING=False`（与 Notebook 默认一致），本次只做模板决策链路验证，不声称完整微调结论

## 一、60 把 09–13 收成一份交付报告

`60` 不重复实现训练循环，而是把前面已跑通的机制收口：

```text
09 SFT data       input_ids / attention_mask / labels
      │
10 LoRA           target modules / rank / alpha / dropout
      │
11 Scheduler      lr schedule counted by optimizer update
      │
12 Accumulation   micro batch -> effective batch
      │
13 E2E report     initial/final train loss + val loss
      │
      ▼
60 LoRA project   data audit + loss mask + parameter ledger + artifacts + decision
```

项目页最小产物（必须同时记录，缺一不可）：

| 模块 | 必须记录 | 用途 |
|---|---|---|
| 数据 | 样本数、空 response、重复样本、超长样本 | 证明训练输入可信 |
| Loss | supervised tokens、padding supervised tokens | 证明 loss 口径正确 |
| 配置 | target modules、rank、alpha、dropout、lr、effective batch | 保证可复现 |
| 账本 | trainable params、param ratio | 证明 LoRA 是否省参数 |
| 训练结果 | train/val loss、step time、peak memory | 判断效果和成本 |
| 交付 | adapter、tokenizer、merge check、sanity generation | 判断是否能交付 |
| 决策 | accept / tune / reject | 输出项目结论 |

`60` 作为 60–65 统一模板，外层报告固定为 `fine-tuning-project/v1`：`config / baseline / candidates / quality / resources / artifacts / decision / environment`。

## 二、60 的 5 个 TODO 到底在判断什么

### TODO1 audit_sft_examples：数据可信度先行

遍历 `prompt/response` 二元组，统计 `total_samples / empty_response_count / duplicate_count / over_length_count / avg_total_chars`。

- 空 response → 该样本没有有效监督，loss 会被稀释
- 重复 `(prompt, response)` → 小数据下放大小过拟合风险（测试用例里第 1、3 条完全重复，`duplicate_count=1`）
- 超长（`len(prompt)+len(response) > max_total_chars`）→ 会改变截断和显存口径
- 本机演示用干净 2 条样本：`total=2, empty=0, dup=0, over=0, avg=32.0`

### TODO2 loss_mask_report：SFT 最关键的正确性检查

把 `attention_mask` 和 `labels` 展平后对齐：

- `labels != -100` 的 token 才会进 loss
- `attention_mask == 0` 的 padding 不应进 loss，否则就是口径 bug
- 输出 `total / non_padding / supervised / padding_supervised / supervised_ratio=supervised/non_padding`

测试口径：`mask=[[1,1,1,0],[1,1,0,0]]`, `labels=[[-100,7,8,-100],[-100,9,-100,3]]` → `total=8, non_padding=5, supervised=4, padding_supervised=1, ratio=0.8`。`padding_supervised=1` 来自第二行最后一个位置（mask=0 但 label=3）。

本机干净演示：`mask=[[1,1,1,0]], labels=[[-100,7,8,-100]]` → `total=4, non_padding=3, supervised=2, padding_supervised=0, ratio=0.6667`。

### 给定实现：config + 参数账本

- `effective_batch_size = micro_batch_size * accum_steps`（与第 12 节梯度累积口径一致；测试 `2*4=8`）
- 单层 LoRA：`rank*(in_dim+out_dim)`（A 是 `rank*in`，B 是 `out*rank`），只统计 adapter，不含冻结底座
- 完整线性层：`in_dim*out_dim`（本节不计 bias，突出主线）
- 占比：`trainable/total`。测试 `in=out=8, rank=2 → 32/64=0.5`
- 真实维度换算（本机实测输出）：`4096/rank8 → 65,536 / 16,777,216 = 0.3906%`；`4096/rank16 → 0.7812%`；`8192/rank16 → 0.3906%`。rank 翻倍则占比翻倍，hidden 翻倍则分母平方增长、占比减半

### TODO3 summarize_lora_project：delta 方向别写反

- 资源类：`baseline - lora`，正数表示 LoRA 更省/更快
- loss 类：`lora - baseline`，正数表示 LoRA 更差
- 测试：`baseline(1000, 20.0ms, 1024MB, 0.40/0.50)` vs `lora(100, 22.0ms, 768MB, 0.42/0.52)` → `param_reduction=0.9, mem_delta=256.0, time_delta=-2.0, train_delta=0.02, val_delta=0.02`
- 注意 `time_delta=-2.0` 是负数：LoRA 旁路带来了 2ms 额外开销，但仍在可接受阈值内（见 TODO5）

### TODO4 check_lora_project_readiness：训练前闸门

6 个闸门，任一命中即 `ready=False`：`empty_response / duplicate_examples / padding_supervised / no_supervised_tokens / merge_not_checked / sanity_generation_not_checked`。这里只判“能不能信”，不直接判 accept/reject。

### TODO5 recommend_lora_decision：决策顺序是重点

判断顺序固定（错序会导致测试的 tradeoff 分支失败）：

1. `!ready → tune`（先修可信度）
2. `param_reduction < 0.5 → reject`（省得不够就不用 LoRA）
3. `val_loss_delta > 0.03 → tune`（优先调 rank/插层/lr）
4. `!(mem_gain_ok || speed_ok) → tune`（`mem>=128MB` 或 `time>=-3.0ms` 任一成立即可；测试 `mem=32, time=-6.0` 双不达标 → tune）
5. 否则 `accept`

本机干净链路：`param_reduction=0.9, val_delta=0.02, mem=256.0, time=-2.0, ready=True` → `accept`。

> 易错：LoRA 省的是“可训练参数 + 优化器状态”，不是底座权重消失了；val loss 必须单独看，train loss 接近不代表可交付。

## 三、61 把 04–13 收成结构验证报告

```text
04 Attention      attention pattern / head grouping / masking
      │
05 Block          norm / attention / FFN / residual wiring
      │
08 Tricks         architecture-level efficiency constraints
      │
09 SFT            input_ids / labels / loss mask consistency
      │
13 E2E report     train loss / val loss / step time / memory
      │
      ▼
61 Architecture   baseline vs candidate + parameter ledger + delivery decision
```

最小产物：baseline（参数/显存/step time/score，缺一不可）+ candidate（改动模块/参数变化/资源变化）+ 对比（score delta、memory delta、部署影响）+ 决策（accept/tune/reject）。

### TODO1 validate_architecture_baseline：baseline 先合法

必填 `params / memory_mb / step_time_ms / score / deploy_cost`，缺失记 `missing: xxx`；另校验 `params>0, memory>0, step_time>0, deploy_cost>0`。注意 `score` 只要求存在，不判正负（0 也是合法分数）。baseline 不合法时决策直接 `reject + repair_baseline_measurement`，不进入候选比较。

### TODO2 summarize_architecture_candidates：先成池，不先下结论

统计 `candidate_count / names / best_candidate(按 score 最高) / param_deltas=params-baseline / changed_module_union`。测试 `small_norm(-4) / wide_ffn(+8)`，最佳为 `small_norm(0.72)`。

### TODO3 compare_architecture_pair：统一差分

`param/memory/step_time/score/deploy` 全部 `candidate - baseline`。测试 `small_norm vs baseline` → `param=-4, mem=-100, score=+0.06, deploy=+0.05, step=-2.0`。

### TODO4 recommend_candidate：预算 + 部署 + 资源三层漏斗

顺序：baseline 校验 → 参数预算过滤 → 部署代价过滤 → 显存/step_time 联合选池 → 按 `score最高、显存最小` 选优 → accept/tune/reject。

- 无参数可行 → `reject + reduce_candidate_scope`
- 有参数可行但部署全超 → `tune + refine_modules_or_capacity`（选部署增量最小者）
- 联合池：`both_ok || step_ok || mem_ok || deploy_ok`（宁可退而求其次，不直接 reject）
- 最终 `accept` 要求四条件全满足：`score_delta>0 && deploy<=阈值 && mem<=阈值 && step<=阈值`
- `score>0` 但资源/部署任一超 → `tune`；`score<=0` → `reject + fallback_to_baseline`

测试矩阵（`baseline 100/1200MB/92ms/0.66/1.0`）：

| 场景 | 预算/边界 | 结果 |
|---|---|---|
| small_norm vs wide_ffn，`budget=102, deploy<=0.1, mem<=0, step<=4` | wide_ffn 超预算出局 | accept small_norm |
| memory_heavy(0.78) vs balanced(0.71)，`mem<=80` | heavy mem +250 出局 | accept balanced（分数不是唯一标准） |
| 同上但 `deploy<=0.03` | 两者 deploy +0.08/+0.06 全超 | tune balanced |
| slow_gain，`step<=4` 但实际 +7.0 | step 超 | tune |

本机输出与测试完全一致：`accept small_norm, promote_to_extended_eval`。

## 四、本机复现命令（已验证）

```bash
chcp 65001
$env:PYTHONIOENCODING='utf-8'
python task60_lora_project.py
python task61_arch_exploration.py
# 或重定向为 UTF-8 日志：
python task60_lora_project.py > 打卡材料/task6_60_61/60_run.log 2>&1
python task61_arch_exploration.py > 打卡材料/task6_60_61/61_run.log 2>&1
```

关键输出（`60_run.log`）：

```text
PASS test_lora_project_template
audit: total=2, empty=0, dup=0, over=0, avg=32.0
mask: total=4, non_padding=3, supervised=2, padding_supervised=0, ratio=0.6667
hidden=4096, rank=8 -> trainable=65,536, full=16,777,216, ratio=0.3906%
summary: param_reduction=0.9, mem_delta=256.0, time_delta=-2.0, loss_delta=0.02/0.02
readiness: ready=True, issues=[]
decision: accept
```

关键输出（`61_run.log`）：

```text
PASS test_architecture_project_template
baseline_check ready=True
summary best_candidate=small_norm, param_deltas small_norm=-4 wide_ffn=8
pair_small_norm param=-4 mem=-100 score=+0.06 deploy=+0.05
decision accept small_norm promote_to_extended_eval
```

## 五、打卡结论（一句话版本）

- 60：干净数据 + 正确 loss 口径 + `q/v_proj rank8` 下，LoRA 省 90% 可训练参数、省 256MB、慢 2ms、val +0.02 → **accept**，下一步可试 rank16/扩 target modules 做 tune 分支对照。
- 61：同预算同部署边界下，`small_norm` 以更少参数（-4）、更少显存（-100MB）、更快 2ms 换 +0.06 分 → **accept 并进入 extended eval**；`wide_ffn` 参数/显存/部署全涨 → 淘汰；只看分数会误选 `memory_heavy`，必须联合资源漏斗。

## 六、与 Task4/5 的衔接

- Task4（09/10）：SFT 只学 response（prompt 置 -100）+ LoRA 只训 A/B（B 零初始化保证 ΔW=0 起步）→ 60 的 TODO1/TODO2 就是这两件事的项目级检查。
- Task5（13 + vm60）：Task5 验证训练流程能跑通；60 追问“省的参数是否值得交付”；61 追问“改的结构是否值得继续投入”。三者共用 `train/val loss + step time + peak memory` 同一口径。
