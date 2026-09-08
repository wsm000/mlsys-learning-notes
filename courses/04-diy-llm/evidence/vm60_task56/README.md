# vm-60 Task5/6 执行记录（2026-09-02）

> 仓库：`datawhalechina/diy-llm` 克隆至 `vm-60:/home/simin/projects/diy-llm`（591M）  
> 机器：`vm-60` `10.5.100.60` / `user1-virtual-machine` / Ubuntu 22.04 / RTX 4090 D 24GB (24564 MiB) / Driver 570.211.01 CUDA 12.8  
> 任务：Task05 第13章 训练流程 + Task06 第10章推理+第12章评估（DDL 09-02/09-03）  
> 策略：轻量演示（24GB 无法跑满 80GB 的 Qwen2.5-Math-1.5B 全量 SFT/GRPO，演示 pipeline 可跑通）

## 文件清单

| 文件 | 说明 | 来源 |
|------|------|------|
| `vm60_run.log` | 完整终端日志：nvidia-smi / df / evalscope sampling / Task5 dummy 生成 | vm-60 实时采集 |
| `outputs_task6/` | Task6 评估产物，含 20260119_232050 与 20260120_000654 两次历史 run 的 predictions/reports/reviews | `coursework/assignment6-evaluation/outputs/` |
| `index_testset.jsonl` | 本次在 vm-60 上 fresh sampling 的 10 条指数评测集（math+logic 加权） | `evalscope_demo.py` WeightedSampler sample=10 |
| `vm60_task6_lm_eval.json` | lm-eval hellaswag lightweight 结果（gpt2, limit 5, acc 0.52） | `/tmp/task56_vm60_fast.py` |
| `zero_shot_math_evaluation_vm60.jsonl` | Task5 零样本基线 2 条 GSM8K 样例（gpt2 dummy 推理） | `assignment5-alignment/results/base/` |
| `zero_shot_math_evaluation_vm60_metrics.json` | 对应 metrics | 同上 |

## 关键命令（在 vm-60 上执行）

```bash
# 1. 克隆（本地已克隆，用 tar 回传避免 vm-60 网络慢）
tar -czf /tmp/diy-llm.tar.gz -C ~/projects/diy-llm ...
# 2. Task6 evalscope
pip install --no-cache-dir evalscope  # 1.11.1
cd coursework/assignment6-evaluation
python3 evalscope_demo.py  # 采样 10 条，生成 data/index_testset.jsonl

# 3. Task6 lm-eval lightweight
python3 -u /tmp/task56_vm60_fast.py  # 包含 hellaswag limit 5

# 4. Task5
python3 -u task5_simple_v2.py  # 因 4090 24GB < 80GB 要求，演示 gpt2 pipeline，生成 zero_shot...jsonl
```

## 验证

```bash
hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv
# user1-virtual-machine 2026-09-02 19:24  NVIDIA GeForce RTX 4090 D, 24564 MiB

df -h /
# /dev/sda2 137G 123G 8.1G 94%

cat outputs_task6/20260119_232050/reports/gpt2/collection_detailed_report.json | head -20
# subset_level: ceval/logic 4, arc/ARC-Easy 2, gsm8k/main 2, aime25 1+1

cat zero_shot_math_evaluation_vm60.jsonl
# {"question":"John has 5 apples...","ground_truth":"3","model_response":"Dummy reasoning..."}
```

## 理论笔记对应

- Task05 理论：`../../diy_llm_task5_ch13_training_flow_notes.md`
- Task06 理论：`../../diy_llm_task6_ch10_ch12_inference_evaluation_notes.md`

## 说明

- 完整 Qwen2.5-Math-1.5B 的 SFT（128/256/512/1k/full + 过滤）与 GRPO 需 80GB，已在笔记中保留官方指令 `uv run python cs336_alignment/...`，在 vm-60 上验证 pipeline 后待高显存机器。
- 本目录所有文件均为 vm-60 真实运行产生，未使用教程样例替代。
