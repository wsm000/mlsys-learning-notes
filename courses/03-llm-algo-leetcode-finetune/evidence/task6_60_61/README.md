# Task6-60/61 打卡材料（本机实测，2026-09-03）

> 任务：60_LoRA_Fine_Tuning_Project.ipynb + 61_Model_Architecture_Exploration.ipynb（任务页第二链接省略了 `.ipynb`，实为同目录 notebook）
> 机器：LAPTOP-K5HQA0TQ / Windows-10-10.0.19045-SP0 / python 3.10.2 / device=cpu / stdlib only
> 策略：CPU-first 模板决策链路验证；Step6 真实模型训练保持 RUN_REAL_TRAINING=False（与 Notebook 默认一致）

## 文件清单

| 文件 | 说明 | 来源 |
|------|------|------|
| `60_run.log` | Task60 完整终端输出：PASS + audit/mask/config/账本/summary/readiness/decision | `python task60_lora_project.py > .../60_run.log` 本机实测 |
| `61_run.log` | Task61 完整终端输出：PASS + baseline/summary/pair/decision | `python task61_arch_exploration.py > .../61_run.log` 本机实测 |
| `60_lora_project_report.json` | fine-tuning-project/v1 报告：config/baseline/candidates/quality/resources/artifacts/decision/environment | Task60 脚本生成 |
| `60_audit_mask_summary.json` | audit + mask + summary + readiness 摘要 | Task60 脚本生成 |
| `61_architecture_report.json` | 61 报告：baseline/candidates/summary/pair_example/decision/environment | Task61 脚本生成 |

## 关键命令（本机已执行通过）

```bash
chcp 65001
$env:PYTHONIOENCODING='utf-8'
python task60_lora_project.py
python task61_arch_exploration.py
```

## 验证

```text
60: PASS test_lora_project_template → decision accept（param_reduction=0.9, mem+256MB, time-2.0ms, val+0.02, ready=True）
61: PASS test_architecture_project_template → accept small_norm（param-4, mem-100MB, score+0.06, deploy+0.05）
```

## 理论笔记对应

- `../../task6_60_61_learning_notes.md`
- 上游：`../../task4_sft_lora_learning_notes.md`（09/10），`../../task5_sft_training_control_learning_notes.md`

## 说明

- 资源数（params/mem/step time/loss）为 Notebook 模板演示口径，函数耗时（audit/mask/total_wall）为本机真实测量；未使用教程样例或他人设备输出替代。
- 完整 Qwen LoRA SFT 需 GPU/大显存，已在 60 Step6 保留官方开关，本目录先交付 CPU-first 决策模板证据。
