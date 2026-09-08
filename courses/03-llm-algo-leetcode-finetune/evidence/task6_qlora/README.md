# Task6 综合项目打卡材料:A线 LoRA原理+对比(2026-09-04终版)

选题:SwanLab SFT/6二选一选A线(lora1原理+对比);DeepSeek-LoRA调参为主线,LoRA vs QLoRA对比收口,LLaMAFactory YAML可迁移(模板,未在磁盘紧张的vm-60上执行cli,已声明)。本地已复验,vm-60真训2000步已回传。

## 文件清单

| 文件 | 说明 | 来源 |
|---|---|---|
| lab_run.log | lab 完整终端输出:PASS test_lab + 账本/换算/NF4/sweep/recommend | python task6_lora_qlora_lab.py 本机实测 |
| lab_report.json | 7B换算+NF4+sweep+recommend(显存为公式估算,mock 已标注) | 同上脚本生成 |
| vm60_mock_run.log | sweep 链路验证输出:PASS vm60_sweep(mock,simulated=True) | python task6_vm60_lora_sweep.py --mock 本机实测 |
| vm60_sweep_report.json | vm-60 真训2000步(simulated=False,best rank4) | vm-60 实测回传 |
| vm60_sweep_20.json | vm-60 真训20步复现(09-04,simulated=False,best rank16,val 3.41,与09-03一致) | vm-60直跑回传(独立文件名) |
| vm60_qlora_real_compare.json | QLoRA真对比:GPT2真权重NF4 mse 0.00083<均匀0.00172;BNB 4bit 205MB<bf16 243MB | `task6_qlora_real_compare.py` vm-60直跑 |
| qlora_real_compare.log | 上条脚本完整终端输出 | vm-60直跑回传 |
| vm60_qlora_train.json | QLoRA真训练三档(4bit基座,20步最优16val3.51;100步最优4val3.32;2000步全过拟合val6.2-6.7) | `task6_qlora_train_sweep.py` vm-60直跑 |
| qlora_train.log | 上条三档训练完整终端输出 | vm-60直跑回传 |
| vm60_real_run.log | 2000步真训完整终端输出:三档train近0/val 5.9到6.4,背诵证据 | vm-60 实测回传 |
| llamafactory_qlora_qwen25_05b_vm60.yaml | QLoRA 4bit 模板(Qwen2.5-0.5B) | lab 脚本生成 |
| llamafactory_lora_qwen25_05b_vm60.yaml | LoRA bf16 模板(Qwen2.5-0.5B) | lab 脚本生成 |
| ../../Task6_综合项目打卡报告_A线_LoRA原理与对比.md | 综合项目打卡报告正文+300字表单粘贴版 | 2026-09-04新增 | 
| ../../vm60_task6_light_check.sh | vm-60磁盘紧张版轻量复核(默认只检查不训练,不装包) | 2026-09-04新增 |

## 本机验证(已执行通过)

PASS test_lab:7B 80908MB/14199MB/4335MB,NF4 mse 0.00057 < uniform 0.00073,预算内最优 C(r32/q_v).
PASS vm60_sweep(真训,simulated=False):20步rank16最优(val 3.41,09-04经vm60_sweep_20.json复现);100步交叉为rank4最优(val 3.25,历史,待补独立json);2000步全档过拟合(train约0.01,val 5.9到6.4,训练样本能背诵但泛化崩,现存json+log,best rank4).结论取100步rank4.峰值约434MB.注:20/100步json与2000步同名先后被覆盖,现盘只保留2000步文件,历史数字见 ../../task6_lora_qlora_notes.md 第四节;复现用 ../../vm60_task6_light_check.sh --run-20/--run-100(写/tmp独立文件).本地--mock默认拒绝覆盖真报告,需--force或换--out。

## vm-60 复核(磁盘紧张版,默认不训练)

```bash
bash vm60_task6_light_check.sh                 # 只检查环境+YAML,不装包不下载
bash vm60_task6_light_check.sh --run-20        # 另加20步冒烟 -> /tmp/vm60_sweep_20.json(独立文件,勿覆盖打卡json)
bash vm60_task6_light_check.sh --run-100       # 另加100步交叉 -> /tmp/vm60_sweep_100.json
```
判读:/tmp下json里 simulated=False 且 train_loss_last 小于 train_loss_first;传回本机请改名(如 vm60_sweep_20.json),勿覆盖 vm60_sweep_report.json(2000步)。

历史真训(已执行,2026-09-03):`python3 ~/task6_vm60_lora_sweep.py --model /home/simin/local_models/gpt2 --max-steps 20 --out /tmp/vm60_sweep_report.json`。

## 理论笔记

见 ../../Task6_综合项目打卡报告_A线_LoRA原理与对比.md(打卡正文)与 ../../task6_lora_qlora_notes.md(理论全文,含 DataWhale 社区与三篇教程地址,非抄教程).
上游:../../task4_sft_lora_learning_notes.md,../../task5_sft_training_control_learning_notes.md,../../task6_60_61_learning_notes.md