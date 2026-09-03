# Task6 打卡材料:LoRA/QLoRA 小参调参(2026-09-03)

选题:DeepSeek-LoRA调参线为主线,LoRA vs QLoRA 对比收口,LLaMAFactory YAML 可迁移.本地已跑通,vm-60 待真训覆盖.

## 文件清单

| 文件 | 说明 | 来源 |
|---|---|---|
| lab_run.log | lab 完整终端输出:PASS test_lab + 账本/换算/NF4/sweep/recommend | python task6_lora_qlora_lab.py 本机实测 |
| lab_report.json | 7B换算+NF4+sweep+recommend(显存为公式估算,mock 已标注) | 同上脚本生成 |
| vm60_mock_run.log | sweep 链路验证输出:PASS vm60_sweep(mock,simulated=True) | python task6_vm60_lora_sweep.py --mock 本机实测 |
| vm60_sweep_report.json | vm-60 真训结构化报告(最终2000步版,simulated=False;另有20/100步结论见笔记) | vm-60 实测回传,覆盖本地mock版 |
| vm60_real_run.log | 2000步真训完整终端输出:三档train近0/val 5.9到6.4,背诵证据 | vm-60 实测回传 |
| llamafactory_qlora_qwen25_05b_vm60.yaml | QLoRA 4bit 模板(Qwen2.5-0.5B) | lab 脚本生成 |
| llamafactory_lora_qwen25_05b_vm60.yaml | LoRA bf16 模板(Qwen2.5-0.5B) | lab 脚本生成 |

## 本机验证(已执行通过)

PASS test_lab:7B 80908MB/14199MB/4335MB,NF4 mse 0.00057 < uniform 0.00073,预算内最优 C(r32/q_v).
PASS vm60_sweep(真训,simulated=False):20步rank16最优(val 3.41);100步交叉为rank4最优(val 3.25);2000步全档过拟合(train约0.01,val 5.9到6.4,训练样本能背诵但泛化崩).结论取100步rank4.峰值约434MB.

## vm-60 真训(一条命令,几分钟)

python3 task6_vm60_lora_sweep.py --model openai-community/gpt2 --max-steps 20
判读:vm60_sweep_report.json 里 simulated=False 且 train_loss_last 小于 train_loss_first;把该 json 与终端输出传回本目录覆盖即可打卡.

## 理论笔记

见 ../../task6_lora_qlora_notes.md(含 DataWhale 社区与三篇教程地址,非抄教程).
上游:../../task4_sft_lora_learning_notes.md,../../task5_sft_training_control_learning_notes.md,../../task6_60_61_learning_notes.md