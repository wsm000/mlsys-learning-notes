# Task3 Colab 真实模型实测证据（T4 · Qwen2.5-0.5B · WikiText-2）

来源：`code/task3_colab_real_model_new.ipynb` 在 Google Colab GPU 运行时的导出 ZIP（`20260922_012052.zip`，本目录即其解压内容）。运行时间 2026-09-22 01:20（UTC，Colab 目录名）。

## 环境与来源（provenance.json）

| 项 | 值 |
|---|---|
| GPU | Tesla T4，可见显存 14.56 GiB |
| torch / CUDA | 2.11.0+cu128 / 12.8 |
| transformers / datasets | 4.48.3 / 3.2.0 |
| 模型 | Qwen/Qwen2.5-0.5B，commit `060db6499f32faf8b98477b0a26969ef7d8b9987`，494,032,768 参数 |
| 数据 | Salesforce/wikitext `wikitext-2-raw-v1`，commit `b08601e04326c79dfdd32d625aee71d232d685c3` |
| 权重/优化器 | FP32 权重 + AdamW(foreach=False)，autocast 记录为 bfloat16 |
| 输入 | seq=128、EOS 分隔、无 padding；effective batch=2；3 warmup + 5 measured；逐 batch sha256 见 `inputs.json`，`fixed_inputs.pt` 为固化输入 |

## 本次导出包含

- `baseline.json` / `accumulation.json` / `checkpoint.json`：三策略逐步记录（耗时、loss、allocated/reserved 峰值）、初始与最终 held-out loss、静态预算
- `decisions.json`：三策略全部 `PASS`（同初始权重 identity、同输入哈希、初始/最终验证 loss 阈值、吞吐 ≥ baseline 40%、reserved + 1 GiB ≤ 14 GiB）
- `comparison.png` / `training_loss.png` / `steps.csv` / `environment.json` / `config.json` / `provenance.json` / `inputs.json` / `export_manifest.json`

## 本次导出**不**包含（如实说明）

Notebook 中 4.2 增项的 6 个实测单元格（显存账本精确分解、activation offload、缩小 batch、缩短序列、8-bit 优化器与 4-bit 权重静态实测、预算敏感性扫描、profiler trace 四分类）**在本次导出时尚未运行**，因此 ZIP 中没有对应 JSON/图表。补齐后需重新运行导出单元格再归档。对应缺口文件：`memory_ledger.json`、`activation_offload.json`、`small_batch_eff1.json`、`short_seq64.json`、`quant_adamw8bit.json`、`weight_nf4_static.json`、`budget_sensitivity.json`、`summary.json`、`trace_summary.json`、`profiled_*_trace.json`、`comparison_extended.png`、`training_loss_extended.png`。

## 判定口径

- 数值一致性：同一初始权重、同一批固定输入下的一步 loss 与参数更新接近（阈值见 `decisions.json`），**不是收敛结论**。
- 吞吐门槛 `min_throughput_ratio=0.40`、loss 阈值、reserved + 1 GiB 安全余量 ≤ 预算，均为脚本内显式规则；判决输出 `verdict` 而非人工挑选。
- 8 步短跑；工作负载固定；单卡，无通信测量。