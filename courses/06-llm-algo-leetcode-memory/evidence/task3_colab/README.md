# Task3 Colab 真实模型实测证据（完整版 · T4 · Qwen2.5-0.5B · WikiText-2）

来源：`code/task3_colab_real_model_new.ipynb` 在 Google Colab GPU 的完整运行，导出 ZIP `../task3/20260922_044059.zip`（本目录即其解压内容，24 个文件）。
早前一次不完整运行（`20260922_012052.zip`，仅主表三策略）保留在上级目录备查。

## 环境与来源

| 项 | 值 |
|---|---|
| GPU | Tesla T4，可见显存 14.56 GiB |
| torch / CUDA | 2.11.0+cu128 / 12.8 |
| transformers / datasets | 4.48.3 / 3.2.0 |
| 模型 | Qwen2.5-0.5B，commit 060db6499f32faf8b98477b0a26969ef7d8b9987，494,032,768 参数 |
| 数据 | Salesforce/wikitext wikitext-2-raw-v1，commit b08601e04326c79dfdd32d625aee71d232d685c3 |
| 口径 | FP32 权重 + 优化器，autocast bf16，seq=128、EOS 分隔无 padding、effective batch=2、3 warmup + 5 measured，逐 batch sha256 |

## 实测结果总览

| 策略 | peak reserved | peak allocated | tokens/s | 初始 loss | 最终 held-out loss | 显存节省 | 吞吐比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 9.457 | 8.424 | 408.6 | 3.0038 | 2.6046 | 0% | 1.000 |
| 梯度累积 1x2 | 9.740 | 9.080 | 355.9 | 3.0038 | 2.6038 | -2.99% | 0.871 |
| checkpoint | 9.271 | 8.410 | 395.0 | 3.0038 | 2.6046 | +1.97% | 0.967 |
| activation offload | 9.254 | 8.410 | 261.3 | 3.0038 | 2.6046 | +2.15% | 0.640 |
| 8-bit AdamW | 5.932 | 5.170 | 429.6 | 3.0038 | 2.6033 | +37.27% | 1.052 |

4-bit 权重（仅静态/前向，不训练）：加载后 allocated 0.444 GiB、前向峰值 0.736 GiB、实存权重字节 0.420 GiB（FP32 权重 1.840 GiB，省 76%）、held-out loss 3.1022（FP32 同点 3.0038，质量代价 +0.098）。

## 三份关键明细

- memory_ledger.json：参数 1.8404 / 梯度 1.8404 / Adam 状态 3.6808 = 常驻 7.3617 GiB；激活+临时 0.0365 GiB；峰值-常驻 1.0156 GiB；分配器缓存 1.501 GiB。
- budget_sensitivity.json：Notebook 内按 workload 字符串过滤，基线三策略该字段为空而被排除，只保留 offload/quant。用同一批实测数据与同一判定规则（reserved+1 GiB <= 预算、吞吐 >= baseline 40%、|delta loss| <= 0.15）重算的正确表见 ../../notes/task03-colab-real-model.md 第 5 节：6 GiB 不可行；8/10 GiB 仅 8-bit AdamW；>=12 GiB 全部可行，推荐 8-bit AdamW（吞吐最高）。
- trace_summary.json：baseline 89,670 事件 vs offload 118,221 事件；offload 的 gpu_memcpy 从 54 次 / 0.42 ms 涨到 1,904 次 / 303.28 ms，同步等待从 7 次 / 104.58 ms 涨到 932 次 / 461.13 ms——offload 的代价确实转移到数据传输与同步。

## 补跑说明

- small_batch_eff1.json / short_seq64.json / summary.json 的 7 行版本是主表跑完后在同一会话补跑的（未重新打包 ZIP）；两者的逐步记录、初始/最终 held-out loss 与峰值均来自该次补跑输出。
- decisions.json 只含主表三策略（该单元格在增项之前执行）；offload 与 8-bit 的判定记录在 summary.json。

## 判定口径

- 数值一致性：同一初始权重、同一批固定输入下的一步 loss 与参数更新接近；8 步短跑，不是收敛结论。
- 增项行的 status=COMPLETE 只表示跑完且有限，不是与 baseline 的等价性判定；工作负载相同的五行（三主表 + offload + 8-bit）才可互比显存与吞吐。
- 4-bit 行是推理侧静态/前向测量，不参与训练侧吞吐比较。