# Task1 证据：硬件与显存账本

> 运行环境：vm-60 · NVIDIA RTX 4090D (24GiB, 114 SM) · torch 2.9.1+cu128 · CUDA 12.8 · Python 3.10
> 复跑：`python3 task1_shot_41.py` / `task1_shot_42.py` / `task1_shot_43.py`（脚本见 [../../code/](../../code/)），三次运行均以 `ALL_TESTS_PASS` 结束。

## 文件

| 文件 | 覆盖内容 |
|---|---|
| `task1_41_runshot.png` | 4.1 最小打卡：01/02/06 节测试通过总览、dtype 账本、LLaMA-7B 参数量分解与 MFU、16Φ 真机账本、ZeRO 分摊表 |
| `task1_42_runshot.png` | 4.2 增项1：03/12 节测试、内存层级、真机 HBM 带宽、FP32/TF32/FP16/BF16 GEMM 吞吐、FP16 上溢 vs BF16 保范围 |
| `task1_43_runshot.png` | 4.3 增项2：Part02 04 节 MHA/GQA/KV Cache 实跑、KV Cache 账本（MHA/GQA/MQA/MLA）、FlashAttention 分块模型、真机 naive vs SDPA 峰值曲线 |
| `task1_43_notebook_runshot.png` | **Part02 04 节 notebook 运行截图**：解答 cell（TODO 已补全，实际执行计数 In[2]）+ 自带测试真实输出 In[3]（`[PASS] All Tests Passed!`） |
| `task1_41.log` / `task1_42.log` / `task1_43.log` | 三段脚本在 GPU 上的完整 stdout |
| `task1_probe_granular.log` | 颗粒度补测：逐项参数账（52.01M → 198.40/198.40/396.80 MiB）、逐张量字节账（logits 125.0→126.0 MiB）、算子间搬运带宽（softmax 512 MB @835 GB/s）、不可预测项（launch 7.72 µs、reserved 不归还、分配粒度） |
| `task1_probe_fusion.log` | 融合阶梯 A/B/C/D/E（7.537 → 5.289 → 5.029 → 1.914 / 调参后自写内核 1.905 ms）、逐笔回收 2.248 / 0.260 / 3.115 ms 与带宽反推 955 / 1034 GB/s、tile 扫描（最佳 2.003 ms @137.3 TFLOPS，BN=128+stages=3 撞 shared memory 上限）、训练态 A/B/D 对照（峰值 4.06 / 4.00 / 0.19 × S²） |

## 关键结果（当次运行实测）

**4.1（01/02/06 节）**

- GPU 张量 4096×4096：FP32 64.0 MiB / BF16 32.0 / FP16 32.0 / FP8 E4M3 16.0
- LLaMA-7B 参数量复算 6.738 B（embedding 131.07M / attn 2147.48M / ffn 4328.52M / ln 0.26M / head 131.07M）
- 真机 GEMM：BF16 8192³ 141.9 TFLOPS（规格峰值 165.2 → MFU 85.9%）；FP32 4096³ 47.8；TF32 66.9
- 训练 step 账本（TinyGPT P=52.03M, B=8, S=256, FP32+bf16 autocast+AdamW，MiB）：
  参数 200.4（理论 198.5，4Φ）· 梯度张量 198.5（198.5，4Φ）· 优化器状态 400.1（396.9，8Φ）· 常驻 815.1（793.9，16Φ）
  forward 峰值 1065.3 · backward 峰值 1315.3（整步峰值）· activation 峰值增量 ≈ 500.2
  框架 workspace 16.1 MiB —— 用 `memory_allocated` 增量直接当梯度会读到 214.6，多出来的就是它
- ZeRO（7B，FP16+Adam，8×80GB 预留 20%）：DDP 112.00 / ZeRO-1 38.50 / ZeRO-2 26.25 / ZeRO-3 14.00 GB 每卡

**4.2（03/12 节）**

- 真机 GEMM：FP32(无 TF32) 48.7（59.0%）· FP16 142.7（86.4%）· TF32 64.6（78.2%）· BF16 141.5（85.7%）；FP16/FP32 = 2.9×
- 真机 HBM 带宽（读+写 copy）940.0 GB/s = 规格 1008 GB/s 的 93.2%
- 范围实测：`float16(70000) = inf`，`bfloat16(70000) = 70144`（末位偏差 0.206%）

**4.3（Part02 04 / 14 / topic01）**

- 04 节 GroupedQueryAttention 解答版：MHA/GQA 前向与 KV Cache 自回归更新断言通过，`kv_cache[0].shape = (2, 4, 6, 32)`；与 `torch.nn.functional.scaled_dot_product_attention(is_causal=True)` 最大误差 **0.0e+00**
- KV Cache 单 token 单层：MHA 32.00 KB · GQA 4.00 KB · MQA 0.50 KB · MLA 1.12 KB；LLaMA-3-8B 全层：4K=0.500 / 8K=1.000 / 32K=4.000 / 128K=16.000 GiB，真机按 8K 分配实测 1.000 GiB = 理论 1.000 GiB
- 真机 attention 峰值（B=1,H=32,D=128,bf16）：S=512 36.0 vs 4.1 MiB（8.9×）· 1024 136.0 vs 8.1（16.7×）· 2048 528.0 vs 16.3（32.5×）· 4096 2080.0 vs 32.5（64.0×），输出误差 < 7.8e-3

## 颗粒度补测（train/infer 分离）

- **推理态**（N=4096，S² 单趟 1024 MiB）：A 7.537 → B 5.289 → C 5.029 → D 1.914 / E 1.905 ms，峰值 2080 MiB → 32.0 MiB。A→B 省 2.248 ms（反推 955 GB/s）、C→D 省 3.115 ms（反推 1034 GB/s），与实测 copy 939 GB/s 同量级 → 这两笔几乎是纯搬运。
- **训练态**（fwd+bwd，N=4096）：A 22.638 ms / 4160.0 MiB（4.06×S²）、B 18.208 ms / 4096.0 MiB（**4.00×S²**）、D 7.716 ms / **193.0 MiB（0.19×S²）**。
  **折 scale 在训练里仍省 4.430 ms，但不省显存** —— 反向要消费 $P$，且还会再造 $dP/dS$ 同量级临时量，峰值 ≈ 4× S²；只有 FlashAttention 的"重算换保存"（D）才能把它降到 0.19× S²（比 B 小 21 倍）。
- 结论：**推理态的显存账本与训练态不同构**，推理结论不能直接当作训练优化预期。
- **重算粒度**（B=1, S=4096, d=512, H=8, ffn=1376, L=4）：① naive 不重算 59.0 ms / 4847.2 MiB；② naive + 整块 checkpointing 76.3 ms / 2274.0 MiB（0.47× 峰值，**+29% 时间**）；③ FA-backward 19.6 ms / 1776.2 MiB（**0.37× 峰值且 0.33× 时间**）；④ 两者叠加 22.4 ms / **1401.0 MiB**（0.29× 峰值，0.38× 时间）。
  **判据**：峰值被 $S^2$ 占住 → FA-backward（顺手还省时间）；被 $O(N\cdot d)$ 逐层激活占住 → 整块 checkpointing；两者叠加再降一档。
- **FA 之后剩下的峰值是谁**（B=1, S=4096, V=32000, L=4, FA-backward）：消融得每层斜率 ≈104 MiB/层、MLP ≈261 MiB、manual CE 多 242 MiB；**参数完全不变**的干净对照显示，chunked CE 每块 1024/512 token 分别回收 **309.1 / 402.8 MiB**，代价只有 +0.4 / +0.1 ms。
  结论：剩下的大头是 $O(N\cdot V)$ 的 logits 家族（logits + CE 中间量 + 反向 dlogits，每个 $S\times V$ bf16 张量 = 250 MiB），**与 attention 无关**；短序列时大头是 $S^2$，长序列 + 大词表时迁移到 $N\cdot V$。

## 口径说明

- 峰值与常驻均用 `torch.cuda.max_memory_allocated()` / `memory_allocated()`；测量前 `reset_peak_memory_stats()`，并先做一次极小 cuBLAS 调用预热框架工作区（baseline 8.13 MiB），避免固定开销被算进某一段增量。
- 吞吐用 CUDA 同步后的 wall-clock 平均（warmup 3–5 次、计时 10–20 次），单次运行存在几 % 抖动；融合实验的 tile 扫描每档 20 次取平均，训练态每档 10 次。
- 训练态峰值受 allocator 历史影响（同配置重复运行可差 ~1.6%），因此训练态结论按"量级 + 是否改变 S² 倍数"判断，不按 MiB 级差值判断；规格峰值取 RTX 4090D 官方稠密算力（FP32/TF32 82.6、FP16/BF16 Tensor Core 165.2 TFLOPS）与带宽 1008 GB/s。
