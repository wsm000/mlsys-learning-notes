# Task1 证据：硬件与显存账本

> 运行环境：vm-60 · NVIDIA RTX 4090D (24GiB, 114 SM) · torch 2.9.1+cu128 · CUDA 12.8 · Python 3.10
> 复跑：`python3 task1_shot_41.py` / `task1_shot_42.py` / `task1_shot_43.py`（脚本见 [../../code/](../../code/)），三次运行均以 `ALL_TESTS_PASS` 结束。

## 文件

| 文件 | 覆盖内容 |
|---|---|
| `task1_41_runshot.png` | 4.1 最小打卡：01/02/06 节测试通过总览、dtype 账本、LLaMA-7B 参数量分解与 MFU、16Φ 真机账本、ZeRO 分摊表 |
| `task1_42_runshot.png` | 4.2 增项1：03/12 节测试、内存层级、真机 HBM 带宽、FP32/TF32/FP16/BF16 GEMM 吞吐、FP16 上溢 vs BF16 保范围 |
| `task1_43_runshot.png` | 4.3 增项2：Part02 04 节 MHA/GQA/KV Cache 实跑、KV Cache 账本（MHA/GQA/MQA/MLA）、FlashAttention 分块模型、真机 naive vs SDPA 峰值曲线 |
| `task1_41.log` / `task1_42.log` / `task1_43.log` | 三段脚本在 GPU 上的完整 stdout |

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

## 口径说明

- 峰值与常驻均用 `torch.cuda.max_memory_allocated()` / `memory_allocated()`；测量前 `reset_peak_memory_stats()`，并先做一次极小 cuBLAS 调用预热框架工作区（baseline 8.13 MiB），避免固定开销被算进某一段增量。
- 吞吐用 CUDA 同步后的 wall-clock 平均（warmup 5 次、计时 20 次），单次运行存在几 % 抖动；规格峰值取 RTX 4090D 官方稠密算力（FP32/TF32 82.6、FP16/BF16 Tensor Core 165.2 TFLOPS）与带宽 1008 GB/s。
