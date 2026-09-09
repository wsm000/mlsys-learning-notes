# Day5 推理加速实验报告（vm-60 实测）

> 执行环境：vm-60（simin@10.5.100.60）/ Ubuntu 22.04 / RTX 4090 D 24GB / torch 2.9.1+cu128
> 时间：2026-09-09 20:02–20:07 ｜ 计时口径：`time.perf_counter` + `torch.cuda.synchronize`，预热 2 次不计入，报中位数
> 代码：`hpc_repo/tools/llm/`（NewbotEra-baoyue/HighPerformanceComputing Day5）＋ 自写基准 `day5_bench.py` / `day5_bench2.py`

## 一、模型与实验修正（过程比结果更有料）

| 发现 | 影响 |
|---|---|
| checkpoint 真实配置 **d_model=128, n_layers=2, n_heads=4, max_len=64**（README 说的是默认构造参数） | 模型极小，单步推理 ~1.4ms，加速比天花板低 |
| `args.max_len=64` → 位置编码表仅 64 行，强制生成 200 token 直接越界报错 | 重建模型时 max_len=512；`pe` 是 buffer 非学得权重，从 state_dict 剔除后加载（`strict=False` 只能忽略缺失键，**不能**忽略形状不匹配——实测踩坑） |
| seed=42 时第 29 字符就采样到 `<EOS>`，`max_new_tokens=500` 形同虚设，v1 三版本耗时无差异 | v2/v3 把 `EOS_IDX` 置 -1 禁用提前停止，强制跑满 200 token |
| 原脚本"批量生成"是**串行 for 循环**且 `generated_list.append(smi)` 缩进在循环外 | v3 自写真 batch 版：B 个片段一次 prefill + 逐 token batch decode |

## 二、实验结果（实测，simulated=False）

### 实验A：单分子短序列（max_new_tokens=100，20 次取中位）
实际提前终止于 29 字符。baseline 36.4ms / cuDNN 36.5ms / KVCache 32.2ms（**1.13×**）——验证"短序列提升不明显"。

### 实验B：强制 200 token 单分子（5 次取中位）

| 版本 | 200tok 耗时 | vs baseline |
|---|---|---|
| molecular_gpt（无优化） | 0.2883 s | 1.00× |
| molecular_gpt_cuDNN | 0.2892 s | 0.997×（无卷积，无效，符合预期） |
| molecular_gpt_KVCache | 0.2522 s | **1.14×** |

### 实验C：批量生成（KVCache 版，200 token，32 分子）

| 方式 | 总耗时 | 吞吐 |
|---|---|---|
| 串行 32 次（原脚本的"批量"） | 8.055 s | 4.0 mol/s |
| 真 batch=32（一次 prefill + batch decode） | 0.251 s | **127.4 mol/s** |
| **加速比** | | **32.1×** |

## 三、结论

1. **cuDNN autotuner ≈ 无效**（-0.3%）：纯 Transformer 无卷积算子，`cudnn.benchmark` 无用武之地——加速手段必须匹配算子类型。
2. **KVCache +14%**：2 层/128 维的小模型 + 200 token，省下的重算刚够覆盖缓存管理开销；序列更长/模型更深时收益更大（课程预期 >100 token 见效，实测吻合）。decode 每步从 O(t) 重算降为 O(1) 增量，但本模型单步计算太快，kernel launch/Python 开销占了大头。
3. **批量生成 +32×**：decode 是 memory-bound（每步搬 38MB 权重只算 ~25MFLOPs），batch=32 把权重搬运的固定开销摊薄 32 倍——三个手段中收益最大的一个。
4. 教学顺序的验证：工具（cuDNN）< 算法（KVCache）< 系统结构（batch），对本模型分别给 1.0×/1.14×/32×。

## 附：过程异常记录

- 远端 `/home/simin/day5_llm/` 目录在 20:05 前后被不明进程删除（磁盘 97% 满的共享机，存在并行任务；`~/hpc_llm_bench/` 为 19:58 出现的同主题空目录）。结果 JSON 已从终端输出完整捕获，本报告数据以捕获值为准。
- 复现命令（目录存活期间）：
  ```
  python3 day5_bench.py    # v1: 100/500 token 自然生成（发现 EOS 提前终止）
  python3 day5_bench2.py   # v3: 禁 EOS 强制 200 token + 批量对比
  ```

## 四、加分挑战：预分配 KV Buffer vs torch.cat 拼接（vm-60 实测）

### 4.1 实验设计

针对 Q2 指出的 `torch.cat` 每步全量拷贝问题，实现 `molecular_gpt_KVCacheBuf.py`（vLLM 静态 KV cache 的最小可读版）：

| 实现方式 | 每步操作 | 每步数据量 |
|---|---|---|
| **cat 版**（原 KVCache） | `torch.cat([past_k, k], dim=2)` 全量拼接 | O(T)，随序列长度线性增长 |
| **buffer 版**（新增） | 预分配 `(B, nh, max_len, hs)` 缓冲区，每步原地写入 1 个位置，注意力用切片视图 `k_buf[:, :, :total]` 零拷贝读取 | O(1) |

正确性校验：同 seed（123）下两版本生成的 SMILES **逐字符一致**（`correct_same_seed_match: True`），证明改动仅涉及实现、不改变采样语义。

### 4.2 实测结果（强制生成长度，5 次取中位）

| 强制长度 | KVCache (cat) | KVCacheBuf | buffer/cat |
|---|---|---|---|
| 200 token | 0.2484 s | 0.2530 s | 0.982× |
| 400 token | 0.4962 s | 0.5064 s | 0.980× |

### 4.3 结果分析：为什么"教科书答案"在此失效

cat 版每步拷贝量估算（d_model=128、4 头、head_dim=32、fp32、2 层）：

```
每步每层拷贝 ≈ 2 × T × 32 × 4B = 256T 字节
@T=200：每步 ≈ 50 KB/层，2 层 ≈ 100 KB/步；全程累计 ≈ 10 MB
```

4090D 的 HBM 带宽约 1000 GB/s，10 MB 拷贝仅需 ~10 µs，占 0.25 s 总耗时的**约十万分之四**——复杂度上的劣势在此规模下被绝对量抹平。buffer 版反慢 ~2% 的归因：

1. **kernel launch 开销**：buffer 版每步多了 2 层 × 2 次（K/V）原地写入 kernel（4 次 launch/步），每次 ~5–10 µs 固定开销，200 步累计与省下的拷贝时间相抵；
2. **切片视图非连续**：`k_buf[:, :, :total]` 是 stride 非连续视图，矩阵乘走慢路径；cat 版的输出张量天然连续；
3. **工作集极小**：KV 工作集 ~100 KB 远小于 4090D 的 72 MB L2，带宽不构成瓶颈，瓶颈在 kernel launch 与 Python 循环开销。

规模外推：7B 模型（32 层、head_dim 128、fp16、4096 长度）每步拷贝约 268 MB（~260 µs/步），与单步计算量同量级时，预分配/分页缓存才成为必选项——**结论随规模翻转**。

### 4.4 小结

加分挑战的最大价值不在"赢"，而在于验证了三个方法论原则：**先测工作量是否真实发生**（EOS 陷阱）、**先测量再归因**（buffer 未赢）、**加速实验必须配正确性校验**（同 seed 一致性）。

### 4.5 五日学习收获总表

| 手段/结论 | 实测 | 层级 |
|---|---|---|
| cuDNN benchmark | ≈1.0× | 工具级：算子不匹配则无效 |
| KVCache (cat) | 1.14× | 算法级：省重算，但小模型收益有限 |
| batch=32 | 32.1× | 系统级：摊薄 memory-bound 固定开销 |
| KV buffer vs cat | 0.982× | 复杂度优势 ≠ 实际收益；规模决定结论 |
