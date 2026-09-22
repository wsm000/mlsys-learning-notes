# 显存优化 Task3 打卡：训练侧测量与预算决策

微信群昵称：【自行填写】

选择：**4.1 + 4.2**。

学习社区：[DataWhale](https://www.datawhale.cn/) · 教程：[llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode)。

本次在 **vm-60 / RTX 4090 D 24GB / PyTorch 2.9.1+cu128** 实测。固定 6 层残差 MLP、同初始权重与合成数据、effective batch=4、seq=512、FP32，warmup 3 次后重复 9 次。完整 step 仅边界同步，阶段诊断和 profiler 分开执行。

| 策略 | allocated 峰值 MiB | reserved 峰值 MiB | step 中位数 ms | tokens/s |
|---|---:|---:|---:|---:|
| baseline | 230.09 | 258 | 7.838 | 261,295 |
| 梯度累积 1×4 | 146.62 | 164 | 23.431 | 87,407 |
| checkpoint | 149.12 | 178 | 13.442 | 152,358 |
| saved-tensor offload | 147.12 | 174 | 24.670 | 83,015 |

我的理解与结论：

1. 看 compute、memory、transfer、scheduling 证据，不能把 GPU 利用率或总耗时当成瓶颈结论。baseline 阶段诊断中 backward 3.100 ms 最大；没有硬件带宽计数器，不能据此断言计算/带宽饱和。
2. 可复查 baseline 必须固定模型、数据、有效 batch、序列、精度与重复次数，同时记录 allocated/reserved、耗时和数值一致性。单卡 communication 不适用，合成张量切片不代表真实磁盘加载。
3. checkpoint 节省约 35.2% allocated，却让 step 慢约 71.5%。显存下降只有在预算和吞吐约束下才有意义。
4. 以 reserved + 64 MiB 余量、吞吐至少 baseline 25% 为门槛，256 MiB 预算选 checkpoint；512 MiB 预算选 baseline；64/128/192 MiB 无可行方案。预算为人为设定，未制造真实 OOM。
5. baseline 与 offload 的独立 trace 中 memcpy 分别为 4 次/0.173 ms、66 次/10.784 ms，说明卸载确实带来额外搬运。事件累计时间不等于墙钟时间。

同时在 **Colab / Tesla T4 / Qwen2.5-0.5B + WikiText-2 / torch 2.11.0+cu128** 上做了真实模型全量增项实测（同初始权重、同固定输入、逐 batch sha256、3 warmup + 5 measured）：

| 策略 | reserved 峰值 GiB | tokens/s | 最终 held-out loss | 显存节省 | 吞吐比 |
|---|---:|---:|---:|---:|---:|
| baseline | 9.457 | 408.6 | 2.6046 | 0% | 1.000 |
| 梯度累积 | 9.740 | 355.9 | 2.6038 | −2.99% | 0.871 |
| checkpoint | 9.271 | 395.0 | 2.6046 | +1.97% | 0.967 |
| activation offload | 9.254 | 261.3 | 2.6046 | +2.15% | 0.640 |
| 8-bit AdamW | 5.932 | 429.6 | 2.6033 | +37.27% | 1.052 |
| 缩小 batch 4→1 * | 9.369 | 273.3 | 2.6648 | +0.93% | 0.669 |
| 缩短序列 128→64 * | 9.359 | 268.3 | 2.9337 | +1.04% | 0.657 |

* 后两行工作负载改变，只比显存与吞吐，不与主表比 loss。4-bit 权重仅推理侧实测：权重 1.840 → 0.420 GiB（−76%），held-out loss 3.1022（FP32 同点 3.0038）。

真实模型上的账本结论：参数 1.8404 + 梯度 1.8404 + Adam 状态 3.6808 = 常驻 7.3617 GiB，占 allocated 峰值 87.4%，而激活+临时只实测到 0.0365 GiB。所以激活侧四个旋钮（缩 batch、缩序列、checkpoint、offload）全部被压到 ±3% 以内、得不偿失；唯一有量级收益的是压缩常驻状态（8-bit AdamW 省 37.27% 且吞吐还快 5.2%）。profiler trace 也印证：offload 的 memcpy 从 54 次/0.42 ms 涨到 1,904 次/303.28 ms、同步等待从 7 次/104.6 ms 涨到 932 次/461.1 ms，而 gemm 计算量两边一致——代价确实转移到了传输与同步。

**验证：ALL_TESTS_PASS**，四种策略均通过同初始状态的一步 loss 和参数更新一致性门槛。这不代表真实 LLM 长期收敛；量化与缩短上下文仅作理论比较。

完整笔记：【发布自己的 task03-training-measurement-budget.md 后填写可访问链接，或将笔记正文附在此评论下】

以下为真实实测 JSON/日志/trace 的排版证据图。提交时把三张 PNG 拖入 GitHub 编辑框，替换下列本地链接：

![固定配置与数值检查](01_baseline_quality.png)

![显存、吞吐与预算决策](02_strategy_budget.png)

![真实 CPU/CUDA profiler 证据](03_profiler_evidence.png)