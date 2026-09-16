# llm-algo-leetcode-memory（显存优化方向）

本目录收纳 Datawhale 新课 `llm-algo-leetcode` 中**显存优化方向**的学习记录。与 `03-llm-algo-leetcode-finetune`（同仓库的微调线）分开归档。课程为 Notebook-first 算法实战教程，以 LLM 为主线，通过可运行、可验证、可回顾的 Notebook 从"会看"走到"会写、会调、会优化"。

## 课程信息

- 教程地址：<https://github.com/datawhalechina/llm-algo-leetcode>
- 打卡链接（Task0 issue #148 / Task1 issue #149）：<https://github.com/datawhalechina/llm-algo-leetcode/issues/149>
- 组队链接（系统组队）：<https://m.datawhale.cn/activity/587?register=true>
- 正式学习从下周一启动，打卡自周一起逐步开放；有打卡筛选制度。

## 专题定位

研究**训练和推理中的显存对象、生命周期与预算取舍**，最终通过固定 workload、真实 GPU 测量和 profiling 形成可复现的优化决策。不是单独讲某个技巧，而是回答四个问题：

1. 显存被什么占用？
2. 压力出现在哪个阶段？
3. 优化把代价转移到了哪里？
4. 当前方案是否值得采用？

- **训练侧**：参数、梯度、optimizer state、activation、临时张量。
- **推理侧**：权重、KV Cache、请求并发、临时 attention 空间。
- **共享基础**：dtype、内存层级、带宽、profiling——但项目证据不能混用。

## 显存优化学习路线

推荐从 Part02 PyTorch 算法实战入手，可结合 Part01 数学硬件与系统学习，再按需学习 Part03/Part04。显存方向覆盖：

1. 显存对象与生命周期
2. dtype、模型规模、硬件与显存账本
3. 单机训练显存策略
4. 训练侧测量与预算决策
5. 推理侧 KV Cache 与容量
6. 量化与显存容量扩展
7. 分布式显存与系统级扩展

## Task 列表

| Task | 主题 / 学习材料 | 打卡笔记 | 实测证据 |
|---|---|---|---|
| Task 0（预习） | 显存对象与生命周期：P0 07 Autograd、Part02 17/18 反向传播、topic 02 训练侧显存压力（[Issue #148](https://github.com/datawhalechina/llm-algo-leetcode/issues/148)） | [task00-memory-objects-lifecycle.md](../../05-llm-algo-leetcode-memopt/notes/task00-memory-objects-lifecycle.md)（归档在原 05-llm-algo-leetcode-memopt 目录） | [task0 证据](../../05-llm-algo-leetcode-memopt/evidence/task0/) |
| **Task 1** | **硬件与显存账本**：Part01 01 dtype / 02 参数量与 FLOPs / 06 显存计算与 ZeRO（+03/12 选做、14/topic01/Part02 04 扩展）（[Issue #149](https://github.com/datawhalechina/llm-algo-leetcode/issues/149)） | [task01-hardware-and-vram-ledger.md](notes/task01-hardware-and-vram-ledger.md) | [task1 证据](evidence/task1/) |
| Task 2 | 待发布 | 待写 | 待写 |

### Task1 学习材料

| 层次 | 内容 | 链接 |
|---|---|---|
| 最小 | 01 数据格式与混合精度 | [01_Data_Types_and_Precision.ipynb](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.ipynb) |
| 最小 | 02 参数量与算力推导 | [02_LLM_Params_and_FLOPs.ipynb](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.ipynb) |
| 最小 | 06 显存计算与 ZeRO 优化 | [06_VRAM_Calculation_and_ZeRO.ipynb](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.ipynb) |
| 可选 | 03 GPU 物理架构与内存层级 | [03_GPU_Architecture_and_Memory.ipynb](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb) |
| 可选 | 12 Tensor Core 与混合精度 | [12_TensorCore_and_Mixed_Precision.ipynb](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.ipynb) |
| 扩展 | topic 01 显存账本与指标 | [01_vram_ledger_and_metrics.md](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/topic_discussion/memory_performance_tuning/01_vram_ledger_and_metrics.md) |
| 扩展 | Part02 04 多头注意力（MHA/GQA） | [04_Attention_MHA_GQA.ipynb](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb) |
| 扩展 | 14 FlashAttention 显存模型 | [14_FlashAttention_Memory_Model.ipynb](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.ipynb) |

## 内容索引

| 产出 | 说明 |
|---|---|
| [notes/task01-hardware-and-vram-ledger.md](notes/task01-hardware-and-vram-ledger.md) | Task1 打卡笔记：dtype → 参数规模 → 硬件条件 → 显存账本四问，含 16Φ 账本与真机实测对照、ZeRO 分摊表、4.2 GPU/内存层级/混合精度、4.3 MHA/GQA/MLA 与 FlashAttention |
| [code/task1_shot_41.py](code/task1_shot_41.py) · [42](code/task1_shot_42.py) · [43](code/task1_shot_43.py) · [task1_common.py](code/task1_common.py) | Task1 可复跑脚本：教程函数逐条复跑 + 真机实测 + 自动出图（vm-60 上 `ALL_TESTS_PASS`） |
| [code/04_solved.ipynb](code/04_solved.ipynb) | Task1 增项2 要求的 Part02 04 节解答版 notebook（已执行，含测试通过输出） |
| [evidence/task1/](evidence/task1/) | Task1 打卡截图（4.1/4.2/4.3 + 04 节运行截图）与完整运行日志 |

## 学习方法

沿用本仓库惯例：每个任务产出 `notes/` 打卡笔记 + `code/` 可运行代码 + `evidence/` 实测证据。显存优化的数字必须注明测量环境与口径（torch.cuda.max_memory_allocated / nvidia-smi / profiler 采样区间），优化前后用同一固定 workload 对比，并明确"代价转移到了哪里"。
