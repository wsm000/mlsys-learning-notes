# mlsys

本目录收纳 MLSys 系列笔记，按 task 编号排序。

## 内容索引

| 笔记 | 说明 |
|---|---|
| [task0_mlsysim_learning_notes.md](notes/task00-mlsysim.md) | MLSys.im 基础与任务 0。 |
| [task1_hello_roofline_learning_notes.md](notes/task01-hello-roofline.md) | Roofline 与任务 1 主线。 |
| [task1_rmsnorm_swiglu_learning_notes.md](notes/task01-rmsnorm-swiglu.md) | RMSNorm / SwiGLU 专题。 |
| [task2_memory_wall_two_phases_learning_notes.md](notes/task02-memory-wall-two-phases.md) | Memory Wall 与两阶段优化。 |
| [task2_rope_attention_learning_notes.md](notes/task02-rope-attention.md) | RoPE / Attention 专题。 |
| [task02-kvcache-mha-gqa-mla.md](notes/task02-kvcache-mha-gqa-mla.md) | KV cache 三机制专题：MHA / GQA / MLA（cache 内容、账本对比、权重吸收与解耦 RoPE）。 |
| [Task3-KVCache-打卡笔记.md](notes/task03-kvcache-打卡笔记.md) | KV Cache 与模型规模权衡（issue #133 打卡底稿）。 |
| [task04-data-algo-system-opt-打卡笔记.md](notes/task04-data-algo-system-opt-打卡笔记.md) | 数据/算法/系统优化与 Data Wall（issue #136 打卡底稿，含 cpu_throughput 含义与 8 vs 50 GB/s 场景）。 |
| [mlsys_task3-5/task3_kv_cache_model_scale.ipynb](notebooks/mlsys_task3-5/task3_kv_cache_model_scale.ipynb) | Task 3 实验 notebook：KV Cache 与模型规模（E1–O4）。 |
| [mlsys_task3-5/task4_data_algo_system_opt.ipynb](notebooks/mlsys_task3-5/task4_data_algo_system_opt.ipynb) | Task 4 实验 notebook：数据/算法/系统联合优化（issue #136）。 |
| [mlsys_task3-5/task5_cluster_cost_integration.ipynb](notebooks/mlsys_task3-5/task5_cluster_cost_integration.ipynb) | Task 5 实验 notebook：集群成本与系统集成（issue #137）。 |
| [学习笔记_模型结构_05-08_LLaMA_Block与MoE.md](notes/模型结构-05-08-LLaMA-Block与MoE.md) | LLaMA Block 组装与 MoE 专题。 |

## 代码与证据

- [mlsys_task3-5/](notebooks/mlsys_task3-5/)：三个实验 notebook 与构建脚本 `_build_notebooks.py`

