# 第15–17章 · 算子优化 Agent（教学目录）

对齐教程：
- 设计：[第16章 算子优化 Agent 设计](../../../docs/part3-agent/chapter16/)
- 实战：[第17章 多轮优化实战](../../../docs/part3-agent/chapter17/)

本目录提供 **vector_add** 非交互跑通样例（第 17 章实战入口）：预置 task / reference / baseline，交给 `kernel_optimize` Agent 自动测峰值、profiling、生成候选并权威裁决。

## 目录

```
chapter15/
├── README.md
├── fixtures/vector_add/
│   ├── baseline.py      # 故意偏慢的 Triton baseline
│   ├── reference.py     # PyTorch oracle
│   └── task.json        # v2 任务合同
├── run_vector_add.sh    # AMD 机一键非交互
└── logs/                # 运行产物（gitignore）
```

## 跑通

```bash
cd code/part3-agent
uv sync
source ./activate-rocm.sh

# 需要 ~/.config/hello-gpu/kernel-agent.env
uv run python -m kernel_optimize --batch chapter15/fixtures/vector_add
# 或
bash chapter15/run_vector_add.sh
```

可视化：`bash chapter16/run_and_visualize.sh`

