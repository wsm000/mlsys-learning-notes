# day5_agent · 第 14–17 章教程镜像与运行指引

本节对应 Day 5 打卡（Agent 生成、评测并迭代优化候选）。所有教程原文来自
[datawhalechina/hello-gpu](https://github.com/datawhalechina/hello-gpu) `dev` 分支，已镜像到本目录，虚拟机/云工作区可直接对照运行。

## 目录结构

```text
day5_agent/
├── tutorial/
│   ├── docs/part3-agent/chapter14~17/index.md    # 各章正文
│   ├── docs/part3-agent/chapter17/optimization-report.md  # 参考跑次的完整工具调用链报告
│   ├── docs/cloud/amd-radeon-cloud-index.md      # 开发者云注册与免费算力说明
│   ├── code/part3-agent/                          # kernel_optimize 主循环/工具 + 评测器 + fixtures
│   │   ├── kernel_optimize/  (agent.py / tools.py / prompts.py …)
│   │   ├── chapter14/        (evaluate.py / worker.py / task_spec.py / source_policy.py …)
│   │   └── chapter15/fixtures/vector_add/  (baseline.py / reference.py / task.json)
│   └── notebooks/part3-agent/chapter14~17.ipynb  # 打卡运行用的 notebook
│       └── _cell_map.txt                          # 每个 notebook 的单元格地图（markdown 标题 + 0 基下标）
```

## 打卡前的三个前置

1. **注册开发者云**：<https://developer.amd.com.cn/login?source=91kadjjnI>（微信扫码最快），完善资料领 100 小时免费算力。
2. **获取 DeepSeek-V4-Flash-0731 API Key**：在平台 Public Free Model APIs 里选择该模型生成 Key（Token Factory），按章节 16 notebook 的模型接口说明粘贴。
3. **复制 code/part3-agent 到可运行 ROCm 的环境**：
   ```bash
   cd code/part3-agent
   uv sync && source ./activate-rocm.sh
   uv run python -c "import torch; print(torch.cuda.get_device_name(0), torch.__version__)"
   ```

## 打卡执行顺序（机器人照抄）

1. 读完 `docs/part3-agent/chapter14/index.md`（什么是 Agent、权威 vs 自由工具）。
2. 跑 `notebooks/part3-agent/chapter15.ipynb` 步骤 2–7（人工调一遍工具，认识裁判）。
3. 跑 `notebooks/part3-agent/chapter17.ipynb` 步骤 1–8（Day 5 打卡主战场），
   产物在 `runs/part3-agent/chapter17/<run_id>/`：`baseline-benchmark.json`、`trajectory.jsonl`、
   `best.py`、`best-benchmark.json`、`best-profile.json`、`agent-report.md`、`comparison-report.md`、`viz/`。

## 三张截图对应的单元格（chapter17.ipynb）

| 截图 | 单元格（markdown 标题；括号内为 0 基下标） | 要显示的内容 |
|---|---|---|
| 1 Agent 运行轨迹 | 「步骤 5：运行完整多轮优化」(18)，可同屏「步骤 4：建立本次实战工作区并测 baseline」(16) | 工具调用 `[step N] compile_kernel/bench_kernel/profile_kernel/accept_candidate` 与返回摘录；baseline bench JSON |
| 2 候选源码与裁决 | 「步骤 6：读取轨迹并复测最终 best」(20) + 终端 `cat runs/.../best.py` | 轨迹表（status/accepted/improvementFraction）、门禁 `ok:true`、接受/拒绝 reason |
| 3 三版本对比 | 「步骤 8：生成对比报告」(24) + 自制三行表 | baseline vs Day 4 人工版 vs Agent 的 median_ms 与加速比（同一 task 契约口径） |

未加速也是有效结果：保留 `trajectory.jsonl` 与 `agent-report.md`，在结论中说明原因，不要改任务/输入/评测规则。

## 单元格地图速查（章节 → 步骤标题）

- chapter14.ipynb：步骤2 准备依赖(12) · 步骤3 定位 ReAct 主循环(14) · 步骤4 查看工具 schema(16)
- chapter15.ipynb：步骤2 依赖(12) · 步骤3 GPU 环境(14) · 步骤4 工作区(16) · 步骤5 baseline 与候选(18) · 步骤6 依次调用权威工具(20) · 步骤7 晋升记录(22)
- chapter16.ipynb：步骤2 定位仓库(13) · 步骤3 依赖(15) · 步骤4 配置模型接口(17) · 步骤5 检查接口与频率(19) · 步骤6 GPU 环境(21) · 步骤7 batch 工作区(23) · 步骤8 多轮优化 Agent(25) · 步骤9 轨迹(27)
- chapter17.ipynb：步骤1 定位仓库(10) · 步骤2 依赖/字体/模型接口(12) · 步骤3 模型与 GPU(14) · 步骤4 baseline(16) · 步骤5 多轮优化(18) · 步骤6 轨迹与 best(20) · 步骤7 可视化(22) · 步骤8 对比报告(24)