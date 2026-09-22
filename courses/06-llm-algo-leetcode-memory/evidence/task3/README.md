# Task3 打卡材料

对应 [Issue #169](https://github.com/datawhalechina/llm-algo-leetcode/issues/169)，4.1 + 4.2。GPU 实验均在 vm-60 执行，无模型下载。

## 提交顺序

1. 复制 `issue169-checkin.md` 内容到 issue 评论编辑框，自行填写微信群昵称。
2. 将 `01_baseline_quality.png`、`02_strategy_budget.png`、`03_profiler_evidence.png` 拖入评论，使用 GitHub 上传生成的链接替换文案中的本地图片路径。
3. 如果希望笔记可在线访问，将笔记发布到自己的仓库后填写笔记链接；本地相对路径不能被 issue 读者访问。
4. 发布后截取包含 **issue 标题 + 自己的 GitHub ID + 微信群昵称** 的页面，发至微信群。

这里的 PNG 是原始实测结果的排版证据图，图内已明确注明，不冒充终端或网页原始截图。若组织者严格要求终端原屏截图，可在终端展示 `task3_run.log` 的 `ALL_TESTS_PASS` 及结果并自行截图。未代发 GitHub 评论或微信群消息。

## 可复核产物

- `task3_results.json`：固定配置、9 次重复、数值一致性、allocated/reserved 峰值、独立阶段测量、预算决策。
- `task3_run.log`：vm-60 原始运行日志与 ALL_TESTS_PASS。
- `task3_environment.log`：GPU、Python/PyTorch、磁盘与源码哈希。
- `task3_cpu_cuda_trace.json` / `task3_offload_trace.json`：baseline 与卸载的真实 CPU/CUDA trace。
- `task3_profiler_table.txt`：profiler 原始聚合统计；嵌套统计不可直接相加。
- `task3_commands.txt`：执行和回传命令。
- `../../code/task3_training_budget.py`：可复跑实验。
- `../../code/task3_render_evidence.py`：从 JSON/trace 生成图片。
- `../../notes/task03-training-measurement-budget.md`：完整学习笔记。

## 复现

在有 PyTorch CUDA 的 vm-60 中：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 task3_training_budget.py --out-dir ./evidence
python3 task3_render_evidence.py --evidence ./evidence
```

出图依赖现有 Pillow 和 DejaVuSansMono 字体。合成残差 MLP 用于验证测量和决策方法；不是完整 LLM 收敛实验。量化、改变有效 batch、缩短上下文仅作理论比较，不混入固定 workload 的实测主表。
