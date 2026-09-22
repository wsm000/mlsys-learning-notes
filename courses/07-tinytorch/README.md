# 从Tensor到Transformer——构建 GPT 模型（TinyTorch 语言专线）

> 14 天 8 Task 学习计划，覆盖 TinyTorch 模块 01-08 + 10-13，跳过 09（卷积）。
> 运行环境：vm-60（`simin@10.5.100.60`，ssh 别名已配好）/ Ubuntu 22.04 / RTX 4090 D 24GB。
> 打卡仓库：[lynnyulinlin-debug/mlsys_on_device](https://github.com/lynnyulinlin-debug/mlsys_on_device)，任务说明在各 Task 的 issue 中。

## Task 列表与打卡地址

| Task | 主题 | 打卡地址 | 截止时间 |
|---|---|---|---|
| Task 0 | 环境配置与整体认识（本次学习第一步） | [issue #1](https://github.com/lynnyulinlin-debug/mlsys_on_device/issues/1) | 0916 中午 12 点 |
| Task 1 | 张量与网络层 | [issue #2](https://github.com/lynnyulinlin-debug/mlsys_on_device/issues/2) | 0919 中午 12 点 |
| Task 2 | 损失函数与数据加载 | [issue #21](https://github.com/lynnyulinlin-debug/mlsys_on_device/issues/21) | 0921 中午 12 点 |
| Task 3 | 自动微分引擎 | [issue #22](https://github.com/lynnyulinlin-debug/mlsys_on_device/issues/22) | 0923 中午 12 点 |
| Task 4-7 | 待公告补充（其余 4 个 Task） | 待发布 | 待发布 |

## 打卡要求（两步走 + 三要素）

1. **第一步**：在本 Task 对应 issue 下上传测试通过截图（未裁剪，含命令与 PASS 结果）。
2. **第二步**：发到群里，群公告截图必须同时包含：
   - issue 标题（如"从Tensor到Transformer——构建 GPT 模型 | Task 0 环境配置与整体认识"）；
   - GitHub ID；
   - 微信昵称。

## 目录结构

- `notes/` —— 按 Task 编号排序的学习笔记。
- `code/taskNN/` —— 实验与取证脚本（在 vm-60 上运行）。
- `evidence/taskNN/` —— 真实运行产生的截图、日志与打卡文案（勿用教程样例或他人设备输出替代）。

## 内容索引

| 笔记 | 说明 |
|---|---|
| [task00-环境配置与整体认识.md](notes/task00-环境配置与整体认识.md) | Task 0：课程定位（Don't import it. Build it.）、vm-60 环境配置实跑、168 测试全通过、产出物清单、避坑与打卡文案。 |
| [task00-讲解-TinyTorch整体认识与工具链.md](notes/task00-讲解-TinyTorch整体认识与工具链.md) | Task 0 学习内容讲解：三层架构与 20 模块、里程碑设计、src→modules→tinytorch 工具链闭环、三层测试体系、语言专线数据流全景、Task 1 预习要点与自测题。 |
| [task01-张量与网络层.md](notes/task01-张量与网络层.md) | Task 1：Tensor/Activations/Layers 实现要点、TOP3 性质、非线性必要性，150 passed。 |
| [task03-自动微分引擎.md](notes/task03-自动微分引擎.md) | Task 3：Autograd 三层结构（Function 基类/建图/反向遍历）、requires_grad·grad_fn·backward 协作、动态图对框架设计的影响、计算图可视化、53 passed + 239 全量回归、两条实测局限。 |

## 代码与证据（Task 0）

| 文件 | 说明 |
|---|---|
| [code/task0/task0_verify.sh](code/task0/task0_verify.sh) | vm-60 上的环境验证脚本，输出即打卡截图内容 |
| [code/task0/make_runshot.py](code/task0/make_runshot.py) | 把实跑日志渲染成终端风格截图 |
| [code/task0/make_roadmap.py](code/task0/make_roadmap.py) | 生成「语言专线」学习路线图 |
| [evidence/task0/task0_env_verify_runshot.png](evidence/task0/task0_env_verify_runshot.png) | 环境验证截图（含 issue 标题、GitHub ID、20/20 导出、25 + 168 passed） |
| [evidence/task0/tinytorch_roadmap.png](evidence/task0/tinytorch_roadmap.png) | 「语言专线」学习路线图 |
| [evidence/task0/task0_verify.log](evidence/task0/task0_verify.log) | 实跑原始日志 |
| [evidence/task0/issue1-checkin.md](evidence/task0/issue1-checkin.md) | issue #1 打卡文案（贴到 issue 评论即可） |

## 代码与证据（Task 1）

| 文件 | 说明 |
|---|---|
| [code/task1/task1_verify.sh](code/task1/task1_verify.sh) | vm-60 上的 Tensor / Activations / Layers 分模块验证入口 |
| [notes/task01-张量与网络层.md](notes/task01-张量与网络层.md) | Task 1 实现思路、TOP3 性质、激活层作用与网络组成 |
| [evidence/task1/task1_vm60.log](evidence/task1/task1_vm60.log) | vm-60 SSH 实跑原始日志，150 passed |
| [evidence/task1/task1_vm60_runshot.png](evidence/task1/task1_vm60_runshot.png) | 未裁剪终端风格测试截图，含 Issue 标题、GitHub ID 和 PASS 结果 |
| [evidence/task1/issue2-checkin.md](evidence/task1/issue2-checkin.md) | issue #2 打卡文案 |

## 代码与证据（Task 3：自动微分引擎，Module 06 Autograd）

> Task 3 为纯 NumPy 实现，与 Task 2 一致改在本地 Windows 实跑。

| 文件 | 说明 |
|---|---|
| [code/task3/task3_verify.ps1](code/task3/task3_verify.ps1) | 一键复现入口：环境信息 → 导出 06 → 模块自测 → pytest（06 / 01-06）→ 引擎证据脚本，日志统一 UTF-8 落库 |
| [code/task3/export_autograd.py](code/task3/export_autograd.py) | mini nbdev 导出：src/06_autograd/06_autograd.py → tinytorch/core/autograd.py |
| [code/task3/run_selftest.py](code/task3/run_selftest.py) | 模块自测运行器：加载 Module 06 源码并执行 test_module() |
| [code/task3/demo_autograd.py](code/task3/demo_autograd.py) | 引擎证据脚本：API 语义 / 解析 vs 数值梯度 / 广播归约 / 菱形累加 / zero_grad·retain_graph / 计算图手算对照 / XOR 训练闭环 |
| [code/task3/pytest_by_file.py](code/task3/pytest_by_file.py) | 逐文件运行 tests/06_autograd 并生成干净汇总日志 |
| [code/task3/make_runshot.py](code/task3/make_runshot.py) | 实跑日志 → 终端风格打卡截图（CJK/emoji 混排渲染） |
| [evidence/task3/task3_06_autograd_runshot.png](evidence/task3/task3_06_autograd_runshot.png) | 截图①：模块自测全过（ALL TESTS PASSED） |
| [evidence/task3/task3_pytest_runshot.png](evidence/task3/task3_pytest_runshot.png) | 截图②：pytest 逐文件 + Module 06 汇总 + 01-06 全量回归（53 / 239 passed） |
| [evidence/task3/task3_demo_runshot.png](evidence/task3/task3_demo_runshot.png) | 截图③：7 组数学/语义校验（中心差分 9.32e-08、手算对照、XOR MSE 0.00117） |
| [evidence/task3/task3_tito_complete_runshot.png](evidence/task3/task3_tito_complete_runshot.png) | 截图④：官网验收方式 `tito module complete 01..06`（06：单元 13/13 → 导出 → 集成 17/17 → marked complete） |
| [code/task3/task3_tito.ps1](code/task3/task3_tito.ps1) | 官方 CLI 完成流复现脚本（装依赖 → jupytext 生成 notebook → tito complete 01..06 → 截图 → 回归） |
| [code/task3/make_notebooks.py](code/task3/make_notebooks.py) | src/*.py → modules/*.ipynb（jupytext），供官方 tito 导出步骤使用 |
| [code/task3/tito_keylines.py](code/task3/tito_keylines.py) | 从 tito 全量日志抽取关键行，生成打卡截图用精简日志 |
| [evidence/task3/task3_tito_complete.log](evidence/task3/task3_tito_complete.log) | 官方 CLI 全量原始输出（520 行） |
| [evidence/task3/task3_tito_keylines.log](evidence/task3/task3_tito_keylines.log) | 官方 CLI 关键行汇总 |
| [evidence/task3/task3_pytest_by_file.log](evidence/task3/task3_pytest_by_file.log) | pytest 逐文件汇总原始日志 |
| [evidence/task3/task3_autograd_demo.txt](evidence/task3/task3_autograd_demo.txt) | 引擎证据脚本原始输出 |
| [evidence/task3/issue22-checkin.md](evidence/task3/issue22-checkin.md) | issue #22 打卡文案（贴到 issue 评论即可） |
