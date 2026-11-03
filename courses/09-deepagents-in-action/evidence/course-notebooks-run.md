# 第1–2章原始Notebook运行记录

源码：https://github.com/datawhalechina/deepagents-in-action/tree/f93fe0a8be843eae77efaf85448035ddc54ff7ea/notebooks

vm-60路径：/home/simin/course09-quanty/course-source。按notebooks/uv.lock安装独立Python3.12.13，未修改课程源码。2026-10-13实际执行：

```bash
uv sync --project notebooks --locked
uv run --project notebooks --locked python -m course_notebooks.run ch01-agent-harness ch02-quickstart ch02-research-assistant --output-dir /home/simin/course09-quanty/outputs/course-notebooks
uv run --project notebooks --locked pytest notebooks/tests/test_ch01.py notebooks/tests/test_ch02.py -q
```

三份执行器输出均为passed (offline)。对应官方测试输出51 passed in 2.07s。依赖版本、源码/锁指纹与执行状态见course-notebooks/report.json；执行后的ipynb、HTML和Markdown一并保存。

- 第1章：同样echo工具比较create_agent与create_deep_agent，观察框架执行与默认工具集。
- 第2章快速上手：工具schema、串行换算→计算、工具异常与参数验证。
- 第2章研究助手：本地搜索样例、规划状态与虚拟文件；offline不使用真实网页搜索服务。

注意：所有默认脚本模型输出不是对真实模型工具选择能力的证明。真实DeepSeek调用由Quanty独立Trace提供。不声称已运行这三本的live/Tavily版本。
