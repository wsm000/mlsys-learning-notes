# DataFlow · AI-Ready Data 实战（18 天七段任务）

> 教程主仓：https://github.com/OpenDCAI/DataFlow
> 图文教程（飞书）：https://wcny4qa9krto.feishu.cn/wiki/I9tbw2qnBi0lEakmmAGclTysnFd
> 视频教程（B站）：https://space.bilibili.com/3546929239689711
> 官方文档：https://opendcai.github.io/DataFlow-Doc/zh/

按任务组织笔记、代码与证据。所有实验均在 vm-60（RTX 4090 D 24GB）真实执行；本地资源不可达的部分（外网 API、HF/modelscope 下载）如实标注替代方案。

| Task | 主题 | 目录 |
|---|---|---|
| Task 1 | PDF→模型（1.1–1.2.4） | `~/dataflow-task1`（vm-60），归档见 `evidence/task1` |
| Task 2 | PDF→结构化文本→QA | `~/dataflow-task2`（vm-60），归档见 `evidence/task2` |
| Task 3 | 多模态数据：公式/图表/版式 VQA 处理（3天，已完成） | [notes/task03](notes/task03-多模态数据公式图表版式VQA.md) · [code/task3](code/task3) · [evidence/task3](evidence/task3) · [打卡](evidence/task3/dataflow-task3-checkin.md) |

## Task3 结论速览

- **解析层（公式/图表/版式）**：MinerU2.5-2509-1.2B 本地 VLM 解析真实可用——LaTeX 公式、HTML 表格、带题注插图、版面框全覆盖（4 页 40 项 / 2 页 64 项）。
- **抽取层（QA）**：本地 qwen3:1.7b 无法遵守 id 协议，产物结构有效但内容错误；教程原版需 gemini-2.5-pro。链路 22/22 验收通过，抽取质量如实标注为不可用。
- 网络限制下的完整替代方案：Verge 代理下载模型权重 + tiktoken 缓存 → scp 上台；MinerU 本地 vlm-transformers 后端替代 API/Flash。
