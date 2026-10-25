# 课程 10：DataFlow · AI-Ready Data

18 天七段任务实战。本目录收录 Task 1（教程 1.1 至 1.2.4）与 Task 3（多模态数据：公式/图表/版式 VQA 处理）。

## Task 1：PDF→QA→LoRA 微调

- [Task 1 实验报告](task1/Task1_实验报告.md)：PDF→QA→真实 LoRA 微调，以及训练前后对比。
- [1.2.4 实验结果截图](task1/Task1_1.2.4_实验结果.png)：实际日志和结果文件的浏览器展示页截图，非原始终端截图。
- [验收记录](task1/verification.json)：CPU 三步输出 44→42→42；GPU LoRA 48 步；独立重载的 24 个权重张量逐项一致。
- [运行与材料说明](task1/README.md)。

PDF 实验采用本地资源适配：PyMuPDF 提取自带文本的 PDF，Ollama 合成问答，Transformers＋PEFT 微调 Qwen3.5-2B。不是默认 MinerU、多卡 vLLM、LlamaFactory 一键命令的逐项复现。仅 4 条训练数据、1 条留出数据，验证链路，不据此声称泛化提升。

## Task 3：多模态数据（公式/图表/版式 VQA）

- [学习笔记](notes/task03-多模态数据公式图表版式VQA.md)：MinerU 版式解析 + Case 6 VQA 算子 + Case 7 全链路的实测记录与环境限制。
- [打卡文案](evidence/task3/dataflow-task3-checkin.md)。
- [复现代码](code/task3/)：本地改造版 PDF VQA Extraction 流水线、Case 6 脚本、MinerU 冒烟测试、验收脚本。
- [证据](evidence/task3/)：MinerU 版面识别可视化截图、content_list、全链路日志（验收 22/22）、降级解析器对照。

Task 3 同样采用本地资源适配：MinerU2.5-2509-1.2B 本地 VLM 做公式/图表/版式解析（经代理下载权重），Ollama qwen3:1.7b 做 QA 抽取。公式/图表/版式解析质量真实可用；QA 抽取受本地 1.7B 模型限制，产物结构有效但内容错误，不能作为训练数据（教程原版需 gemini-2.5-pro）。

课程来源：[图文教程](https://wcny4qa9krto.feishu.cn/wiki/I9tbw2qnBi0lEakmmAGclTysnFd)、[DataFlow 仓库](https://github.com/OpenDCAI/DataFlow)、[官方文档](https://opendcai.github.io/DataFlow-Doc/zh/)。
