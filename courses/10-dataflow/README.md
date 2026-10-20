# 课程 10：DataFlow · AI-Ready Data

18 天七段任务实战。本目录目前收录 Task 1（教程 1.1 至 1.2.4）。

- [Task 1 实验报告](task1/Task1_实验报告.md)：PDF→QA→真实 LoRA 微调，以及训练前后对比。
- [1.2.4 实验结果截图](task1/Task1_1.2.4_实验结果.png)：实际日志和结果文件的浏览器展示页截图，非原始终端截图。
- [验收记录](task1/verification.json)：CPU 三步输出 44→42→42；GPU LoRA 48 步；独立重载的 24 个权重张量逐项一致。
- [运行与材料说明](task1/README.md)。

PDF 实验采用本地资源适配：PyMuPDF 提取自带文本的 PDF，Ollama 合成问答，Transformers＋PEFT 微调 Qwen3.5-2B。不是默认 MinerU、多卡 vLLM、LlamaFactory 一键命令的逐项复现。仅 4 条训练数据、1 条留出数据，验证链路，不据此声称泛化提升。

课程来源：[图文教程](https://wcny4qa9krto.feishu.cn/wiki/I9tbw2qnBi0lEakmmAGclTysnFd)、[DataFlow 仓库](https://github.com/OpenDCAI/DataFlow)、[官方文档](https://opendcai.github.io/DataFlow-Doc/zh/)。
