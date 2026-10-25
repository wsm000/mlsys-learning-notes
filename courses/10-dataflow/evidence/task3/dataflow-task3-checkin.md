# DataFlow Task3 打卡 · 多模态数据：公式/图表/版式 VQA 处理

> GitHub ID: wsm000 | 微信昵称: empty
> 运行环境: vm-60 · NVIDIA RTX 4090 D (24GB) · torch 2.9.1+cu128 · CUDA 12.8 · Python 3.10.12
> 软件: open-dataflow 1.0.11 ([pdf2vqa]) · mineru 2.6.8 (vlm-transformers) · MinerU2.5-2509-1.2B (本地权重) · Ollama qwen3:1.7b / moondream:1.8b
> 教程: [飞书](https://wcny4qa9krto.feishu.cn/wiki/I9tbw2qnBi0lEakmmAGclTysnFd) · [DataFlow 仓库](https://github.com/OpenDCAI/DataFlow) · [官方文档 Case 6](https://opendcai.github.io/DataFlow-Doc/en/guide/l4872nj0/) / [Case 7](https://opendcai.github.io/DataFlow-Doc/en/guide/vqa_extract_optimized/)

## 做了什么

按教程 Case 6（VQA 算子）+ Case 7（PDF VQA Extraction Pipeline）在 vm-60 上真实跑通"公式/图表/版式 → VQA 数据"全链路，算子链零修改，仅替换两个本地资源适配点（MinerU 后端、抽取 LLM）。

**1. 版式解析（MinerU2.5 VLM，本地权重）**——两份示例 PDF：

| 输入 | 页数 | 条目 | 公式/表格/图 |
|---|---|---|---|
| questionextract_test.pdf（扫描版数学讲义） | 4 | 40 | LaTeX 公式、equation 2 条、图3.13~3.16 带题注 |
| math_question+math_answer.pdf（2025 GDFZ 入学真卷，问答分离） | 2 | 64 | 成绩表转 HTML `<table>`、`$x \triangle y = 5x+2y$` 等公式、5 张题图带题注 |

两份 PDF 数字文本层已损坏（pypdf 抽出乱码/空），纯文本提取路线拿不到内容，VLM 解析是唯一可行路径——这正是本任务的价值点。

**2. Case 6 VQA 算子**：`PromptedVQAGenerator` + 本地 moondream:1.8b（Ollama OpenAI 兼容端点），3 张教材渲染页 OCR/描述/VQA 三种提示词均真实返回。

**3. Case 7 全链路**：PDF_Merger → MinerU(本地 vlm-transformers) → MinerU2LLMInputOperator → ChunkedPromptedGenerator+QAExtractPrompt → LLMOutputParser → QA_Merger → VQAFormatter，最终产出 ShareGPT 格式 `qa.json`（messages + images）。验收脚本 22/22 通过。

## 环境限制与如实声明

- vm-60 出境被封：HuggingFace / ModelScope / GitHub / mineru.net 均不可达，官方的 MinerU API（需 MINERU_API_KEY）与 Flash-MinerU（需 vllm）路线不可用。**MinerU2.5-2509-1.2B 权重（2.2GB）经本机 Clash Verge 代理下载后 scp 上台**；tiktoken cl100k BPE 缓存同样经代理中转。
- 系统缺 python3.10-venv，用 virtualenv + `--system-site-packages` 复用已有 torch 2.9.1+cu128，未重复安装。
- 修了一个本项目特有的依赖冲突：~/.local 的 kernels 0.16.1 与 venv 的 huggingface-hub 0.36.2 在 py3.10 上 ABI 报错，按 transformers 4.57.6 自己的 pin 装 kernels==0.9.1 影子覆盖解决。
- 发现并适配 mineru 2.6.8 的行为：所有 vlm-* 后端输出目录硬编码 `vlm/`，与 DataFlow 算子按 `{backend}/` 拼路径不符，用符号链接垫片解决（算子链未改）。

## 质量边界（重点）

QA 抽取环节官方用 gemini-2.5-pro；本地只有 qwen3:1.7b，实测**不遵守"只输出 id"协议**（照抄原文）、答案 hallucination（16 条里 13 条答 "Yes"）。因此：
- 抽取产物**结构有效、内容错误**，只能证明链路连通，**不能作为训练数据**，也不宣称任何抽取质量指标；
- 公式/图表/版式**解析**环节质量真实可用（见证据中的 md / content_list / layout 可视化）。

## 证据（courses/10-dataflow/evidence/task3/）

- `runshot_task3_mineru_layout_math1/math2.png`：MinerU 版面识别可视化（题号框、表格、插图、公式）
- `mineru_math1/2.md + _content_list.json`：结构化解析产出
- `task3_pipeline.log / task3_verify.log`：全链路日志 + 22/22 验收
- `case6_ocr_results.json`：本地 VLM 的 VQA 结果
- `llm_raw_math1.txt / math*_extracted_vqa.jsonl / fallback_*`：1.7B 抽取质量证据（含降级解析器对照）
- `install_df3.log / verify_df3d.log / smoke_mineru.log`：环境与冒烟

复现：`ssh vm-60 && cd ~/dataflow-task3 && source ./env.sh && python pdf_vqa_extract_pipeline.py && python verify_task3.py`（脚本在本仓库 `courses/10-dataflow/code/task3/`）。
