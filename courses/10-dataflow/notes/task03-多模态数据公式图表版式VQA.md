# Task3 · 多模态数据：公式/图表/版式 VQA 处理（3 天）

> 教程主仓：https://github.com/OpenDCAI/DataFlow
> 图文教程（飞书）：https://wcny4qa9krto.feishu.cn/wiki/I9tbw2qnBi0lEakmmAGclTysnFd
> 官方文档：https://opendcai.github.io/DataFlow-Doc/zh/
> 对应文档页：[Case 6. Visual Question Answering (VQA)](https://opendcai.github.io/DataFlow-Doc/en/guide/l4872nj0/)、[Case 7. PDF VQA Extraction Pipeline](https://opendcai.github.io/DataFlow-Doc/en/guide/vqa_extract_optimized/)
> 官方流水线：`dataflow/statics/pipelines/api_pipelines/pdf_vqa_extract_pipeline.py`（本仓库 `code/task3/` 下有本地改造版）

## 环境与证据边界（先讲清楚能说什么、不能说什么）

| 项 | 实际情况 |
|---|---|
| 机器 | vm-60（Linux x86_64，RTX 4090 D 24GB，CUDA 12.8） |
| Python / torch / numpy | 3.10.12 / 2.9.1+cu128（venv 以 `--system-site-packages` 复用 `~/.local` 的 torch，未重复安装）/ 1.26.4 |
| DataFlow / MinerU | open-dataflow 1.0.11（`[pdf2vqa]` extra）/ mineru 2.6.8（`vlm-transformers` 后端） |
| 版式解析模型 | MinerU2.5-2509-1.2B（2.2GB，本地权重，经本机 Verge 代理从 HuggingFace 下载后 scp 上台） |
| VQA 视觉模型 | moondream:1.8b（Ollama 本地，1.62GB） |
| QA 抽取 LLM | qwen3:1.7b（Ollama 本地 OpenAI 兼容端点，教程原版用 gemini-2.5-pro API） |
| MinerU API / Flash-MinerU(vllm) / HF / modelscope / github | **均不可用**：vm-60 出境被封（DNS 污染/连接失败），无 MINERU_API_KEY |
| 因此本任务**不报告** | 任何"抽取 QA 语义质量"结论；1.7B 本地模型的抽取结果结构有效但内容错误（见 §5） |

## 1. 任务链路（官方 Case 7 七步）

```
PDF_Merger → FileOrURLToMarkdownConverter(Local/API/Flash, MinerU 版式解析)
→ MinerU2LLMInputOperator(展平+重新编号 id)
→ ChunkedPromptedGenerator + QAExtractPrompt(LLM 按页抽 <qa_pair>，只输出 id 列表)
→ LLMOutputParser(id 回填原文+图片，写 extracted_vqa.jsonl，拷 vqa_images/)
→ QA_Merger(按章节+题号配对问答，写 merged_qa_pairs.jsonl/.md)
→ VQAFormatter(转 ShareGPT: messages + images)
```
Case 6 是其中的图像入口：`PromptedVQAGenerator` + `APIVLMServing_openai`，图片 base64 进 OpenAI 兼容端点，可做 OCR/描述/视觉问答。

## 2. 环境搭建（vm-60）

1. venv：系统缺 `python3.10-venv`（ensurepip 不可用），沿用 Task 1 的做法用 `virtualenv --system-site-packages`；torch 2.9.1+cu128 从 `~/.local` 继承，**没有重复装 torch**。
2. `pip install "open-dataflow[pdf2vqa]"` → 装上 open-dataflow 1.0.11 + mineru[vlm] 2.6.8（torch/transformers 4.57.6/accelerate，**不需要 vllm/sglang**）。
3. **kernels 冲突（本项目特有坑）**：`~/.local` 里有 kernels 0.16.1（别的项目用的），它要求 huggingface-hub≥1.10，而 venv 里是 0.36.2；py3.10 上 `str | None`  dataclass 校验直接抛 `StrictDataclassFieldValidationError`，transformers 导入即崩。修法：venv 内装 `kernels==0.9.1`（正好是 transformers 4.57.6 自己的 `hub-kernels` extra 的 pin），影子覆盖 ~/.local 版本，不动别人的环境。
4. tiktoken：`ChunkedPromptedGenerator` 默认 `cl100k_base`，首次使用要从 `openaipublic.blob.core.windows.net` 下载 BPE——vm-60 不通。修法：本机走 Verge 代理下载后 scp 到 `~/df3_assets/tiktoken_cache`，设 `TIKTOKEN_CACHE_DIR` 指向它。
5. MinerU 模型：HF/modelscope 在 vm-60 上均不可达。**走本机 Clash Verge 代理（127.0.0.1:7897）用 python requests 下载**（curl 的 schannel 后端在沙箱里坏掉，报 SEC_E_NO_CREDENTIALS，换 OpenSSL 后端的 requests 即可），2.2GB `opendatalab/MinerU2.5-2509-1.2B` 下载后 scp 到 `~/df3_assets/`。
6. `~/mineru.json` 手写（`configure_model` 原本要从 jsdelivr 拉模板，也被墙）：`models-dir.vlm` 指向本地模型目录 + `model-source: local` + `MINERU_MODEL_SOURCE=local`。

## 3. MinerU 版式解析实测（公式/图表/版式）

后端选 `vlm-transformers`（纯 transformers 推理，显存占用低；Flash 版要 vllm，API 版要 key）。输入用官方示例 PDF：

| 输入 | 页数 | content_list 条目 | 类型分布 | 图片 |
|---|---|---|---|---|
| questionextract_test.pdf（扫描版数学讲义，例4~例31） | 4 | 40 | text 23 / header 7 / image 4 / page_number 4 / **equation 2** | 6（图3.13~3.16，带题注） |
| math_question.pdf + math_answer.pdf（2025 GDFZ 入学数学真卷，问答分离） | 2 | 64 | text 54 / **table 2** / image 5 / list 1 / header 2 | 7（第3题图/第5题图/第7题图…带题注） |

证据（`evidence/task3/`）：
- `mineru_math1.md` / `mineru_math2.md`：公式全部转成 LaTeX（`$\triangle ABC$`、`$x \triangle y = 5x + 2y$`）；成绩表识别为 HTML `<table>`；图注保留（"第3题图"）。
- `mineru_*_layout.pdf`：带 bbox 的版面可视化。
- 耗时：4 页约 52s（含 6.5s 模型加载，4090D）；2 页约 42s+65 框。

**注意**：这两份 PDF 的数字文本层是坏的（pypdf 抽出来是乱码/空白），纯 PyMuPDF 路线（Task 2 用法）拿不到内容——这正是 MinerU VLM 解析的价值所在。

## 4. Case 6 VQA 算子 + Case 7 全链路实测

### Case 6（PromptedVQAGenerator，本地 VLM）
`code/task3/vqa_case6.py`：3 张教材渲染页 → moondream:1.8b（Ollama `/v1` 端点）。
- OCR 提示词与 VQA 提问提示词各跑一遍，均正常返回（`evidence/task3/case6_ocr_results.json`）。
- moondream 能正确描述"中文数学教材、含几何图与公式、页码 86/58"，但**不做公式转写**——1.8B VLM 的能力边界，如实记录。
- 官方示例 `pic_path.json` 里是相对路径（`../example_data/...`），在 task3 工作目录下直接 FileNotFound；已改为绝对路径（`code/task3/pic_path.local.json`）。

### Case 7 全链路（本地改造版）
`code/task3/pdf_vqa_extract_pipeline.py`，相对官方的改动只有 4 处，算子链 itself 未动：
1. MinerU 用 `FileOrURLToMarkdownConverterLocal(mineru_backend="vlm-transformers", mineru_source="local", mineru_model_path=...)`；
2. **输出目录垫片**：mineru 2.6.8 把所有 vlm-* 后端的输出目录硬编码为 `vlm/`（`cli/common.py: parse_method="vlm"`），而算子按 `{backend}/` 拼路径——子类解析后建 `vlm-transformers -> vlm` 绝对路径符号链接，上游七个算子零修改；
3. LLM 换 `APILLMServing_request` → Ollama `http://localhost:11434/v1/chat/completions`（qwen3:1.7b，`max_tokens=8192`，`max_chunk_len=12000` 适配 40k 上下文）；
4. `vqa.py` 里 `from dataflow.operators.core_vision import PromptedVQAGenerator`（1.0.11 的 LazyLoader 不支持子模块路径导入）。

端到端跑通，`code/task3/verify_task3.py` **22/22 通过**（`evidence/task3/task3_verify.log`）：MinerU 产出、converted id 唯一、extracted 字段齐全、merged 配对、ShareGPT 的 messages 角色与 images 路径存在。

## 5. 本地 1.7B 抽取质量：结构有效、内容错误（如实记录）

教程原版用 gemini-2.5-pro 做 QA 抽取。换成本地 qwen3:1.7b 后：
- **不遵守"只输出 id"协议**：prompt 明令 `<question>`/`<solution>` 只填 json 条目的数字 id、禁止抄原文，且官方 prompt 自带的示例就是原文——模型照示例抄原文（`evidence/task3/llm_raw_math1.txt`）。加"只输出 id"的强调指令和正误对照示例后，章节标题能变成 id，question 仍然抄原文。
- **答案幻觉**：16 条 extracted 里 13 条答案是"Yes"；唯一像样的"3.14"来自题目里的"π取3.14"（模式匹配，非推理）。
- 解析器按设计把非数字字段丢弃 → question 多为空 → QA_Merger 后 math1 只剩 1 对、math2 1 对（`math*_merged_qa_pairs.md`）。
- 另跑一版**降级解析器**（`TASK3_PARSER_FALLBACK=1`，id 回填失败时保留原文）：内容能落下来（`fallback_math1_merged_qa_pairs.md`），但题文只落到标签（"例4"）、解答串到下一题——**分段能力不足**，且图片引用（`<pic>` 按 id 定位）丢失。

结论：**公式/图表/版式的"解析"这一环本地完全可行且质量好；"QA 抽取"这一环需要强 LLM**。1.7B 的输出只能证明链路连通，不能作为训练数据。这也是教程选 gemini-2.5-pro 的原因。

## 6. 复现

```bash
ssh vm-60
cd ~/dataflow-task3
source ./env.sh                      # PATH/DF_API_KEY/MINERU_*/TIKTOKEN_CACHE_DIR
python test_mineru_local.py ./example_data/PDF2VQAPipeline/questionextract_test.pdf intermediate_smoke   # 单文件冒烟
CASE6_PROMPT=ocr python vqa_case6.py                                                        # Case 6
python pdf_vqa_extract_pipeline.py                                                          # Case 7 全链路
python verify_task3.py                                                                      # 验收
# 降级解析器变体：TASK3_PARSER_FALLBACK=1 python pdf_vqa_extract_pipeline.py
```
模型与缓存不随仓库分发：`~/df3_assets/MinerU2.5-2509-1.2B`（2.2GB）、`~/df3_assets/tiktoken_cache`、Ollama 的 qwen3:1.7b 与 moondream:1.8b。

## 7. 材料索引（evidence/task3/）

| 文件 | 内容 |
|---|---|
| install_df3.log / verify_df3d.log | 安装与 import 验收（含 kernels 修复后全绿） |
| smoke_mineru.log | 4 页扫描 PDF 冒烟：rc=0，52.1s，40 items |
| mineru_math1/2_content_list.json + .md + _layout.pdf | 版式解析结构化产出（公式/表格/图题） |
| task3_pipeline.log / task3_verify.log | 全链路运行日志与 22/22 验收 |
| llm_raw_math1.txt / math*_extracted_vqa.jsonl | 1.7B 原始输出与解析后 QA（质量差的证据） |
| fallback_math1_merged_qa_pairs.md | 降级解析器输出 |
| qa_sharegpt.json | ShareGPT 格式最终产物（messages+images） |
| case6_ocr_results.json | moondream 对 3 页教材图的 VQA 结果 |
| prompt_exp.log / prompt_exp2.log | "只输出 id" 强调实验记录 |
| vqa_images/ | QA 引用到的教材插图 |
