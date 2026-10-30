# Task4 · 不同 AI 应用场景的数据需求：从下游接口反推数据契约

> 课程：DataFlow · AI-Ready Data 实战（10-dataflow，Task 4）
> 教程主仓：https://github.com/OpenDCAI/DataFlow ｜ 官方文档：https://opendcai.github.io/DataFlow-Doc/zh/
> 本文写作日期：2026-10-29 ｜ 代码基线：本地仓库 `DataFlow-main`（`dataflow/version.py` = **1.0.11**，与 vm-60 上 `open-dataflow 1.0.11` 一致）
> 本文只做「文档 + 源码」分析，不做端到端数据质量结论；实测部分单独标注。**不包含任何 benchmark 数字的复述或编造**。

## 0. 证据分级（先声明能说什么、不能说什么）

| 级别 | 来源 | 本文中的用法 |
|---|---|---|
| **【文档】** | 官方文档本地副本 [`agenticrag_pipeline.txt`](../evidence/task4/docs/agenticrag_pipeline.txt)（2025-07-14）、[`text2sqlpipeline.txt`](../evidence/task4/docs/text2sqlpipeline.txt)（2025-06-17） | 场景定位、流水线流程、字段说明的转述依据 |
| **【源码】** | `DataFlow-main` 1.0.11 的 pipeline / operator / prompt 文件，逐文件核对 | 字段真实来源、默认值、过滤逻辑、算子间 key 传递 |
| **【实测】** | 2026-10-29 本机 Python 3.10.2 探针（同一份源码逻辑在 DSH 内置 3.12.14 下亦得到相同结果）；原样 `exec` 仓库 F1 评分器源码，仅把 `tqdm`/`dataflow.*` 打成桩 | 只用于验证「评分器对中文/标点的行为」这一条可静态判定的结论 |
| **【经验】** | 公开常识性的数量级判断 | 仅出现在对比表的「典型规模」列，明确标注为非实测 |
| **不写** | 未运行的结果 | 不写任何「某 pipeline 产出 N 行、准确率 X%」的 Task4 数字；Task4 端到端实测由同组其他任务产出，见 `evidence/task4` |

判断一条数据「合不合格」，本质是回答：**下游模型吃什么形态 → 监督信号从哪来 → 信号能否被机器验证 → 多大规模、如何去重 → 需要带哪些元数据**。下面把这个框架用在 7 类场景上。

## 1. 通用判断框架（7 类场景共用）

| 维度 | 要回答的问题 | 对数据的具体影响 |
|---|---|---|
| ① 下游接口 | 训练/推理时模型看到的张量是什么？ | 决定「数据单元」：纯文本 doc / (instruction, output) / (prompt, chosen, rejected) / (query, docs, answer) / (question, schema, SQL) / (image, question, answer) |
| ② 监督信号 | 标签是「下一个 token」还是「人类答案」还是「可执行结果」？ | 决定标签成本与噪声上限；弱信号场景必须靠 LLM-judge/投票补强 |
| ③ 可验证性 | 答案能否被规则/执行器/检索证据判定？ | 决定规模上限：可判定 → 可放宽到万级、可合成；不可判定 → 千级、必须重人工 |
| ④ 规模与去重 | 去重是精确、n-gram 还是语义？ | 预训练级用 MinHash；QA/RAG 级用近重复 + 语义多样性；偏好对还要控长度偏差 |
| ⑤ 元数据 | 出问题能否追溯到来源？ | source/url、语言、db_id、schema 版本、生成模型版本、prompt 版本、检索器版本、难度分层、许可证 |

**数据契约（data contract）最小字段**：输入字段（模型可见）+ 监督字段（loss 用）+ 验证字段（过滤/评分用）+ 溯源字段（模型/prompt/来源版本）。DataFlow 的算子全部走显式 `input_key/output_key` + `FileStorage.step()` 逐步落盘，天然适合把「契约」固化成 jsonl schema。

## 2. 七类场景逐节拆解

### 2.1 预训练 / 继续预训练

- **数据单元**：文档级纯文本。仓库样例 `dataflow/example/GeneralTextPipeline/pt_input.jsonl`（10 行）：`{"raw_content": "Thermodynamics is a branch of physics ..."}`；其中有重复段落和一条德语样本，正是给过滤器准备的。
- **必需字段**：`raw_content`（或 token 化后的 id 序列）。**可选字段**：`id/source/url/lang/token_count/quality_score/dup_cluster`。
- **质量门槛**（源码默认值）：词数 20–100000、句子数 3–7500、平均词长 3–10、符号词比 <0.4、大写词比 <0.2、唯一词比 ≥0.1、字符数 ≥100、MinHash 5-gram/阈值 0.9 去重、watermark/HTML/emoji/多余空格清理。【源码】`statics/pipelines/cpu_pipelines/text_pt_filter.py`
- **可验证信号**：弱监督，没有「标签」；只能用过滤器保留率、去重率、语言比例、下游 benchmark 作代理。**不存在单条样本级的正确性判定**。
- **评测指标**：困惑度/下游任务平均分（模型侧）；数据侧看保留率、重复率、语言/领域分布。
- **DataFlow 对应**：`cpu_pipelines/text_pt_filter.py`（23 个算子：20 个 filter + 3 个 refiner）、`gpu_pipelines/text_pt_synthetic.py`（合成）、`operators/text_pt`（`MetaSampleEvaluator` 等元数据打分）。【文档】README 的 SlimPajama-627B → 100B token 子集示例（转述，不引用数字）。

### 2.2 SFT 指令微调

- **数据单元**：`{"instruction", "output"}`（可选 `input` 作为指令附加上下文）。样例 `dataflow/example/GeneralTextPipeline/sft_input.jsonl`（5 行）里 2 行是故意留的垃圾（`"tell me why"`、`"asdfjkl; asdfjkl;"`），说明 SFT 数据必须先过滤再训练。
- **必需字段**：`instruction`、`output`。**可选字段**：`input/system/language/source/quality_score/rm_score/alpagasus_score/category/difficulty/teacher_model`。
- **质量门槛**：指令完整可执行、输出与指令对齐、长度 20–1000 词（`WordNumberFilter`）、去重、LLM 质量分（Alpagasus 3–5 分区间）、奖励模型分（RM 默认保留 [0.2, 0.8]，太低=差数据，太高=太简单）、格式/拒答一致性。
- **可验证信号**：中弱——强模型打分、奖励模型、规则过滤；代码/数学子域可执行验证另算。
- **评测指标**：held-out 指令集胜率、领域 benchmark、长度/多样性分布；数据侧看 judge 均分与分数分布、过滤保留率。
- **DataFlow 对应**：`api_pipelines/text_sft_synthesis_pipeline.py`（`CondorGenerator` → `CondorRefiner` → `AlpagasusFilter(min_score=3,max_score=5)`）、`cpu_pipelines/text_sft_filter.py`、`gpu_pipelines/text_sft_synthetic.py`、`operators/text_sft`（`RMScorer`/`RMFilter`，模型 `OpenAssistant/reward-model-deberta-v3-large-v2`）。【源码】

### 2.3 偏好对齐与 RL（RLHF / DPO / RLVR）

- **数据单元**：两种。
  1. 偏好对：`{"prompt", "chosen", "rejected", "chosen_score", "rejected_score", "annotator"/"judge"}`；
  2. 可验证 RLVR：`{"prompt", "answer", "verifier"(执行器/答案匹配规则/证明检查)}`，奖励来自规则执行，不来自人。
- **必需字段**：`prompt`、`chosen`/`rejected` 或 `answer+verifier`；**可选字段**：难度、来源模型、长度、token 数、打分差 `margin`、过滤阶段分数。
- **质量门槛**：难度带（全对/全错都无梯度）、去重（同一 prompt 多种回答会稀释对比）、控长度偏差（DPO 爱学长度）、chosen/rejected 确实「同一分布、仅质量不同」、judge 校准。
- **可验证信号**：RLVR 强（执行器/规则）；偏好对弱（人/强模型），因此偏好数据规模通常远小于 SFT。
- **DataFlow 对应（诚实说明）**：【源码】全仓检索 `preference|dpo|rlvr|reward`，只在奖励模型过滤/评分出现，**DataFlow 1.0.11 没有成对偏好数据生成 pipeline**。可复用的能力拼图是：`operators/text_sft/eval/rm_sample_evaluator.py`（0–1 奖励分）、`operators/text_sft/filter/rm_filter.py`（`min_score=0.2, max_score=0.8`）、reasoning pipeline 的答案 judge 与难度分层、AgenticRAG 的多跳验证器/F1（可当 reward 信号，见 §2.5）。
- **数据侧评测**：pairwise 准确率、奖励模型分差分布、长度差分布。

### 2.4 固定 RAG（检索增强，非 agent 轨迹）

- **数据单元**：知识库条目 `{"doc_id","text","metadata"}` + 训练/评测样本 `{"query","gold_docs"/"gold_chunk_ids","answer",[evidence]}`。
- **链路与必需字段**：【文档】`kbcleaning_pipeline.py` 的 PDF/URL → Markdown → 512-token chunk → 清洗 → QA；【源码】`KBCChunkGenerator(split_method="token", chunk_size=512, tokenizer_name="Qwen/Qwen2.5-7B-Instruct")`、`KBCTextCleaner(lang="en")`、`Text2MultiHopQAGenerator(num_q=5)`。另一个入口 `text2qa_pipeline.py`：`KCenterGreedyFilter(num_samples=5)` 先选语义多样文本 → `Text2QAGenerator(input_question_num=3)` → `Text2QASampleEvaluator` 输出 question_quality / answer_alignment / answer_verifiability 三组 grade+feedback。
- **质量门槛**：chunk 自包含（不跨主题截断）、答案能在 chunk 内定位（grounding）、问题必须依赖该 chunk（去掉文档应无法回答或答错）、近重复 chunk 去重、出处可回链。
- **可验证信号**：抽取式可定位（span 匹配）、EM/F1、LLM judge、检索命中率；**注意**：RAG 数据的「答案对」不等于「检索库对」，两者要分别校验。
- **评测指标**：检索 Recall@k / nDCG、端到端 EM/F1/faithfulness；数据侧看 question 可答率、答案 grounding 率。
- **DataFlow 对应**：`api_pipelines/kbcleaning_pipeline.py`、`api_pipelines/text2qa_pipeline.py`、`operators/knowledge_cleaning`、`operators/core_text`（`Text2QAGenerator/Text2QASampleEvaluator/KCenterGreedyFilter`）。

### 2.5 Agentic RAG（多步检索 + 训练/奖励）

- **数据单元**：两种形态。
  1. 单跳原子任务（官方 `agentic_rag_pipeline.py`）：输入 `{"id","contents"}`，原子生成 10 个字段 + 第二步 `F1Score` 共 11 个（见 §3.1）；样例 `dataflow/example/AgenticRAGPipeline/eval_test_data.jsonl`（10 行，contents 为维基段落，含标题行、580–741 字符、尾部截断）。
  2. 多跳（`agentic_rag_multihop_core_pipeline.py`）：输入 `{"hop_1": {"question","answer","doc","final_question","final_answer","refined_answer"}}`，逐跳扩展出 `hop_2...hop_k`（样例 `multihop_pipeline_data.jsonl` 仅 1 行）。
- **单跳的数据配方（源码核对，这也是本任务的核心）**：
  1. `identifier + conclusion/R` 抽取 → 原子事实 `candidate_tasks_str`；
  2. 以「conclusion + R」生成问题 Q，`answer` = conclusion；
  3. `refined_answer` = 清洗后的规范答案；`optional_answer` = 多参考答案列表；
  4. `llm_answer` = **不给文档**直接回答，`llm_score` = LLM 判分（`answer_score`）；**过滤 `llm_score < 1`** → 保证「模型没有上下文答不出来」；
  5. `golden_doc_answer` = 给文档回答，`golden_doc_score` = 判分；**过滤 `golden_doc_score >= 1`** → 保证「文档里确实能答出来」；
  6. 最后每篇文档 `head(max_question=10)` 控量，并单独跑 `AgenticRAGQAF1SampleEvaluator` 产出 `F1Score`。
  这 6 步把「必须检索（非 shortcut）+ 检索可答（grounded）」写死成数据过滤条件，是本任务最值得复用的设计。
- **多跳验证（防 shortcut 泄漏）**：逐跳 `InferenceCheck/ComparisonCheck` → 对 `reasoning` 复答并用 `EssEqPrompt` 判分（`answer_score<1` 淘汰）→ **shortcut check**：去掉一份支持文档再回答，仍答对则判为 shortcut 泄漏并淘汰 → 最终多跳合并问题再判一次。这是仓库里唯一显式做「删文档反事实」校验的 pipeline。
- **规模/成本模型（按源码估算，非实测）**：每个候选任务先跑 6 次 LLM 调用（identifier、conclusion、question、clean、llm_answer、llm_score）；通过 `llm_score<1` 过滤的再加 golden_doc + golden_doc_score 2 次；最终保留集再加 optional_answer 1 次，即「淘汰样本 6 次、最终样本约 9 次」。因此单跳 pipeline 的瓶颈是 API 成本，不是算力。`max_per_task=10`、`max_question=10` 是仅有的两个限流旋钮（本组 Task4 改造版另设默认 3/2，见 `code/task4/agentic_rag/agentic_rag_task4.py`）。
- **可验证信号**：LLM judge（0/1 recall score）+ token-F1（有语言局限，见 §6）+ 多跳反事实验证。
- **评测指标**：端到端 RAG 的 EM/F1、检索命中、多跳准确率；数据侧看 `llm_score/golden_doc_score/F1Score` 分布与淘汰率。
- **DataFlow 对应**：`api_pipelines/agentic_rag_pipeline.py`（单跳 + F1）、`api_pipelines/agentic_rag_multihop_core_pipeline.py`（`FlashRAGServing` + `FormatStrPromptedGenerator` + `PandasOperator` + `GeneralFilter`）、`AgenticRAGDepthQAGenerator`（纵深）、`AgenticRAGWidthQAGenerator`（横向组合）。

### 2.6 Text2SQL（SFT + CoT + 执行验证）

- **数据单元**：`{"db_id","SQL"}`（+ 可选 `question/evidence`）＋**数据库文件本身**（SQLite：`db_id.sqlite/.db`，db_id=文件名；或 MySQL 连接）。样例 `pipeline_refine.jsonl` 60 行（3 个数据库各 20 行：california_schools / student_club / thrombosis_prediction）；生成模式用 `empty.jsonl`（0 行，仅作 FileStorage 占位）。
- **必需字段**：`db_id`、`SQL`；问题可选（无则由 `Text2SQLQuestionGenerator` 生成）。**生成模式**完全不需要 SQL 样本，只要数据库目录。
- **质量门槛（refine 顺序）**：`SQLExecutabilityFilter`（只留 SELECT/WITH + `EXPLAIN` 生成查询计划，不真正执行）→ `SQLVariationGenerator`（数据替换/函数变换/难度变换）→ 再过滤 → 问题生成（`question_candidates_num=3`，多候选 embedding 选余弦中心）→ `Text2SQLCorrespondenceFilter`（问题-SQL-Schema 一致性，LLM 判 yes/no）→ `Text2SQLPromptGenerator`（零 LLM 调用）→ CoT 生成多轨迹 → 执行一致性投票 → 组件难度 + 执行难度分级。
- **可验证信号**：强——SQL 能否被数据库执行/EXPLAIN；结果集比对；CoT 轨迹可投票。
- **评测指标**：Execution Accuracy（执行结果相等）/ Exact Match、按难度分桶通过率；数据侧看可执行率、一致率、投票有效率、难度分布。
- **失败模式**：gold SQL 本身错/超时；SQLite 与 MySQL 方言差异；问题里泄漏 SQL 关键词；CoT prompt 会把参考答案塞给模型并明令「不要提到参考解」，但模型仍可能照抄（【源码】`Text2SQLCotGeneratorPrompt`）；投票在全部候选执行失败时**回退到第一条 CoT**（不淘汰），可能把不可验证轨迹放进训练集。
- **DataFlow 对应**：`api_pipelines/text2sql_pipeline_gen.py`（合成）、`api_pipelines/text2sql_pipeline_refine.py`（优化）、`statics/pipelines/cpu_pipelines/text2sql_pipeline.py`（无 LLM：执行过滤 → prompt → 组件难度）、`api_pipelines/text2vecsql_pipeline_gen.py`。

### 2.7 多模态 VQA（图表/公式/版式）

- **数据单元**：ShareGPT 风格 `{"messages":[{role,content}...],"images":[...]}`；链路中间态是 PDF 页 → MinerU Markdown/版面 → 逐题抽取 → 合并 QA。样例入口 `PDF2VQAPipeline/vqa_extract_test.jsonl`（`input_pdf_paths/name`）。
- **必需字段**：图片路径（存在且与消息中引用一致）、`messages` 角色配对、问题/答案；**可选字段**：题号、页码、bbox、图表类型、来源 PDF。
- **质量门槛**：图-问-答三元对齐、答案可在图中定位、跨题/跨页不串、公式用 LaTeX、表格保留结构、图片去重（同一插图多处引用）。
- **可验证信号**：弱——VLM judge/人工；规则只能查结构（id 回填、路径存在、角色交替）。【任务3实测】本地 1.7B 模型抽取「结构有效、内容错误」：16 条里 13 条答案答 Yes，图片引用按 id 定位失败；这说明多模态 VQA 的瓶颈在抽取模型能力，不在链路。
- **评测指标**：VQA 准确率/答案相似度、图表/公式子集准确率、抽取结构验收率；数据侧看 QA 对数、图引用完整率。
- **DataFlow 对应**：`api_pipelines/pdf_vqa_extract_pipeline.py` 七算子（`PDF_Merger` → `FileOrURLToMarkdownConverterFlash/API` → `MinerU2LLMInputOperator` → `ChunkedPromptedGenerator(QAExtractPrompt)` → `LLMOutputParser` → `QA_Merger` → `VQAFormatter`）；Case 6 `PromptedVQAGenerator` + `APIVLMServing_openai`。更完整的多模态生态在 DataFlow-MM（README 提及，本仓库不含）。

## 3. 字段映射表

### 3.1 AgenticRAG：输入 `contents` → 输出字段（单跳官方 pipeline）

| 字段 | 产生位置 | 类型/取值 | 含义 | 源码核对要点 |
|---|---|---|---|---|
| `contents` | 输入 | str | 原始文档文本（样例还带 `id`） | pipeline 显式传 `input_key="contents"`；算子 `run()` 默认其实是 `"prompts"` |
| `identifier` | AtomicTaskGenerator 第 1 步 | str（LLM 原始 JSON 文本） | 文档内容标识（如实体名），用于生成问题时给「数据出处」暗示 | 只做了字符串清洗，未反序列化成对象 |
| `candidate_tasks_str` | AtomicTaskGenerator 第 2 步 | JSON str，`{"conclusion","R"}` | 可独立成题的原子事实 + 关系/主体描述 | 每个候选任务展开成一行；`max_per_task` 截断 |
| `question` | 第 3 步 | str | 由 identifier+conclusion+R 生成的问题 | 要求「答案唯一、可由问题推出」 |
| `answer` | 第 3 步 | str | = conclusion（初始答案） | 未清洗 |
| `refined_answer` | 第 4 步 | str | 清洗后的规范答案 | 供 F1 与训练用 |
| `llm_answer` | 第 5 步 | str | **无上下文**直接回答 | 与 refined_answer 一起送 LLM 判分 |
| `golden_doc_answer` | 第 7 步 | str | 给原始文档后的回答 | 与 refined_answer 再判分 |
| `llm_score` | 第 6 步 | int（LLM `answer_score`） | 无上下文答案与 refined 的一致性 | 过滤条件 `< 1` 才保留（模型答不出=需要检索） |
| `golden_doc_score` | 第 8 步 | int（LLM `answer_score`） | 给文档后的答案与 refined 的一致性 | 过滤条件 `>= 1` 才保留（文档可答） |
| `optional_answer` | 第 9 步 | list[str] | 多参考答案（LLM 输出 JSON 解析失败则回退原文） | 供 F1 多参考答案匹配 |
| `F1Score` | **第二步独立算子** | float [0,1] | `refined_answer` vs `golden_doc_answer` 的 token-F1 | 由 `AgenticRAGQAF1SampleEvaluator` 产生，不是原子生成器 |

说明（【源码】`agenticrag_atomic_task_generator.py`）：原子生成器内部还有一个 `get_f1_score()`（prediction=golden_doc_answer，ground_truths=optional_answer），但 `run()` 从未调用它——属于死代码；真正落盘的 F1 来自 pipeline 第二步。此外 `llm_score/golden_doc_score` 的 0/1 判定由 prompt 返回的 `answer_score` 决定，不是 token 相似度。

### 3.2 Text2SQL：输入 → 输出（gen vs refine）

| 字段 | gen（合成） | refine（优化） | 来源算子 | 说明 |
|---|---|---|---|---|
| `db_id` | ✅ | ✅ | 输入/SQLGenerator/SQLVariationGenerator | 数据库文件名 |
| `SQL` | ✅（LLM 生成） | ✅（输入 + 变体） | SQLGenerator / SQLVariationGenerator | gen 的 SQL 由 schema 合成 |
| `question` | ✅ | ✅（重新生成） | Text2SQLQuestionGenerator | refine 输入里的 question 会被置空后重生成（见 §7-③） |
| `question_type` | ✅ | ✅ | Text2SQLQuestionGeneratorPrompt | 8 种风格随机：Formal/Colloquial/Imperative/Interrogative/Descriptive/Concise/Vague/Metaphorical；Vague/Metaphorical 会要求外部知识 |
| `evidence` | ✅ | ✅ | Text2SQLQuestionGenerator | 外部知识/问题背景，来自 `[EXTERNAL-KNOWLEDGE-START]` 块，可空 |
| `prompt` | ✅ | ✅ | Text2SQLPromptGenerator | 训练用 prompt：问题 + schema + 提示；零 LLM 调用 |
| `cot_responses` | ✅（中间） | ✅（中间） | Text2SQLCoTGenerator | list[str]，`sampling_num` 条候选推理轨迹；投票后删除 |
| `cot_reasoning` | ✅ | ✅ | Text2SQLCoTVotingGenerator | 执行一致性投票选出的最终 CoT（无有效候选时回退第一条） |
| `sql_component_difficulty` | ✅ | ✅ | SQLComponentClassifier | 规则打分（子查询 +2、JOIN 数、IN/EXISTS、GROUP BY/HAVING、函数、ORDER/LIMIT、集合运算 +2、多列…），阈值 [2,4,6] → easy/medium/hard/extra |
| `sql_execution_difficulty` | ✅ | ✅ | SQLExecutionClassifier | 同一 prompt 生成 `num_generations` 条 SQL，与 gold SQL 执行结果比对得 `cnt_true`；阈值 [2,5,9] → extra/hard/medium/easy（**注意 labels 反向**） |
| `sql_complexity_type` | ✅（仅 gen） | ❌ | SQLGenerator | Simple/Moderate/Complex/Highly Complex 随机目标级别 |
| `sql_variation_type` | ❌ | ✅（仅 refine） | SQLVariationGenerator | 变体类型标签；该算子会用变体行**替换**整张表，只填 db_id/SQL/variation_type |

gen 与 refine 的流程差异（【源码】两个 pipeline 文件）：gen 第 2 步就是 `SQLExecutabilityFilter`，随后直接问题生成；refine 在过滤与变体生成之间多一次可执行性过滤，且 `SQLVariationGenerator` 之后原有字段全部置空。两者都不跑 `SQLExecutionFilter`（真正执行 SQL 的过滤器），执行只发生在 CoT 投票和 `SQLExecutionClassifier` 内部。

## 4. 跨场景对比

| 场景 | 数据单元 | 监督信号 | 环境依赖 | 可验证性 | 典型规模（经验量级，非实测） | 主要失败模式 |
|---|---|---|---|---|---|---|
| 预训练/继续预训练 | doc 级纯文本 | 自监督（下一 token） | 无（离线 tokenizer） | 弱 | 1e9–1e12 token | 重复/模板/低质网页、语言混杂 |
| SFT 指令微调 | (instruction, output) | 人类/强模型示范 | 无（离线） | 中弱（judge） | 1e4–1e6 条 | 指令歧义、答案冗长/不忠实、垃圾指令未过滤 |
| 偏好对齐/RL | (prompt, chosen, rejected) 或 (prompt, verifier) | 人类偏好 / 可执行验证 | RLVR 需执行器/环境 | 偏好弱、RLVR 强 | 1e3–1e5 对 | 长度偏差、难度全对/全错、judge 偏见 |
| 固定 RAG | (query, gold docs, answer) | 文档证据 | 检索库 + embedding | 中（可定位性） | 1e3–1e5 条 | chunk 截断、答案跨段、问题泄漏答案 |
| Agentic RAG | (contents → QA + 双答案 + 分数) / 多跳轨迹 | LLM judge 0/1 + token-F1 + 反事实验证 | 检索器（多跳需 FlashRAG/faiss） | 中（LLM 判分，非硬验证） | 1e3–1e4 条 | shortcut 泄漏（不检索也能答）、grounding 幻觉、成本高 |
| Text2SQL | (question, schema, SQL[+CoT]) | 数据库执行结果 | **数据库实例**（SQLite/MySQL） | **强** | 1e3–1e5 条 | gold SQL 错、方言差异、问题泄漏 SQL、无效 CoT 回退 |
| 多模态 VQA | (image(s), messages) | 人工/强 VLM 标注 | 解析器（MinerU）+ VLM | 弱（结构可验、语义难验） | 1e4–1e5 条 | 图-问-答错配、跨题串行、图片引用丢失、1.7B 模型幻觉 |

一个横向结论：**可验证性越高，越敢合成、越敢上规模；可验证性越低，越依赖强模型/人工与更严的过滤**。Text2SQL（执行验证）› RLVR（规则验证）› 固定 RAG（证据定位）› AgenticRAG（LLM judge + 反事实）› VQA（VLM judge）≈ 偏好对（人）> 预训练（无标签）。

## 5. 共同数据要求 × DataFlow 算子映射

| 共同要求 | 具体做法 | DataFlow 算子 / 文件（1.0.11 源码） |
|---|---|---|
| 去重（文档级） | MinHash 5-gram，num_perm=128，阈值 0.9 | `MinHashDeduplicateFilter`（`general_text`） |
| 去重/多样性（QA 级） | 语义 K 中心贪心选样 | `KCenterGreedyFilter(num_samples=5)`（`core_text`） |
| 长度/格式过滤 | 词数、句子数、平均词长、字符数、符号比、大写比、无标点、样板文字 | `WordNumberFilter / SentenceNumberFilter / CharNumberFilter / MeanWordLengthFilter / SymbolWordRatioFilter / NoPuncFilter / LoremIpsumFilter` 等 |
| 文本清洗 | 去 emoji、去 HTML/URL、去多余空格、去特殊字符 | `RemoveEmojiRefiner / HtmlUrlRemoverRefiner / RemoveExtraSpacesRefiner / SpecialCharacterFilter` |
| 质量打分/过滤（SFT） | Alpagasus 3–5 分；RM 0.2–0.8 分 | `AlpagasusFilter`、`RMScorer/RMFilter`（OpenAssistant reward model） |
| QA 质量评估（RAG） | 问题质量、答案对齐、答案可验证性三组 grade + feedback | `Text2QASampleEvaluator`（`core_text`） |
| CoT 生成 + 多样性过滤 | 生成多条 CoT，n-gram 重叠 0.1–1.0 过滤 | `ReasoningAnswerGenerator` + `ReasoningAnswerNgramFilter` |
| CoT 投票（可执行域） | 从 CoT 抽 SQL → 批量执行 → 按结果集签名分组投票 → 平票时默认最短 SQL（部分分支随机） | `Text2SQLCoTGenerator` + `Text2SQLCoTVotingGenerator`（`base.batch_execute_queries`） |
| 难度分层 | 数学/general 问题难度分类；SQL 组件难度（规则）；SQL 执行难度（多次生成通过数） | `ReasoningQuestionDifficultySampleEvaluator`、`SQLComponentClassifier`、`SQLExecutionClassifier` |
| 执行过滤 | 真执行 vs EXPLAIN 查询计划 | `SQLExecutionFilter`（执行）、`SQLExecutabilityFilter`（EXPLAIN，不执行） |
| 答案/奖励评估 | token-F1、多参考答案 max-F1、0/1 LLM recall score | `AgenticRAGQAF1SampleEvaluator`、`AgenticRAGAtomicTaskGenerator` 内部 recall score prompts |
| 多跳反事实/防捷径 | 删支持文档后仍答对 → 淘汰；逐跳复答一致性 | `agentic_rag_multihop_core_pipeline.py` 的 `EssEqPrompt` + `SingleHopPrompt` |
| 流水线可追溯 | 每步 `FileStorage.step()` 落盘 jsonl，算子显式 input/output key，prompt 可替换 | `FileStorage`、`OperatorABC`、`prompt_restrict`、`DIYPromptABC` |

## 6. 本任务实测：token-F1 评分器在中文/标点上的行为

方法（可复现）：读取仓库原文件 `dataflow/operators/agentic_rag/eval/agenticrag_qaf1_sample_evaluator.py`，原样 `exec`，只把 `tqdm`/`dataflow.*` 依赖替换为桩，然后直接调用 `compute_f1`。**这不是端到端跑 pipeline**，只验证评分函数本身。Lead 已用本机 Python 3.10.2 独立复跑该探针，6 个数值与本表完全一致（2026-10-29）；该结果与解释器版本无关。

| 输入（prediction vs ground truth） | F1 | 原因 |
|---|---|---|
| EN：`University of Nebraska-Lincoln` vs `University of Nebraska–Lincoln` | **0.6667** | 只删 ASCII 标点；en-dash（–）保留 → `nebraska–lincoln` 与 `nebraskalincoln` 不匹配 |
| EN：`Lincoln, Nebraska` vs `University of Nebraska–Lincoln` | 0.0 | 无共同 token（`nebraska` ≠ `nebraska–lincoln`） |
| ZH：`内布拉斯加大学林肯分校` vs 同串 | 1.0 | 完全一致 |
| ZH：同串 vs `位于林肯市的内布拉斯加大学` | **0.0** | 中文无空格 → 整句=1 个 token，无法部分匹配 |
| ZH：`林肯 分校` vs `内布拉斯加大学林肯分校` | 0.0 | 不做子串匹配 |
| ZH：`准确率为 59.0%` vs `59.0%` | 0.6667 | 空格切词后 2 token 对 1 token，precision 0.5/recall 1 |

结论（对数据需求的启示）：
1. `AgenticRAGQAF1SampleEvaluator` 的 F1 是**空格切词 token 袋模型 + 仅 ASCII 标点归一化 + 英文冠词删除**。对英文短答案可用；对**中文答案近似 exact-match**（除非答案内部有空格），用作 RL 奖励会把「语义正确但用词不同」的答案判 0。
2. 若要拿它当中文 RAG 的 reward，必须换评分口径（中文分词/字符级 F1、embedding 余弦、或 LLM judge），否则很容易把 F1 分布压向 0/1 两极，奖励信号几乎没有梯度。
3. 数据元数据里应记录**评分器版本**（本仓库为该文件 1.0.11 版本）；换评分口径等于换奖励函数，跨实验不可直接比较。

## 7. 文档 vs 源码：发现的不一致（如实记录）

| # | 文档说法 | 源码实际 | 影响 |
|---|---|---|---|
| ① | CoT 生成器「输出 cot_reasoning」，pipeline 也传 `output_cot_key="cot_reasoning"` | `Text2SQLCoTGenerator.run()` 固定写 `cot_responses`，**完全忽略 `output_cot_key`**；投票器才写 `cot_reasoning` | 功能正确（投票器读 `cot_responses`），但参数是误导性的；自定义输出列名会失效 |
| ② | 文档 §3.1.1 把 `SQLExecutionFilter` 列为流水线过滤器 | 官方 gen/refine API pipeline 用的是 `SQLExecutabilityFilter`（EXPLAIN 不执行）；`SQLExecutionFilter` 只出现在 `cpu_pipelines/text2sql_pipeline.py` 与 `text2vecsql_pipeline_gen.py` | 「执行过滤」有两种语义，混用会多花数据库执行时间 |
| ③ | refine 模式「对已有数据进行筛选、扩充和增强」，输入 `db_id`+SQL（question 可选保留） | `SQLVariationGenerator` 用 `{col: None}` 重建整行，原 question/evidence 丢失，随后问题生成器重新生成 | 输入里的 question 实际不会被复用；想保留必须改算子 |
| ④ | AgenticRAG 文档把 F1Score 列在原子生成器产出里 | `F1Score` 由独立的第二步 evaluator 产生；原子生成器内部 `get_f1_score()` 是死代码（`run()` 不调用） | 少跑第二步就没有 F1Score；内部 F1 的语义（vs optional_answer）与落盘 F1（vs golden_doc_answer）不同 |
| ⑤ | `SQLExecutionClassifier(num_generations=10)` 可自由调小以省时间 | `num_generations <= thresholds[-1]` 时会被强制改为 `((last//5)+1)*5`；默认阈值 9 → 至少 10 次 | 想降到 5 次必须同时改 `difficulty_thresholds`，否则配置无效 |
| ⑥ | SQL 变体生成器「输出：SQL 变体集合」 | 实际是**替换** dataframe 为变体行（只填 3 列），不是追加 | 下游字段全部重新生成；行数从 N 变成 N×num_variations（若成功） |

另：AgenticRAG 文档示例输入字段叫 `contents`，而 `AgenticRAGAtomicTaskGenerator.run()` 默认 `input_key="prompts"`；官方 pipeline 显式传 `contents` 所以能跑，但直接按算子默认值调用会 KeyError。

## 8. 可复现命令与边界

- 读文档副本：`.dsh/tmp/docs/agenticrag_pipeline.txt`、`.dsh/tmp/docs/text2sqlpipeline.txt`（与 `evidence/task4/docs/` 同源）。
- 核对版本：`read DataFlow-main/dataflow/version.py` → `__version__ = '1.0.11'`。
- 核对算子：`DataFlow-main/dataflow/operators/agentic_rag/generate/agenticrag_atomic_task_generator.py`、`.../eval/agenticrag_qaf1_sample_evaluator.py`、`DataFlow-main/dataflow/operators/text2sql/generate/text2sql_cot_generator.py`、`text2sql_cot_voting_generator.py`、`eval/sql_execution_classifier.py`、`eval/sql_component_classifier.py`。
- 复现 §6 探针：用本机 Python 读取上述 F1 评分器源码 → 打桩 `tqdm/dataflow` → 调用 `compute_f1`；不联网、不调 LLM。
- **边界**：本文没有调用任何 LLM、没有在 vm-60 跑 DataFlow pipeline；Task4 的 AgenticRAG/Text2SQL 端到端实测与产物由同组其他任务负责（`code/task4`、`evidence/task4`）。§2.7 引用的 1.7B 抽取结论来自任务 3 实测（`notes/task03-多模态数据公式图表版式VQA.md`），不是本任务新结论。§6 是评分函数的单元级探针，不代表整条 pipeline 的 F1 质量。本文没有编造任何模型准确率/产出规模数字；出现的行数均为仓库样例文件的真实行数（10/5/60/1/0 行），§6 的 F1 为本机探针结果，均与 Task4 真实运行结果分开。

## 9. 参考

- DataFlow 官方文档（AgenticRAG 数据合成）：https://opendcai.github.io/DataFlow-Doc/zh/guide/agenticrag_pipeline/
- DataFlow 官方文档（Text-to-SQL 数据合成）：https://opendcai.github.io/DataFlow-Doc/zh/guide/text2sqlpipeline/
- DataFlow 仓库：https://github.com/OpenDCAI/DataFlow （本文代码基线 1.0.11）
- 本组证据：[`evidence/task4/docs/agenticrag_pipeline.txt`](../evidence/task4/docs/agenticrag_pipeline.txt)、[`evidence/task4/docs/text2sqlpipeline.txt`](../evidence/task4/docs/text2sqlpipeline.txt)
- 任务3 多模态实测笔记：[`notes/task03-多模态数据公式图表版式VQA.md`](task03-多模态数据公式图表版式VQA.md)
