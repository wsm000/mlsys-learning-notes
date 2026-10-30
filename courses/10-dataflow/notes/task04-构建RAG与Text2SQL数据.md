# Task4 · 构建 RAG 与 Text2SQL 数据（AgenticRAG 数据合成 / Text2SQL Pipeline）

> 教程主仓：https://github.com/OpenDCAI/DataFlow
> 图文教程（飞书）：https://wcny4qa9krto.feishu.cn/wiki/I9tbw2qnBi0lEakmmAGclTysnFd
> 官方文档：[AgenticRAG 数据合成流水线](https://opendcai.github.io/DataFlow-Doc/zh/guide/agenticrag_pipeline/)、[Text-to-SQL 数据合成流水线](https://opendcai.github.io/DataFlow-Doc/zh/guide/text2sqlpipeline/)
> 环境：vm-60（Linux x86_64，RTX 4090 D 24GB）· open-dataflow 1.0.11 · Python 3.10.12
> 模型：AMD Radeon 兼容端点 DeepSeek-V4-Flash（`https://developer.amd.com.cn/radeon/api/v1`）· 本地 embedding：BAAI/bge-small-zh-v1.5
> 本任务产出：AgenticRAG 数据合成流水线实测 · Text2SQL 数据流水线实测 · [不同 AI 应用场景数据需求分析](task04-不同AI应用场景数据需求.md) · [证据](../evidence/task4) · [打卡](../evidence/task4/dataflow-task4-checkin.md)

## 0. 环境与证据边界（先说能说什么、不能说什么）

| 项 | 实际情况 |
|---|---|
| 机器 | vm-60：Linux x86_64，NVIDIA RTX 4090 D（24GB），Python 3.10.12 |
| DataFlow | open-dataflow 1.0.11（`/home/simin/.venv-df3`，含 torch 2.9.1+cu128 / transformers 4.57.6 / sentence-transformers / faiss） |
| 生成模型 | DeepSeek-V4-Flash，经 AMD Radeon 的 OpenAI 兼容端点调用；实测返回标准 `choices[0].message.content`，无 reasoning_content，可直接用于 DataFlow `APILLMServing_request` |
| 向量模型 | BAAI/bge-small-zh-v1.5，本地 HF 缓存 + `LocalEmbeddingServing`（Text2SQL 问题候选去重/选优用） |
| 数据库 | 官方示例库 `Open-Dataflow/dataflow-Text2SQL-database-example`（databases.zip 6.5MB，解压 21MB）：california_schools / student_club / thrombosis_prediction 三个 SQLite 库 |
| 网络边界 | vm-60 当前可访问 AMD Radeon API 与 HuggingFace（部分镜像）；示例库由本机代理下载后 scp 上台，未依赖远端 HF 直连 |
| 本任务**不报告** | 任何"训完模型后的效果提升"数字；只报告数据流水线本身的可运行性、产出结构与质量边界 |

**为什么这类数据值得单独做**：RAG/Text2SQL 是"数据决定上限"的典型场景——RAG 训练数据要区分"有上下文/无上下文"的答案，Text2SQL 数据必须能被数据库执行且问题-SQL-Schema 三者一致。DataFlow 把这两类数据的构造过程拆成可复用算子，本任务逐条跑通并检查产物。

## 1. AgenticRAG 数据合成（官方 pipeline）

官方文档与代码：`dataflow/statics/pipelines/api_pipelines/agentic_rag_pipeline.py`。输入只需要一段**原始文本上下文**（`contents` 字段），流水线自动生成"可验证的原子问答对"并打分。

```
eval_test_data.jsonl (id, contents)
  → AgenticRAGAtomicTaskGenerator   # 单条文本上跑 8 个 prompt 阶段
  → AgenticRAGQAF1SampleEvaluator   # refined_answer vs golden_doc_answer 的 F1
```

`AgenticRAGAtomicTaskGenerator` 内部不是一次 LLM 调用，而是一条**小流水线**（每个阶段一次批量 LLM 调用）：

| 阶段 | prompt 模板 | 作用 |
|---|---|---|
| get_identifier | AtomicTaskGeneratorGetIdentifierPrompt | 从原文提取内容标识符 identifier |
| get_conclusion | AtomicTaskGeneratorGetConlcusionPrompt | 提取候选任务/结论 `candidate_tasks_str`（JSON：conclusion + R） |
| init_question | AtomicTaskGeneratorQuestionPrompt | 由 identifier + 结论生成初始问题 |
| clean_qa | AtomicTaskGeneratorCleanQAPrompt | 清理问题与原始答案 → answer |
| llm_answer | AtomicTaskGeneratorAnswerPrompt | **无上下文**让 LLM 直接答 → llm_answer（用于区分"模型本来就会"与"需要文档"） |
| golden_doc_answer | AtomicTaskGeneratorGoldenDocAnswerPrompt | 只看原文+问题回答 → golden_doc_answer（黄金文档答案） |
| get_recall_score ×2 | AtomicTaskGeneratorRecallScorePrompt | refined_answer 与 llm_answer / golden_doc_answer 的召回分 → llm_score / golden_doc_score |
| more_optional_answer | AtomicTaskGeneratorOptionalAnswerPrompt | 生成可接受备选答案 → optional_answer |

输出字段与文档一致：question / answer / refined_answer / optional_answer / llm_answer / golden_doc_answer / identifier / candidate_tasks_str / llm_score / golden_doc_score，第二步再补 **F1Score**。这套字段的设计意图很清楚：`llm_answer`（无上下文）与 `golden_doc_answer`（有上下文）的差，是 Agentic RAG 训练需要的"检索收益"信号；F1Score 可作为 RL 奖励的一部分。

## 2. Text2SQL 数据流水线（官方 refine / gen 两条）

官方文档：Text-to-SQL 数据合成流水线支持两种模式：

- **数据优化模式（refine）**：输入"数据库文件 + 已有 SQL 样本（db_id + SQL，可选 question/evidence）"，做过滤、变体扩充、问题生成、Prompt/CoT 构建、难度分级；
- **数据合成模式（gen）**：只给数据库文件，从 Schema 从零生成 SQL，再走后续同样的生成链。

两条链共用的关键算子（1.0.11）：

| 类别 | 算子 | 输入 → 输出 | 可验证性 |
|---|---|---|---|
| 过滤 | SQLExecutabilityFilter | SQL+db_id → 能生成查询计划的行 | 数据库 EXPLAIN，不执行即可判定语法/引用错误 |
| 过滤 | SQLExecutionFilter | SQL+db_id → 能真正执行的行 | 真实执行，抓运行时错误 |
| 过滤 | Text2SQLCorrespondenceFilter | question+evidence+SQL+db_id → 一致的子集 | LLM 判问题-SQL-Schema 一致性 |
| 生成 | SQLVariationGenerator（refine） | SQL → 语义等价变体 + sql_variation_type | 变体仍需过可执行性过滤 |
| 生成 | SQLGenerator（gen） | schema → SQL + sql_complexity_type | 生成后过滤 |
| 生成 | Text2SQLQuestionGenerator | SQL+schema → question/evidence（多候选 + embedding 相似度选优） | 逆问题生成，多候选降方差 |
| 构建 | Text2SQLPromptGenerator | question+evidence+db_id → 训练 prompt（含 schema + 指令） | 模板确定性 |
| 构建 | Text2SQLCoTGenerator | SQL+question+db_id → 多条 CoT 候选 cot_responses | 候选后续投票 |
| 构建 | Text2SQLCoTVotingGenerator | cot_responses → 按"执行结果一致"投票选 cot_reasoning | 用数据库执行一致性投票，而非 LLM 自评 |
| 分级 | SQLComponentClassifier | SQL → sql_component_difficulty（组件复杂度 easy/medium/hard/extra） | 规则化 |
| 分级 | SQLExecutionClassifier | SQL+prompt+db_id → sql_execution_difficulty | 多次生成后执行通过率 |

最终字段（官方文档）：db_id、question、question_type、evidence、SQL、sql_variation_type / sql_complexity_type、prompt、cot_reasoning、sql_component_difficulty、sql_execution_difficulty。

设计要点：Text2SQL 数据的每个质量维度几乎都能**外部验证**——执行过滤器验证可执行性、双向生成（SQL→问题、问题+Schema→SQL）验证一致性、CoT 投票验证推理链与执行结果是否一致、难度分级让训练数据可配比。这正是"可验证信号"在数据合成里的落地。

## 3. 不同 AI 应用场景的数据需求

七类场景（预训练 / SFT / 偏好对齐与 RL / 固定 RAG / Agentic RAG / Text2SQL / 多模态 VQA）的数据单元、字段、质量门槛、可验证信号与 DataFlow 对应 pipeline 的对照，见独立笔记：**[不同 AI 应用场景数据需求分析](task04-不同AI应用场景数据需求.md)**。

一句话版本：**数据需求 = 下游训练/推理接口（字段）+ 可验证信号（怎么判对错）+ 规模/去重/难度配比**。RAG 类数据的关键是"有证据才答得对"（grounding + 无上下文对照），Text2SQL 类数据的关键是"能执行且问题与 SQL 一致"（execution + consistency），Agent 类数据的关键是"过程可验证"（工具轨迹、状态变化、最终答案）。

## 4. 实测结果

> 由 vm-60 真实运行产出，逐步数字与原始 jsonl 见 [evidence/task4](../evidence/task4)；本节在运行完成后回填。

## 4. 实测结果

> 全部数字来自 vm-60 真实运行，原始日志与逐步 jsonl 见 [evidence/task4](../evidence/task4)。2026-10-29 执行，模型为 AMD Radeon 兼容端点的 DeepSeek-V4-Flash / GLM-5.3-Flash。

### 4.1 AgenticRAG 数据合成（4 条文档 → 4 条 QA）

配置：官方 `eval_test_data.jsonl` 前 4 条 contents（内容为 Halloween 4、John Scale、The Little Vagabond、Trina Radke 四段维基文本）；`max_per_task=2`、`max_question=1`、`max_workers=2`、进程级 RPM≈12、`max_retries=6`（改造脚本见 [code/task4/agentic_rag](../code/task4/agentic_rag)）。

| 指标 | 实测值 |
|---|---|
| 输入 / 最终样本 | 4 → 4。内部漏斗：4 文档 → 8 个候选任务（max_per_task=2）→ 8 条 QA → 7 条通过 `llm_score<1` 过滤 → 每文档按 `max_question=1` 取 1 题 → 最终 4 行 |
| step1 输出 | 4 行 × 11 个生成字段（identifier / candidate_tasks_str / question / answer / refined_answer / llm_answer / llm_score / golden_doc_answer / golden_doc_score / optional_answer + id） |
| step2 输出 | 4 行，追加 `F1Score`；13 个字段全部非空 |
| 成本 | 66 次 chat 请求（53 次逻辑调用）、654.7 s（≈ 6.1 请求/分钟——20 RPM 用户级限流下的真实吞吐，不是理论并发）。正式跑 3 次 429 + 2 次 provider 504，全部被重试吸收、0 行丢失 |
| F1Score | [1.0, 1.0, 0.7826, 0.0]，均值 **0.696** |
| 双过滤 | `llm_score` 全为 0（无上下文答不出 → 必须检索）；`golden_doc_score` ∈ {1, 2}（给文档可答 → grounded）。4/4 通过 |

**一个真实失败模式**：F1=0 的那条是 `refined_answer=2017-08-30` vs `golden_doc_answer=August 30, 2017`——语义完全一致，但 token-F1 因为日期格式不同判 0；F1=0.7826 的条目是精炼句 vs 更长原文句的部分重叠。这与数据需求分析笔记 §6 的评分器探针结论一致：**token-F1 对格式/日期/多语言敏感，直接当 RL 奖励会引入噪声**，训练前要做答案规范化或换评分口径。

### 4.2 Text2SQL 数据流水线（refine：4 条输入 → 3 条保留）

配置：`pipeline_refine.jsonl` 中 california_schools 的前 4 条（db_id + question + SQL）；`num_variations=1`、`question_candidates=2`、`cot_sampling=1`、`exec_generations=3`、`max_workers=2`、`temperature=0.7`。数据库为本地解压的 SQLite 示例库（3 个库，实际只用到 california_schools）。

| 步骤 | 算子 | 行数 | 说明 |
|---|---|---|---|
| 1 | SQLExecutabilityFilter | 4 | EXPLAIN 可执行性过滤，4/4 通过 |
| 2 | SQLVariationGenerator | 4 | 每行生成 1 个语义变体；**重建行导致 question 全部置空**（源码行为，见数据需求笔记 §7③） |
| 3 | SQLExecutabilityFilter | 4 | 变体再次过滤，4/4 通过 |
| 4 | Text2SQLQuestionGenerator | 4 | 每行 2 个问题候选 + 本地 bge embedding 选优；question 重新生成、附带 question_type/evidence |
| 5 | Text2SQLCorrespondenceFilter | **3** | 问题-SQL-Schema 一致性过滤，淘汰 1 条（4→3） |
| 6 | Text2SQLPromptGenerator | 3 | 零 LLM 调用；prompt 6121–6232 字符（schema + 问题 + 指令） |
| 7 | Text2SQLCoTGenerator | 3 | 每条 1 条 CoT 候选，写入 `cot_responses` |
| 8 | Text2SQLCoTVotingGenerator | 3 | 执行一致性投票选出 `cot_reasoning`，973–1339 字符 |
| 9 | SQLComponentClassifier | 3 | 组件难度：medium / extra / medium |
| 10 | SQLExecutionClassifier | 3 | 执行难度：hard / easy / easy |

最终 3 行 × 10 列（9/10 列非空；`evidence` 为空串，原因见下）；**4→3 的过滤原因**：第 5 步淘汰的是重新生成出的 Metaphorical 类型问题（对应 Data Value Transformations 变体、continuation schools 比例题）——它是唯一带 external knowledge 的样本，被过滤后最终 evidence 字段全为空串。question_type 分布 Formal / Colloquial / Interrogative；sql_variation_type 为 Query Structure Modifications ×2 + Advanced SQL Features ×1。整条链 22:35:02 → 22:40:07 ≈ **5 分 05 秒、0 次 429**（缩采样 + 低并发 + 指数退避后）。

**gen（数据合成）也跑通**：无输入 SQL，由 `SQLGenerator(generate_num=1)` 对 3 个 SQLite 库（california_schools / student_club / thrombosis_prediction）各从 Schema 合成 1 条 SELECT（sql_complexity_type = Simple / Moderate / Complex 各 1）→ 可执行性过滤 3→3 → 问题生成 3（question_type Formal / Concise / Colloquial）→ 一致性过滤 **3→2**（thrombosis_prediction 的一条被淘汰；人工看问题与 SQL 语义一致，疑似 LLM 一致性判定假阴性）→ prompt / CoT / 双难度标签，9 步链路跑通，最终 2 行。运行 22:40:37–22:46:19，0 次 429，2 次上游 504 被重试吸收。

**难度分级与源码坑**：本次把 SQLExecutionClassifier 的 thresholds 从官方 [2,5,9] 改为 [1,2]，`exec_generations=3 > 2`，因此确实只生成 3 次、未被算子放大；若沿用官方 thresholds，`num_generations <= 9` 会触发 `((9//5)+1)*5=10` 的强制放大（见数据需求笔记 §7⑤）。执行难度因而按 3 次生成的执行通过率分档（easy×2 / hard×1）。smoke 版（2 条输入，num_gen=5 + thresholds [2,4]）跑到 step9、step10 因 29 次 429 重试风暴未完成，作为链路证据保留；正式版补全 step10。

### 4.3 不同 AI 应用场景的数据需求

见独立笔记 [task04-不同AI应用场景数据需求.md](task04-不同AI应用场景数据需求.md)：7 类场景的数据单元/字段/质量门槛/可验证信号/评测指标/DataFlow 对应 pipeline，两张字段映射表，跨场景对比表，以及「F1 评分器中文/标点行为」的单元级实测。

## 5. 质量边界与失败分析

1. **端点限流是本任务最大的工程约束**。AMD 端点对同一用户有 20 requests/min 全局限制，且单模型有共享并发上限（实测 GLM-5.3-Flash 报 concurrency limit 32）。两个 pipeline 并发时会互相挤占：第一次 AgenticRAG 尝试就因重试耗尽返回 None、字段缺失（step2 报 Missing required column(s): refined_answer、golden_doc_answer），加进程级 RPM 限速后重跑成功。**结论：数据合成的成本模型要按「限流后的有效吞吐」算，而不是理论并发**（本次 66 请求/654.7s ≈ 6.1 req/min）。
2. **token-F1 的语言/格式敏感**：中文无空格时近似 exact-match；日期 `2017-08-30` vs `August 30, 2017` 判 0；英文 en-dash 未归一化导致 0.6667。做中文 RAG 的 RL 奖励必须换评分口径（中文分词/字符级 F1/embedding 余弦/LLM judge），并记录评分器版本。
3. **Text2SQL 的 CoT 与 gold SQL 不强制字面一致**：投票按执行结果一致性选 CoT，示例中 CoT 给出的 SQL 与 SQL 字段不同但结果集一致（这是设计意图）；源码在候选全部执行失败时会回退第一条 CoT（不淘汰），因此训练前应再做一次执行校验。
4. **AgenticRAG 单跳是软验证**：`llm_score<1` / `golden_doc_score>=1` 都来自 LLM 判分，不是规则验证；多跳 pipeline 有「删支持文档反事实」的 shortcut 检验，但本次未跑（需要 FlashRAGServing + e5 检索索引，时间与配额不足，如实标注）。
5. **未完成/未跑项**（如实记录）：smoke 版 SQLExecutionClassifier step10 因限流中断（正式 refine 已补全）；未做「训练后模型效果」评测，本任务只报告数据流水线可运行性与产出质量边界。此外本任务只用了 1 个库跑 refine、gen 每库 1 条，MySQL / sqlite-vec / text2vecsql 等变体未涉及。

## 6. 复现

远端：vm-60，`source ~/dataflow-task4/env.sh`（PATH 指向 `/home/simin/.venv-df3`，`DF_API_KEY` 从 key 文件读入，`AMD_API_BASE` / `DF_MODEL` 指向 AMD 兼容端点与 DeepSeek-V4-Flash）。数据库：`databases.zip` 解压到 `~/dataflow-task4/db/downloaded_databases/databases/`（california_schools 等 3 个 SQLite）。

```bash
# AgenticRAG：4 条文档，RPM=12 / workers=2（脚本内含进程级限速）
cd ~/dataflow-task4/agentic_rag && bash run_agentic_rag.sh main

# Text2SQL refine：4 条子集，一致性过滤后保留 3 条
cd ~/dataflow-task4/text2sql && python text2sql_refine_task4.py \
  --input subset_refine.jsonl --cache-dir ./cache_refine --prefix task4_refine \
  --num-variations 1 --question-candidates 2 --cot-sampling 1 --exec-generations 3 --max-workers 2
```

本地归档脚本：`code/task4/agentic_rag/`、`code/task4/text2sql/`；证据：`evidence/task4/`。

## 7. 材料索引

| 文件 | 内容 |
|---|---|
| [task04-不同AI应用场景数据需求.md](task04-不同AI应用场景数据需求.md) | 场景数据需求分析笔记（文档 + 实测交叉） |
| evidence/task4/agentic_rag/ | AgenticRAG 运行日志/逐步产物/F1 统计/REPORT |
| evidence/task4/text2sql/ | Text2SQL 运行日志/逐步产物/字段统计/REPORT |
| evidence/task4/docs/ | 官方两篇文档的本地快照（agenticrag_pipeline / text2sqlpipeline） |
| evidence/task4/dataflow-task4-checkin.md | 打卡文案 |
| code/task4/agentic_rag/ | 改造后的 AgenticRAG pipeline + 复现脚本 |
| code/task4/text2sql/ | 改造后的 Text2SQL pipeline + 复现脚本 |
