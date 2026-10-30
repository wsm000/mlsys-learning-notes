# Task4 · Text2SQL 数据流水线运行报告（refine 主链 + gen 合成）

- 日期：2026-10-29（北京时间 22:30–23:00 窗口）
- 远端：vm-60（Linux，RTX 4090 D），环境 `/home/simin/.venv-df3`（open-dataflow 1.0.11 + torch/transformers/sentence-transformers/faiss/openai/pandas，已就绪，**全程未 pip install**）
- 代码版本：open-dataflow 1.0.11 官方 `api_pipelines/text2sql_pipeline_refine.py` / `text2sql_pipeline_gen.py` 的最小改造版
- 结论：**refine 模式 10 个步骤全部真实跑通**（4 条输入 → 最终 3 条含 prompt/CoT/双难度标签样本），**gen 模式也跑通**（3 个库各合成 1 条 SQL 的完整链路）。smoke 阶段在 step10 遭遇账户级限流未能跑完，已如实记录。

---

## 0. 结论摘要

| 项目 | 数字 |
|---|---|
| refine 输入 | 官方 `pipeline_refine.jsonl` 前 4 条（db_id 均为 california_schools） |
| refine 各步行数 | 4 → 4 → 4 → 4 → **3** → 3 → 3 → 3 → 3 → 3 |
| refine 最终样本 | **3 条**，字段 10 个，9/10 字段非空 |
| refine 运行时长 | 22:35:02 – 22:40:07（约 5 分 05 秒），**0 次 429** |
| smoke（2 条链路验证） | step1–step9 各 2 行全通，step10 因限流卡死被终止（29 次 429） |
| gen（合成模式） | 3 个 SQLite 库 ×1 条 SQL → step1-3 各 3 行 → step4 一致性过滤后 2 行 → 最终 **2 行**；**0 次 429、2 次上游 504**；运行 22:40:37–22:46:19（5 分 42 秒） |
| 模型/端点 | DeepSeek-V4-Flash @ AMD Radeon OneClick OpenAI 兼容端点；embedding 为本地 bge-small-zh-v1.5（cuda） |

---

## 1. 环境与约束

### 1.1 软硬件

- 远端机器 `vm-60`，Linux，单卡 RTX 4090 D；工作目录 `~/dataflow-task4/`（远端 `/` 仅剩约 0.9GB，故所有 cache/日志放在 home 下，全程未产生大文件）。
- Python 环境：`/home/simin/.venv-df3`（Python 3.10.12），已装 open-dataflow 1.0.11、torch、transformers、sentence-transformers、faiss、openai、pandas；**没有执行任何安装/升级命令**。
- LLM：AMD Radeon OneClick 的 OpenAI 兼容端点
  `https://developer.amd.com.cn/radeon/api/v1/chat/completions`，
  `model=DeepSeek-V4-Flash`；key 只从环境变量 `DF_API_KEY` 读取（`source ~/dataflow-task4/env.sh`），**未写入任何文件、日志或报告**。
- Embedding：本地 HF 缓存 `~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d...`，用 `LocalEmbeddingServing(model_name=<snapshot>, device="cuda")`，不依赖外部 embedding API。

### 1.2 数据库（本机已解压的官方示例 SQLite，未重新下载）

来自官方 HF 数据集 `Open-Dataflow/dataflow-Text2SQL-database-example` 的解压产物：

| db_id | 文件大小 | 表（列数/行数） |
|---|---|---|
| california_schools | 11.1 MB | frpm(29/9986)、satscores(11/2269)、schools(49/17686) |
| student_club | 2.6 MB | event(7/42)、major(4/113)、zip_code(6/41877)、attendance(2/326)、budget(7/52)、expense(7/32)、income(6/36)、member(9/33) |
| thrombosis_prediction | 7.3 MB | Examination(13/806)、Patient(7/1238)、Laboratory(44/13908) |

（机读版见 `db_stats.json`）

### 1.3 真实限流约束（本任务最大的工程约束，如实记录）

AMD OneClick 端点对**该用户**存在多重限流，本次实测到三种 429：

1. **账户级 20 请求/分钟**（所有模型、所有 pipeline 共享）：
   `{"error":{"message":"Rate limit exceeded: maximum 20 requests per minute for this OneClick user...","code":"rate_limit_exceeded"}}`
2. **DeepSeek-V4-Flash 的 token 级限流**：
   `{"detail":{"error":{"message":"Model API rate limit exceeded; ...","code":"token_rate_limit_exceeded"}}}`
3. **GLM-5.3-Flash 的模型并发上限 32**：
   `{"detail":{"error":{"message":"Model 'GLM-5.3-Flash' is at its concurrency limit (32)...","code":"model_concurrency_rate_limit_exceeded"}}}`

同一时间 `rag-runner` 的 AgenticRAG pipeline 也在用同一端点，配额是共享的。应对策略：
- `max_workers=2`、`max_retries=8`（DataFlow 自带 2^i 指数退避）、`read_timeout=240s`；
- 样本和采样数量压缩（见 §3）；
- 22:36 实测 DeepSeek-V4-Flash 恢复 200 后，正式运行回到该模型（GLM 尝试因并发上限失败，见 §4 时间线）。

---

## 2. 数据与输入

- **refine 模式**：官方示例 `pipeline_refine.jsonl` 共 60 行（california_schools 的 question+SQL）。用 `make_subset.py` 截取**前 4 行**作 `subset_refine.jsonl`；smoke 用前 2 行作 `subset_smoke.jsonl`。
- **gen 模式**：无输入 SQL（`first_entry_file_name=""`，FileStorage 返回空 DataFrame），由 `SQLGenerator(generate_num=1)` 对 3 个库各生成 1 条 SELECT。
- 两个子集与原文件都在远端 `~/dataflow-task4/text2sql/`，证据目录同步了 refine/smoke 的每一步 cache jsonl。

---

## 3. 对官方 pipeline 的改造点

| 配置 | 官方默认 | 本任务 | 原因 |
|---|---|---|---|
| `db_root_path` | `""` 自动从 HF 下载 | `/home/simin/dataflow-task4/db/downloaded_databases/databases` | 远端磁盘紧张，复用手工解压的示例库 |
| `llm_serving` | OpenAI gpt-4o | AMD 兼容端点 + DeepSeek-V4-Flash | 本任务指定；`max_workers=2`、`max_retries=8` 抗 429 |
| `embedding_serving` | `APILLMServing_request`(ada-002) | `LocalEmbeddingServing(bge-small-zh-v1.5, cuda)` | 无 embedding API 配额，本地模型更稳 |
| `num_variations` | 3 | **1** | 时间预算 + 20 RPM 限流 |
| `question_candidates_num` | 3 | **2** | 同上；2 候选时算子直接用随机取一（不走 embedding 距离） |
| CoT `sampling_num` | 默认 3 | **1** | 同上（注意这会使 step8 投票退化为单候选，见 §9） |
| `SQLExecutionClassifier.num_generations` | 10 | **3**（thresholds `[1,2]`，labels `[extra,hard,easy]`） | 该算子内部要求 `num_generations > 最后一个 threshold`，否则强制放大到 `((last//5)+1)*5`（最少 5）；改成 `[1,2]` 后 3 次生成可用 |
| `SQLComponentClassifier` | `[2,4,6]` / easy..extra | 不变 | 纯 SQL 解析，无 LLM 成本 |
| 统计 | 无 | 每步读 cache jsonl 落 `stats_refine.json` / `stats_gen.json` | 证据可追溯 |

**关于算子的 num_generations 强制放大规则（实证）**：`SQLExecutionClassifier.__init__` 中若 `num_generations <= thresholds[-1]`，会被强制改成 `((thresholds[-1]//5)+1)*5`（最少 5）。若沿用官方 thresholds `[2,5,9]` 而把生成次数压到 3，会触发 `((9//5)+1)*5 = 10`，等于白压；因此最终配置把 thresholds 改为 `[1,2]`（3 > 2，不触发放大），**最终执行难度确实基于 3 次生成**（easy=3/3 一致、hard=2/3 一致），并未被放大到 5 或 10。smoke 阶段用 `num_generations=5 + thresholds [2,4]`（5 > 4）同样未触发。

新增文件：`text2sql_refine_task4.py`、`text2sql_gen_task4.py`、`make_subset.py`、`analyze_cache.py`、`run_refine.sh`、`run_gen.sh`、`launch_*.sh`；均通过 `source ../env.sh` 取 key，不 hardcode。

---

## 4. 运行时间线（北京时间）

| 时间 | 事件 |
|---|---|
| 22:25 | 远端数据库/环境确认；下载官方 pipeline 源码与 prompt 模板到本地核对 |
| 22:30:21 | **smoke 启动**（2 条，DeepSeek-V4-Flash，max_workers=4，max_retries=5），step1–3 秒过 |
| 22:31:06–22:32:19 | smoke step4–9 完成；期间开始出现 429（前 13 次账户级 20 RPM，后为 DeepSeek token 级） |
| 22:32:2x | smoke step10 `SQLExecutionClassifier`（10 个 LLM 请求）陷入 429 重试风暴，进程被终止（前后共 29 次 429） |
| 22:33:18 | 按建议切 **GLM-5.3-Flash** 重跑 refine：step1 完成，step2 立即遇 `model_concurrency_rate_limit_exceeded (32)` 8 次，放弃并回滚 |
| 22:35:02 | **正式 refine 启动**（4 条，DeepSeek-V4-Flash，workers=2，retries=8），step1 完成 |
| 22:35:55 | step2/3 完成（变体生成 4 条 + EXPLAIN 过滤 4 条） |
| 22:38:38 | step4 问题生成完成（8 个请求，4 条全部有 question/evidence） |
| 22:39:17 | step5 一致性过滤完成（4→3），step6 prompt 生成完成 |
| 22:39:55 | step7 CoT 生成 + step8 投票完成（3 条） |
| 22:40:07 | step9 组件难度 + step10 执行难度完成；**全链结束，0 次 429** |
| 22:40:37 | **gen 模式启动**（3 库 ×1 条 SQL） |
| 22:41:42 | gen step1 SQL 合成 3 条 + step2 EXPLAIN 全过 |
| 22:44:30 | gen step3 问题生成完成（3 条，期间 2 次上游 504 重试） |
| 22:45:04 | gen step4 一致性过滤 3→2；step5 prompt 完成 |
| 22:46:00 | gen step6 CoT + step7 投票完成（2 条） |
| 22:46:19 | gen step8/9 难度分级完成；**gen 全链结束，0 次 429** |

refine 每步行数（`stats_refine.json`）：

```
step1 ExecutabilityFilter       4
step2 SQLVariationGenerator     4   (num_variations=1)
step3 ExecutabilityFilter       4
step4 Text2SQLQuestionGenerator 4   (2 candidates/row → 8 个 LLM 请求)
step5 CorrespondenceFilter      3   (1 条被 LLM 判定问题-SQL 不一致)
step6 Text2SQLPromptGenerator   3   (零 LLM 调用)
step7 Text2SQLCoTGenerator      3   (3 个 LLM 请求, sampling_num=1)
step8 Text2SQLCoTVotingGenerator 3  (执行一致性投票)
step9 SQLComponentClassifier    3
step10 SQLExecutionClassifier   3   (3 generations/row → 9 个 LLM 请求)
```

总 LLM 调用约 4+8+4+3+9 = **28 次**，5 分钟窗口约 5.6 次/分钟，低于 20 RPM 上限，故本次 refine 0 次 429。

gen 每步行数（gen/stats_gen.json，共 9 步）：

    step1 SQLGenerator                3   (3 库各 1 条: student_club/Moderate, california_schools/Complex, thrombosis_prediction/Simple)
    step2 ExecutabilityFilter        3   (EXPLAIN 全过)
    step3 Text2SQLQuestionGenerator  3   (2 candidates/row → 6 个请求；期间 2 次 504 重试)
    step4 CorrespondenceFilter      3→2  (thrombosis_prediction 的 Colloquial 问题被判定不一致)
    step5 Text2SQLPromptGenerator    2
    step6 Text2SQLCoTGenerator       2   (sampling_num=1)
    step7 Text2SQLCoTVotingGenerator 2
    step8 SQLComponentClassifier     2
    step9 SQLExecutionClassifier     2   (3 generations/row → 6 个请求)

gen 总 LLM 调用约 3+6+3+2+6 = **20 次**，5 分 42 秒完成，0 次 429、2 次上游 504 重试成功。

---

## 5. 逐步结果分析

1. **SQLExecutabilityFilter（step1/step3）**：用 `EXPLAIN QUERY PLAN` 验证可执行性，不真正执行。4/4 原始 SQL、4/4 变体 SQL 全部通过（该库 schema 下 EXPLAIN 均成功）。
2. **SQLVariationGenerator（step2）**：4 个 prompt 返回 4 条可解析变体；`sql_variation_type` 分布：Query Structure Modifications 2、Data Value Transformations 1、Advanced SQL Features 1。注意该算子输出的新行会把 `question` 置空（因此后续问题均为重新生成，原问题不参与后续步骤）。
3. **Text2SQLQuestionGenerator（step4）**：每行 2 个候选 → 8 个请求全部返回；抽取到 question_type 分布：Colloquial / Metaphorical / Interrogative / Formal 各 1；`evidence` 列生成但 4 条值均为空串（只有 Vague/Metaphorical 风格会填外部知识，见 §6 的连锁影响）。
4. **Text2SQLCorrespondenceFilter（step5）**：LLM 判定 3/4 通过（日志 `Correspondence check results: 3 passed, total 4`）。**被丢掉的正是第 2 条输入对应的 Metaphorical 问题**（Data Value Transformations 变体）：
   - question: "Which five continuation schools have the highest share of students who are eligible for free meals, compared to their total student population aged 5-17? (Assume the ratio is calculated as the number of free meal recipients divided by total enrollment for that age group.)"
   - evidence: "Continuation schools are alternative high schools ... free meal eligibility is a common proxy for poverty ..."
   - SQL: SELECT Free Meal Count (Ages 5-17) / Enrollment (Ages 5-17) FROM frpm WHERE Educational Option Type = 'Continuation School' ... ORDER BY ... DESC LIMIT 5;
   **这是本任务 refine 唯一的过滤损失，并导致最终样本 evidence 全空**（该风格被过滤后，其余非 Vague/Metaphorical 风格本就不填外部知识）。
5. **Text2SQLPromptGenerator（step6）**：纯模板 + schema 拼接，3/3 生成 `prompt`（约 6.1–6.2KB，含 M-Schema、Database Engine、问题与"Let's think step by step"指令）。
6. **Text2SQLCoTGenerator（step7）**：3/3 生成 `cot_responses`（每行是长度 1 的列表，因为 sampling_num=1），内容为分步推理 + 末尾 sql 代码块。
7. **Text2SQLCoTVotingGenerator（step8）**：对 `cot_responses` 中解析出的 SQL 实际执行，按结果签名投票；输出 `cot_reasoning` 并移除 `cot_responses` 列，3/3 成功。**单候选使投票逻辑退化**：`_vote_select` 只有 1 个候选时直接取它（若执行失败则回退原文本），因此本任务只验证了该步可运行、可执行，未验证多候选投票的有效性（限制见 §9）。
8. **SQLComponentClassifier（step9）**：纯规则解析 SQL 组件复杂度，3/3；结果 medium 2、extra 1。
9. **SQLExecutionClassifier（step10）**：让模型仅凭 prompt 生成 3 条 SQL，执行并与 gold SQL 结果比对；结果 easy 2、hard 1。按本任务阈值（0–1 extra / 2 hard / 3 easy），即 2 条样本 3 次生成全部与 gold 一致、1 条 2/3 一致——**说明最终 QA/SQL 对自洽度较好，gold SQL 均可执行**。

---

## 6. 最终产出与质量检查

最终 3 条样本字段（`task4_refine_step10.jsonl`）：

`db_id, SQL, question, evidence, prompt, cot_reasoning, sql_variation_type, question_type, sql_component_difficulty, sql_execution_difficulty`（共 10 列）

- **字段齐全度：9/10 非空**。`evidence` 列存在但 3 条均为空字符串（原因见 §5.4：唯一带外部知识的 Metaphorical 样本被一致性过滤丢掉）。其余 9 个字段 3/3 非空。
- 长度统计（quality_checks.json）：prompt 6121–6232B（均值 6174）；cot_reasoning 973–1339B（均值 1131）；SQL 316–1619B（均值 761）；均含最终 SQL。
- 难度分布：组件难度 medium×2、extra×1；执行难度 easy×2、hard×1。
- 变体类型：Query Structure Modifications×2、Advanced SQL Features×1；问题风格：Colloquial / Interrogative / Formal 各 1。
- 抽检（完整样例见 `samples_refine.jsonl`，下面是节选）：

样例 1（Colloquial 风格问题 + 窗口函数变体 SQL）：
```
question: Hey! Could you help me figure out which school in Alameda County has the highest
          proportion of students getting free meals? ... Just give me that top ratio, please!
SQL:      WITH ranked_frpm AS (SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` AS free_meal_ratio,
          ROW_NUMBER() OVER (ORDER BY (CAST(...) AS REAL) / ... DESC) AS rn FROM frpm
          WHERE `County Name` = 'Alameda') SELECT free_meal_ratio FROM ranked_frpm WHERE rn = 1;
cot_head: I'll work through this step-by-step. 1. **Understand the question**: ...
difficulty: component=medium, execution=hard
```

样例 2（Interrogative 风格 + 窗口函数/UNION 变体）：
```
question: Which zip codes contain charter schools that belong to the 'Fresno County Office of
          Education' district, and could you list them in ascending order?
SQL:      -- Transform original query to use a window function and a UNION with a pivot-like aggregation
          ... (变体 SQL)
difficulty: component=extra, execution=easy
```

- **SQL 可执行性独立复核**：直接对最终 3 条 gold SQL 在对应 SQLite 上执行，分别返回 1 行（free_meal_ratio）、3 行（Zip）、1 行（MailStreet），均无异常。
- 可执行性证据链：step1/3 的 EXPLAIN 过滤 → step8 投票时实际执行候选 SQL → step10 生成 SQL 与 gold 执行结果比对，三层都真实执行过；但**未做人工逐条语义审核**，属于弱监督质量结论。

---

## 7. smoke 运行（链路验证，含限流事故记录）

- 输入 2 条 / DeepSeek-V4-Flash / workers=4、retries=5；**step1–step9 各 2 行全部完成**（cache 见 `smoke/`）。
- step10（`SQLExecutionClassifier`，2 行 × 10 generations = 20 个请求）在 22:32 起陷入 429 重试风暴，单次重试最多等待 16s，2 分钟无法推进，进程被主动终止。
- smoke 日志共 **29 次 `status=429`**：其中 16 次 DeepSeek token 级限流、13 次账户级 20 RPM；两种 body 原文已保存在 `smoke.log`。
- smoke 产出字段 8/9 非空（无 execution 难度；evidence 为空串）；问题风格 Concise/Descriptive 各 1，组件难度 hard/medium 各 1。
- 教训：**step10 是 LLM 调用最密集的一步（rows × num_generations）**，在共享 20 RPM 下最容易卡死；正式 run 因此把 num_generations 压到 3、并发压到 2，顺利通过。

---

## 8. gen 合成模式结果

- 输入：无 SQL 数据，SQLGenerator 对 3 个库各生成 1 条 SELECT（generate_num=1），3/3 通过 EXPLAIN（step2 = 3 行）：

    student_club           Moderate  SELECT COUNT(DISTINCT m.member_id) AS active_members_with_dues FROM member m INNER JOIN income i ON ... WHERE i.source='Dues'
    california_schools     Complex   WITH FRPM_CTE AS (SELECT ... FROM frpm WHERE Academic Year='2014-2015' ...) ...
    thrombosis_prediction  Simple    SELECT COUNT(*) FROM Examination WHERE aCL IgG > 1.0 AND aCL IgM > 1.0 AND Thrombosis = 1

- 问题生成（step3 = 3 行）：Formal / Concise / Colloquial 各 1；期间 2 次上游 504 网关超时（47–50s）后自动重试成功（gen.log 中 status=500，body=call_upstream_timeout）。
- **一致性过滤（step4 = 2 行）**：thrombosis_prediction 的 Colloquial 问题被 LLM 判定不一致而丢弃：
    question: "Hey! Could you tell me how many patients had both aCL IgG and aCL IgM levels above 1.0 and also had thrombosis? I'm curious about the count."
    SQL: SELECT COUNT(*) FROM Examination WHERE aCL IgG > 1.0 AND aCL IgM > 1.0 AND Thrombosis = 1;
  人工看二者语义一致，**疑似 LLM 一致性判定的假阴性**（该步只有单次 yes/no 判定，无二次校验）。这暴露了 CorrespondenceFilter 在合成数据上的判定风险。
- 剩余 2 条完成 prompt / CoT / 投票 / 难度分级（step5–9 = 2 行）：
    student_club：Moderate / Formal，component=hard，execution=extra，prompt 4840B，cot 1393B
    california_schools：Complex / Concise，component=extra，execution=extra，prompt 6241B，cot 1917B
- execution=extra（3 次生成中 0–1 次与 gold 执行一致）说明**合成 SQL 比 refine 变体更难**：模型仅凭 prompt 难以写出执行等价的复杂 SQL，符合"从 schema 从零合成更难"的预期。
- 独立执行复核：2 条最终 SQL 在对应 SQLite 上分别返回 1 行、3 行，均无异常。
- 统计与样例：gen/summary_gen.json、gen/samples_gen.jsonl、gen/stats_gen.json、gen/gen.log（gen.log 同目录）；本次 **0 次 429**。

---

## 9. 失败点与限制（诚实边界）

1. **限流是真实瓶颈**：账户级 20 RPM 全局限流 + 模型级 token/并发限流，且与另一 pipeline 共享。本任务的吞吐/步数完全受此约束，smoke 的 step10 就是牺牲品；正式 refine 通过缩采样 + 低并发 + 指数退避才 0 次 429 跑完。任何"更大样本/更多候选"的结论在本任务都未验证。
2. **样本量小**：正式 refine 只有 4 条输入、最终 3 条输出（受限流与截止时间约束）。这是**链路打通 + 工程可行性证据**，不是数据质量或模型效果评估。
3. **CoT 投票退化**：`sampling_num=1` 使 step8 单候选，执行一致性投票没有真正发挥作用；多候选投票逻辑未在本任务验证。
4. **evidence 全空**：唯一可能带外部知识的 Metaphorical 问题被一致性过滤丢弃；如需 evidence 字段非空，应提高 question_candidates 或换风格采样重跑。
5. **原问题被替换**：refine 的变体生成会清空原 question，最终问题是 LLM 基于变体 SQL 重新生成的（pipeline 设计如此），因此与官方 60 条原问题的语义一致性未逐条校验。
6. **覆盖范围有限**：refine 只用了 california_schools；MySQL / sqlite-vec / text2vecsql 变体未涉及；gen 才用到 3 个库。
7. **temperature=0.7**：为了变体/问题/CoT 多样性而调高（官方默认 0），代价是可复现性下降、格式偶发不稳定（本次解析成功率仍 100%）。
8. **gen 一致性过滤疑似假阴性**：thrombosis_prediction 的一条问题与 SQL 人工看语义一致，却被单次 LLM yes/no 判定丢弃（3→2）；说明该过滤器的假阴性风险，生产环境应加二次校验/人工抽检。
9. **上游 504 网关超时**：gen 阶段出现 2 次 status=500（call_upstream_timeout，47–50s），靠 DataFlow 重试恢复；说明除限流外端点还有稳定性抖动。
10. **未做下游训练/评测**：本任务只负责"数据构建"，产出的 jsonl 未经 Text2SQL 模型微调或执行准确率评测。
11. **环境时间异常**：本地会话在 22:46 后经历一次长时间挂起/恢复（工具侧计时异常），证据归档完成时间晚于运行时间；两条 pipeline 真实运行时间以远端日志为准（22:40:07 / 22:46:19 结束，早于 23:59 截止）。

---

## 10. 吞吐估算（共享 20 RPM 端点下的数据合成）

- 单条 refine 样本的 LLM 调用数（本任务配置）= 1（变体）+ 2（问题候选）+ 1（一致性）+ 1（CoT）+ 3（执行难度）= 约 **7 次/样本**；4 条输入约 28 次调用，实测 5 分钟完成（≈5.6 调用/分钟，远低于 20 RPM，说明主要时间花在单次长 prompt 的生成延迟 ≈15–40s，而非限流）。
- 若独占 20 RPM 且 2 并发、平均 20s/请求：约 **6 次调用/分钟 ≈ 0.85 条最终样本/分钟**。
- 官方默认配置（variations=3、候选=3、CoT=3、exec=10）约 15–20 次调用/样本，按同样吞吐约 **0.3–0.4 条样本/分钟**；生成 1 万条高质量 Text2SQL 训练数据在该端点需要 ~20–30 小时，与本地 vLLM 批量生成（无 RPM 限制）不在一个量级——这是"共享 API 端点做数据合成"的硬约束。
- gen 模式约 **7–10 次调用/样本**：3 库各 1 条 SQL → 3+6+3+2+6 = 20 次调用，5 分 42 秒完成（≈3.5 调用/分钟，其中含 2 次上游 504 重试）。
- 以上为量级估算，非精确 benchmark。

---

## 11. 复现命令

远端（vm-60）：

```bash
cd ~/dataflow-task4/text2sql
source ~/dataflow-task4/env.sh

# 1) 生成输入子集（refine 用前 4 条；smoke 用前 2 条）
EX=~/.venv-df3/lib/python3.10/site-packages/dataflow/example/Text2SQLPipeline
python make_subset.py --input "$EX/pipeline_refine.jsonl" --output subset_refine.jsonl --num 4

# 2) refine 全链（后台）
bash run_refine.sh > refine.log 2>&1 &        # 或 launch_refine.sh

# 3) 统计每步行数 / 字段 / 限流次数，抽 final 样本
python analyze_cache.py --cache-dir ./cache_refine --prefix task4_refine \
  --log refine.log --out summary_refine.json --samples-out samples_refine.jsonl --final-step 10

# 4) gen 合成模式（可选）
bash run_gen.sh > gen.log 2>&1 &
python analyze_cache.py --cache-dir ./cache_gen --prefix task4_gen \
  --log gen.log --out summary_gen.json --samples-out samples_gen.jsonl --final-step 9
```

关键实现：`text2sql_refine_task4.py`（refine 10 步，含每步 stats）、`text2sql_gen_task4.py`（gen 9 步）、`make_subset.py`、`analyze_cache.py`；模型/key 全部从 `env.sh` 的环境变量读取，无硬编码。

---

## 12. 证据与代码清单

| 路径 | 内容 |
|---|---|
| `evidence/task4/text2sql/refine/refine.log` | 正式 refine 全量日志（含每步行数、过滤结果、难度统计） |
| `evidence/task4/text2sql/refine/cache/task4_refine_step1..10.jsonl` | 每一步 cache 原始 jsonl（step10 为最终产出） |
| `evidence/task4/text2sql/refine/stats_refine.json` | 每步行数/字段/非空数/参数快照 |
| `evidence/task4/text2sql/refine/summary_refine.json` | 每步统计 + 难度分布 + 日志 429 统计（机读） |
| `evidence/task4/text2sql/refine/samples_refine.jsonl` | 最终 2 条完整样例（含完整 prompt/cot） |
| `evidence/task4/text2sql/refine/final_refine.jsonl` | 最终 3 条样本（step10 副本，便于直接使用） |
| `evidence/task4/text2sql/refine/subset_refine.jsonl` | 输入子集（4 条，来自官方 pipeline_refine.jsonl） |
| `evidence/task4/text2sql/refine/quality_checks.json` | 字段齐全度、长度统计、分布（机读） |
| `evidence/task4/text2sql/smoke/smoke.log` + `task4_refine_smoke_step1..9.jsonl` | smoke 证据（step10 未完成，日志含 29 次 429 原文） |
| `evidence/task4/text2sql/smoke/summary_smoke.json` / `samples_smoke.jsonl` | smoke 统计与样例 |
| `evidence/task4/text2sql/db_stats.json` | 3 个 SQLite 库的表/列/行数统计 |
| `evidence/task4/text2sql/gen/gen.log` | gen 全量日志（含 2 次上游 504 原文、0 次 429） |
| `evidence/task4/text2sql/gen/cache/task4_gen_step1..9.jsonl` | gen 每步 cache（step9 为最终产出） |
| `evidence/task4/text2sql/gen/stats_gen.json` / `summary_gen.json` / `samples_gen.jsonl` | gen 统计与样例 |
| `evidence/task4/text2sql/gen/db_stats.json` | 3 个 SQLite 库表/列/行数统计 |
| `code/task4/text2sql/text2sql_refine_task4.py` | refine pipeline 改造脚本 |
| `code/task4/text2sql/text2sql_gen_task4.py` | gen pipeline 改造脚本 |
| `code/task4/text2sql/make_subset.py` / `analyze_cache.py` | 子集生成 / 统计脚本 |
| `code/task4/text2sql/run_refine.sh` / `run_gen.sh` | 复现 shell |
