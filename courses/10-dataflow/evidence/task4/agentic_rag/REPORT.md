# Task4 · AgenticRAG 数据合成 Pipeline 实测报告

> 课程任务：为 Agentic RAG（含 RL 奖励）构建可验证的高质量问答数据。
> 本文所有数字均来自 vm-60 上 **2026-10-29 22:33–22:44（北京时间）** 的一次真实运行产物，
> 原始日志与 jsonl 见 `courses/10-dataflow/evidence/task4/agentic_rag/`。

## 1. 结论速览

| 项 | 实测值 |
|---|---|
| 输入文档数 | 4（dataflow 安装包 `AgenticRAGPipeline/eval_test_data.jsonl` 前 4 条） |
| 展开的候选任务数 | 8（`max_per_task=2`） |
| 最终输出行数 | **4**（`max_question=1`，每文档保留 1 条） |
| 字段齐全率 | **11/11 = 100%**（文档列出的 11 个字段全部出现） |
| 字段空值率 | **0%**（11 个字段 × 4 行，全非空） |
| F1Score 均值 | **0.6957**（n=4：1.0 / 1.0 / 0.7826 / 0.0） |
| LLM 调用（含重试） | 66 次 HTTP 请求 / 53 次逻辑调用 |
| 端到端耗时 | **654.7 s**（step1 654.7 s，step2 0.0 s，F1 为本地计算） |
| 实测吞吐 | 6.05 req/min（逻辑 4.86 calls/min） |
| 失败请求 | 5 次（3×429 限流 + 2×500 provider 504），**全部被重试吸收，0 行丢失于重试** |
| 解析失败 | 0（`[WARN] Failed to parse` = 0，`recall score_str error` = 0，无 Traceback） |

**一句话**：官方 AgenticRAG 两个算子（原子问答生成 + F1 评分）在 AMD 兼容端点上跑通，
字段链路完整，但**吞吐被端点的 20 requests/min 全局限流硬约束**——这是本任务最真实的工程结论。

## 2. 环境与版本

| 项 | 值 |
|---|---|
| 机器 | vm-60（`user1-virtual-machine`），Linux，NVIDIA RTX 4090 D |
| Python | 3.10.12（`/home/simin/.venv-df3`） |
| open-dataflow | **1.0.11** |
| LLM 端点 | `https://developer.amd.com.cn/radeon/api/v1/chat/completions`（OpenAI 兼容） |
| 模型 | `DeepSeek-V4-Flash` |
| 磁盘 | `/` 仅剩 ~0.86 GB，工作目录置于 `~/dataflow-task4/agentic_rag` |

DataFlow 侧配置：`APILLMServing_request(max_workers=2, max_retries=6, connect_timeout=10, read_timeout=180)`
+ 自研 `ThrottledAPILLMServing_request`（进程级匀速放行，`rpm=12`）。

## 3. 输入数据

来源：安装包内置 `dataflow/example/AgenticRAGPipeline/eval_test_data.jsonl`（共 10 条，字段 `id`/`contents`）。
本次按**前 4 条**（可复现，非随机抽样）构造子集 `input_main_4docs.jsonl`：

| id | contents 长度（字符） |
|---|---|
| 17989685 | 580 |
| 4864947 | 583 |
| 12455656 | 617 |
| 18253508 | 603 |

## 4. Pipeline 结构（官方 2 步，未改算子逻辑）

```
contents (输入)
  │
  ├─ Step1  AgenticRAGAtomicTaskGenerator（8 个 LLM prompt 轮次）
  │    ① get_identifier        → identifier            (content_identifier JSON)
  │    ② get_conclusions       → candidate_tasks_str   (每文档最多 max_per_task 条 {conclusion,R})
  │    ③ init_question         → question + answer     (按候选任务展开，行数翻倍)
  │    ④ clean_qa              → refined_answer
  │    ⑤ llm_answer            → llm_answer            (无上下文作答)
  │    ⑥ get_recall_score      → llm_score = 0/1/2     (过滤 llm_score < 1：剔除"没文档也能答对"的简单题)
  │    ⑦ golden_doc_answer     → golden_doc_answer     (严格基于原文档作答)
  │    ⑧ get_golden_answer_score → golden_doc_score    (过滤 golden_doc_score >= 1)
  │    ⑨ more_optional_answer  → optional_answer (list) (可接受答案集合，供 F1 多参考答案匹配)
  │    ⑩ groupby(contents).head(max_question) → 每文档截断
  │
  └─ Step2  AgenticRAGQAF1SampleEvaluator（纯本地，无 LLM）
       refined_answer vs golden_doc_answer → F1Score（token 级 P/R/F1，支持多参考答案取 max）
```

## 5. 相对官方脚本的改造点

官方：`dataflow/statics/pipelines/api_pipelines/agentic_rag_pipeline.py`

| # | 官方 | 本任务 | 原因 |
|---|---|---|---|
| 1 | `first_entry_file_name="../example_data/..."`（相对路径） | 安装包内**绝对路径** + `T4_INPUT` 覆盖 | 官方默认先 `dataflow init` 到 `run_dataflow/`；相对路径在其他 cwd 下必然 FileNotFoundError |
| 2 | `cache_path="./agenticRAG_eval_cache"` | `./cache_<profile>`（`T4_CACHE`） | 便于按 profile 归档每步产物 |
| 3 | `file_name_prefix="agentic_rag_eval"` | `agentic_rag_task4` | 与本课程证据命名对齐 |
| 4 | `api_url="https://api.openai.com/v1/chat/completions"`, `gpt-4o-mini` | AMD 端点 + `DeepSeek-V4-Flash`（环境变量注入） | 本任务指定模型 |
| 5 | `max_workers=500` | **`max_workers=2`** | 端点限流 20 req/min；500 并发 = 429 风暴（实测见 §8） |
| 6 | 未设 read_timeout（默认 120 s） | `read_timeout=180`，`max_retries=6` | 长文生成偶发 >120 s；504 需退避重试 |
| 7 | 无客户端限流 | 新增 `ThrottledAPILLMServing_request`（`rpm=12`，线程安全匀速放行，覆写 `_api_chat_with_id`） | 让 20 RPM 的服务端限流不被触发；放在脚本内，**不改 dataflow 包** |
| 8 | 算子参数默认（`max_per_task=10, max_question=10`） | `max_per_task=2, max_question=1` | 20 RPM 下控制调用量（10 条文档 ×10 候选 ≈ 600+ 调用不可行） |

算子内部逻辑、prompt 模板、字段名**完全未改**。

## 6. 运行方式（可一键复现）

```bash
# 本地代码 → 远端
scp courses/10-dataflow/code/task4/agentic_rag/{agentic_rag_task4.py,verify_task4_agentic_rag.py,run_agentic_rag.sh} \
    vm-60:/home/simin/dataflow-task4/agentic_rag/

# 远端执行（后台 + 落盘日志）
ssh vm-60 "cd /home/simin/dataflow-task4/agentic_rag && \
  (T4_ROWS=4 T4_RPM=12 T4_MAX_WORKERS=2 T4_MAX_PER_TASK=2 T4_MAX_QUESTION=1 \
   setsid nohup bash run_agentic_rag.sh main > run_main.log 2>&1 < /dev/null &)"
```

`run_agentic_rag.sh` 会：① 用 `json.loads` 取前 N 条生成 `input_<profile>.jsonl`；
② `source /home/simin/dataflow-task4/env.sh`（注入 `DF_API_KEY`/`AMD_API_BASE`/`DF_MODEL`）；
③ 跑 pipeline；④ 调 `verify_task4_agentic_rag.py` 产出 `field_check_<profile>.json` 与首行样例。

## 7. 结果

### 7.1 逐步耗时（4 文档 / 8 候选行，`rpm=12`）

| 阶段 | 耗时 | 调用数 | 单次均值 |
|---|---|---|---|
| get_identifier | 16.7 s | 4 | 4.2 s |
| get_conclusions | 76.2 s | 4 | 19.0 s |
| init_question | 36.3 s | 8 | 4.5 s |
| clean_qa | 54.1 s | 8 | 6.8 s |
| llm_answer | 134.3 s | 8 | 16.8 s |
| golden_doc_answer | 63.4 s | 7 | 9.1 s |
| golden_doc_verify | 103.7 s | 7 | 14.8 s |
| more_optional_answer + 写盘 | 161.8 s | 7 | 23.1 s（含 2 次 429/504 退避） |
| **合计** | **654.7 s** | 53 逻辑 / 66 HTTP | **6.05 req/min** |

限速器按 5 s/请求 放行（12 rpm），但实际均值 9.9 s/请求 → **瓶颈是模型生成延迟与偶发 429/504 退避，不是限速器**。

### 7.2 过滤漏斗

```
4 文档
 → 8 候选任务 (conclusion+R)        ← max_per_task=2
 → 8 QA 对 (question/answer)
 → 7 行 (llm_score < 1)            ← 1 条题目"无文档也能答对"(llm_score>=1) 被剔除，符合算子设计
 → ≥4 行 (golden_doc_score >= 1)   ← 每个文档至少 1 行存活
 → 4 行 (groupby(contents).head(1)) ← max_question=1
```
注：`golden_doc_score` 过滤后的精确行数**无法从最终产物反推**（`head(max_question=1)` 已抹掉同文档多余行），
只能确定 ∈ [4,7]；此处如实标注，不做推测。

### 7.3 字段完整性（`field_check_main.json`）

- 期望字段 11 个：`question, answer, refined_answer, optional_answer, llm_answer, golden_doc_answer,
  identifier, candidate_tasks_str, llm_score, golden_doc_score, F1Score`
- **全部 present：11/11 = 100.0%**
- 每字段空值率 0/4 = 0%
- 实际列（13 列）= 11 个文档字段 + 输入透传的 `id`/`contents`
- 类型：`optional_answer` 为 list（10–16 条备选答案）；`identifier`/`candidate_tasks_str` 为 JSON 字符串；
  `llm_score`/`golden_doc_score` 为 int；`F1Score` 为 float

### 7.4 F1Score 分布

| 统计量 | 值 |
|---|---|
| n | 4 |
| mean | **0.6957** |
| min / median / max | 0.0 / 1.0 / 1.0 |
| =0 行数 | 1 |
| ≥0.5 行数 | 3 |
| =1.0 行数 | 2 |

样本：[1.0, 1.0, 0.7826, 0.0]，属于 n=4 的小样本，**不能当作该 pipeline 的质量分布结论**，只作为链路正确性证据。

### 7.5 两个中间打分（真实值）

| 字段 | 值 | 含义 |
|---|---|---|
| `llm_score` | 4 行全为 **0** | 无上下文时 LLM 答不出 → 说明题目确实依赖文档（这正是该 filter 想要的信号） |
| `golden_doc_score` | [2, 2, 1, 2]，均值 1.75 | 有文档作答时 LLM 判分 ≥1 → 全部通过 `>=1` 过滤 |

## 8. 失败、修复与 429 证据（重要工程结论）

### 8.1 第一次运行（pilot）失败：429 风暴 → step1 空输出

- 配置：`max_workers=6, max_retries=3, rpm 无限速`，2 文档 / `max_per_task=2`
- 结果：**40 次 `status=429`**（原始日志 `run_pilot_429_evidence.log`），
  所有 LLM 响应变 `None` → `json.loads(None)` 失败 → step1 写出 **0 行**（JSONL 仅 1 字节）→
  Step2 报 `ValueError: Missing required column(s): ['refined_answer', 'golden_doc_answer']`
- 服务端 429 原文（两种）：
  - `{"message":"Rate limit exceeded: maximum 20 requests per minute for this OneClick user."}`（**用户级**）
  - `{"message":"Model API rate limit exceeded; please retry later","code":"token_rate_limit_exceeded"}`（**token 级**）
- 结论：**这不是算子逻辑 bug，而是把 GPT-4o 时代的 `max_workers=500` 直接搬到 20 RPM 端点上必然失败**。

### 8.2 修复：进程级 RPM 限流 + 降并发 + 保重试

```python
class ThrottledAPILLMServing_request(APILLMServing_request):
    # 覆写 _api_chat_with_id（HTTP 请求层，比重试层更精确），线程安全匀速放行
    def _throttle(self):
        with self._lock:
            now = time.time(); wait = self._next_slot - now
            if wait > 0: time.sleep(wait); now = time.time()
            self._next_slot = now + self._min_interval; self.n_requests += 1
```
配置 `rpm=12, max_workers=2, max_retries=6, read_timeout=180`，重跑同一份数据。

### 8.3 第二次运行（正式结果）：成功，仅 3 次 429

- 66 次 HTTP 请求中失败 5 次，**全部被 DataFlow `_api_chat_id_retry` 的指数退避吸收**：
  | 类型 | 次数 | 原文 |
  |---|---|---|
  | 429 用户级 20 RPM | 1 | `maximum 20 requests per minute for this OneClick user`（22:41:05） |
  | 429 模型级并发 | 2 | `Model 'DeepSeek-V4-Flash' is at its concurrency limit (448)`（22:43:36/41） |
  | 500 / provider 504 | 2 | `Error from provider self-dploy: 504 Gateway Timeout`（耗时 46.9 s / 107.2 s） |
- 与 pilot 的 40 次 429 相比，**限速把 429 压到 3 次（-92.5%）**，无一行因重试耗尽而丢失。

> 备注：429 属于**共享外部约束**——同端点还有另一个 Text2SQL pipeline 并发（各 2 workers），
> 二者共用 20 RPM 配额，故本次并非独占带宽下的最优点。

## 9. 样例（首行，完整节选）

`id=4864947`（Halloween 4 剧情文本）：

| 字段 | 值（节选） |
|---|---|
| identifier | `{"content_identifier": "Halloween 4: The Return of Michael Myers"}` |
| candidate_tasks_str | `{"conclusion": "Halloween 4: The Return of Michael Myers", "R": "Film title"}` |
| question | What is the film title? |
| answer | Halloween 4: The Return of Michael Myers |
| refined_answer | Halloween 4: The Return of Michael Myers |
| llm_answer | I need more information to identify the film title. Please provide details such as the plot, actors, director, year, or a specific scene. |
| llm_score | 0 |
| golden_doc_answer | Halloween 4: The Return of Michael Myers |
| golden_doc_score | 2 |
| optional_answer | 13 条（含 "Halloween 4"、"Halloween IV"、"H4"、"Halloween 4 (1988)" 等） |
| **F1Score** | **1.0** |

完整样例见 `sample_first_row.json`；4 行全量见 `agentic_rag_task4_step2_FINAL.jsonl`。

## 10. 质量边界（必须如实说明）

1. **F1 评分器是纯 token-level（bag-of-words）**，对答案表述格式极度敏感。真实反例（本产物第 4 行）：
   - `refined_answer = "2017-08-30"`
   - `golden_doc_answer = "August 30, 2017"`
   - 归一化后两者 token 交集为 0 → **F1Score = 0.0**
   - 语义上两者完全等价。→ 该指标适合做 RL 弱奖励/粗筛，**不适合当最终质量判决**；生产上应叠加
     日期/数字归一化或改用 LLM-as-judge / embedding 相似度。
2. **样本量小（n=4）**：均值 0.6957 不具统计意义；本次目的是验证"输入→算子链→11 字段→F1"链路真实可跑通。
3. **算子自带两级过滤会显著收缩数据量**：4 文档 → 8 候选 → 4 行（收缩率 50%）；`llm_score<1`
   专门剔除"无文档即可答对"的题目，这是**有意为之**而非数据丢失。
4. **`golden_doc_score` 是 LLM 自评（1/2 档）**，不是人工标注的 ground truth；其可靠性依赖模型判断。
5. **未做去重/难度分层/多跳扩展**：官方另有 `AgenticRAGDepthQAGenerator` / `WidthQAGenerator`
   与 `agentic_rag_multihop_core_pipeline.py`，本次未运行（时间与配额约束）。
6. **吞吐受外部端点硬限**：20 RPM 共享配额下，本次 6.05 req/min。按此估算全量 10 文档
   （≈165 请求）需 **≈27 分钟**（共享配额）/ **≈15–18 分钟**（独占 20 RPM）。这是数据合成规模的
   真实成本约束：**RAG 数据合成的瓶颈在 LLM 调用数与端点评分速率，不在本地 CPU**。

## 11. 与官方文档的偏差清单

| 文档说法 | 实测 |
|---|---|
| `max_workers=500` 直接可用 | 在 20 RPM 端点上必然 429 风暴（pilot 40 次 429），必须客户端限流 |
| `first_entry_file_name="../example_data/..."` | 相对路径，须 `dataflow init` 到 `run_dataflow/` 才能跑；本任务改绝对路径 |
| 未提及端点限流/重试成本 | 429（用户级 + token 级 + 模型并发级）与 provider 504 是主要长尾延迟来源 |
| 示例默认 `max_per_task=10, max_question=10` | 20 RPM 下不可行（10 文档 ≈ 600+ 调用）；本次 2/1 |
| 文档称 F1 用于"构建 RL 奖励信号" | 实测印证其格式敏感性，宜作弱奖励 |

## 12. 证据文件清单

`courses/10-dataflow/evidence/task4/agentic_rag/`：

| 文件 | 说明 |
|---|---|
| `run_main.log` | 正式运行完整 stdout（含每阶段日志、429/504 每次失败原文、tqdm 进度） |
| `agentic_rag_task4_step1.jsonl` | Step1 输出（原子问答生成，4 行 × 12 列） |
| `agentic_rag_task4_step2_FINAL.jsonl` | **最终输出**（+F1Score，4 行 × 13 列） |
| `field_check_main.json` | 字段完整性/空值率/F1 分布/score 统计（机器可读） |
| `sample_first_row.json` | 首行完整样例（长文本截断标注） |
| `input_main_4docs.jsonl` | 实际输入子集（4 条 `id`/`contents`） |
| `run_pilot_429_evidence.log` | **失败证据**：pilot 阶段 40 次 429 原文（限流问题的直接证据） |

`courses/10-dataflow/code/task4/agentic_rag/`：

| 文件 | 说明 |
|---|---|
| `agentic_rag_task4.py` | 改造后 pipeline（含 `ThrottledAPILLMServing_request`） |
| `run_agentic_rag.sh` | 一键复现（子集构造 → 运行 → 字段校验） |
| `verify_task4_agentic_rag.py` | 字段完整性/F1/空值率检查器（产出 field_check json + sample） |
| `REPORT.md` | 本文件 |
