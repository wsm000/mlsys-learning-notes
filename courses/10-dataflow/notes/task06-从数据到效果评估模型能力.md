# Task6 从数据到效果——评估模型能力（模型性能评测 Pipeline）

> 课程：DataFlow · AI-Ready Data 实战（10-dataflow，Task 6）
> 环境：vm-60（Linux x86_64，RTX 4090 D 24GB，Python 3.10.12）· open-dataflow 1.0.11 · 2026-11-02 实测
> 本任务产出：官方 unified bench eval 六个评测类型全量实测（11 个 run）+ 度量宽松度探针 + 跨模型能力对比 · [代码](../code/task6) · [证据](../evidence/task6) · [打卡](../evidence/task6/dataflow-task6-checkin.md)

## 0. 结论速览

- **官方六个评测类型全部真实跑通**（`key1_text_score` / `key2_qa` / `key2_q_ma` / `key3_q_choices_a` / `key3_q_choices_as` / `key3_q_a_rejected`），外加 `key2_qa` 的 semantic_judge 变体；指标、算子、prompt、storage 与官方 `gpu_pipelines/benchmark_eval/unified_bench_eval_type{1..6}.py` 逐字一致，**只换了 serving**。
- **同一套算子评两个真实模型**：本地 Qwen2.5-0.5B-Instruct（transformers serving，全离线）与 AMD 端点 DeepSeek-V4-Flash（官方 API serving）。共享类型上：多选 API 明显占优（micro-F1 0.952 vs 0.721），多参考 QA 打平（5/6 vs 5/6，错同一条），单参考 QA 的 15/15 vs 13/15 是度量假阴性主导的反转（见发现 1/2）；单选/偏好类只有本地模型能跑（API 没有 loglikelihood 接口）。
- **三个可复现的度量口径发现**（都有 id 级证据）：
  1. `math_verify` 路径 = 抽答案(math_verify) **OR 双向子串包含**。0.5B 拿到 15/15，其中 3 条仅靠子串包含得分（如 gt=`(-1, 0)`、抽取答案 `0`）；官方自带 15 条预填答案（4 条故意答错）实测 11/15，4 条错答案全部判 0——**指标对明确错误严格，对短标签宽松，"满分"不等于"全对"**。
  2. **LaTeX/Unicode 格式差会造成假阴性**：API 模型答 `25\pi`（golden `25π`）、`(1/2)x^2 + C`（golden `x²/2 + C`）都被判 0；同一道题 LLM 裁判判 1。分数对书写格式敏感，不能直接当能力结论。
  3. **多选 prompt 要求输出 JSON，两个模型都不遵守**，靠评测算子的「字母集合」fallback 才拿到分；0.5B 倾向把所有选项都选上（precision 0.5），micro-F1 由此被拉低。
- **核对不信任算子输出**：`verify_task6.py` 逐 id 独立重算（含算子同款三路径归因）、聚合对账、ll 抽核（3/3 与算子 argmax 一致）、API 审计（61 次请求全部 ok），11/11 run `all_ok=True`。

## 1. 任务是什么：官方 benchmark_eval 管线

官方原型（vm-60 上 `dataflow/statics/pipelines/gpu_pipelines/benchmark_eval/`，已原样拷贝到 `evidence/task6/data/official_pipelines/`）：

| 文件 | eval_type | 默认 metric | 数据 | 机理 |
|---|---|---|---|---|
| `unified_bench_eval_type1.py` | `key1_text_score` | `ppl` | 5 条文本 | 不需要生成，直接算文本困惑度 |
| `unified_bench_eval_type2.py` | `key2_qa` | `math_verify` | 15 条数学 QA | 生成→抽答案→与 golden 比对 |
| `unified_bench_eval_type3.py` | `key2_q_ma` | `any_math_verify` | 6 条多参考 QA | 任一参考命中即对 |
| `unified_bench_eval_type4.py` | `key3_q_choices_a` | `ll_choice_acc` | 10 道单选 | 对每个选项 continuation 算 loglikelihood，argmax 为预测（默认不生成） |
| `unified_bench_eval_type5.py` | `key3_q_choices_as` | `micro_f1` | 10 道多选 | 生成→解析选项集合→算 micro-F1 |
| `unified_bench_eval_type6.py` | `key3_q_a_rejected` | `pairwise_ll_winrate` | 10 组偏好 | better/rejected 两条 continuation 的 ll 直接比大小 |
| `unified_bench_eval_type_semantic.py` | `key2_qa` + `use_semantic_judge=True` | `semantic_judge` | 同上 | 生成答案交给 LLM 裁判判对错 |

两个算子：`BenchAnswerGenerator`（统一生成，`_need_generation()` 决定哪些类型需要生成）与 `UnifiedBenchDatasetEvaluator`（统一评测，写回 `eval_score/eval_pred/eval_valid/eval_error` 四列 + 聚合统计落盘 `eval_result.jsonl`）。这套设计把「六种 benchmark 数据格式」归一成「一种算子 + eval_type 开关」，正是"从数据到效果"的最后一环：**前面 Task1-5 做的数据，最终要以模型效果验收**。

## 2. 环境与改造点（相对官方只换 serving）

| 项 | 官方 | 本任务 | 原因 |
|---|---|---|---|
| 生成 serving | `LocalModelLLMServing_vllm`（Qwen2.5-7B-Instruct） | `LocalModelLLMServing_hf`（本地 Qwen2.5-0.5B-Instruct，transformers 直写，见 `code/task6/local_hf_serving.py`） | open-dataflow 1.0.11 的 `[vllm]` extra 锁 `vllm<=0.9.2 → torch==2.7.0`，与 vm-60 现有 torch 2.9.1 冲突（强降会破坏 Task3/4/5 共用环境）；24GB 单卡也放不下 7B 双副本 |
| 裁判 serving | `gpt-4o`（OpenAI） | 同一 serving（本地）或 AMD 端点 `DeepSeek-V4-Flash`（API run） | 无 OpenAI key；AMD 端点为 Task4/5 同款 |
| 网络 | 直连 | **2026-11-02 实测 vm-60 → developer.amd.com.cn 直连 200**（Task5 时期的劫持/Windows CONNECT 隧道已不需要）；端点仍是 20 req/min 用户级限流，客户端 10rpm 匀速放行（Task4/5 同款 ThrottledLogging 包装，请求/响应落盘 `api_calls.jsonl` 不含密钥） | 环境变化，如实更新 |
| 解码 | vllm 默认 temperature=0.7 | 评测场景贪心（temperature=0） | 可复现；API serving 官方默认即 0.0 |
| 路径 | 硬编码 | `T6_INPUT/T6_CACHE/T6_PREFIX` 环境变量 + `run_meta.json` 配置快照 | 一键复现 |

适配器只实现 `LLMServingABC` 三方法（`generate_from_input/start_serving/cleanup`）并暴露 `real_model_path`——评测算子的 `_ll_batch/_ppl_batch` 在 serving 没有 loglikelihood 接口时，会**按 `real_model_path` 自己用 transformers 加载模型**，所以 `ll_choice_acc`/`pairwise_ll_winrate`/`ppl` 三个指标原样跑通，算子一行未改。

两个环境坑记录：① 模型从 NFS（93% 满）加载一次 91s，复制到本地盘后 ~1s；② `UnifiedBenchDatasetEvaluator` 的类默认 `prompt_template=AnswerJudgePrompt`（类而非实例）会被 `prompt_restrict` 拒掉，必须显式传 `None`（官方管线正是传 None）。

## 3. 实测结果（11 个 run，vm-60 真实执行）

数据集全部用官方 example 原样（`dataflow/example/core_text_data/unified_bench_eval_type{1..6}.jsonl`，sha256 见 `evidence/task6/data/manifest.json`），未做任何增删。

| run | eval_type / metric | 模型 | 结果 | 耗时 |
|---|---|---|---|---|
| t1_local | key1_text_score / ppl | Qwen2.5-0.5B-Instruct | ppl_mean=**88.47**（16.8~249.8，5 条） | 2.6s |
| t2_local | key2_qa / math_verify | 同上 | acc=**1.0000**（15/15） | 48.3s |
| t3_local | key2_q_ma / any_math_verify | 同上 | acc=**0.8333**（5/6） | 8.3s |
| t4_local | key3_q_choices_a / ll_choice_acc | 同上 | acc=**0.6000**（6/10） | 2.0s |
| t5_local | key3_q_choices_as / micro_f1 | 同上 | micro_f1=**0.7214** | 47.5s |
| t6_local | key3_q_a_rejected / pairwise_ll_winrate | 同上 | winrate=**0.8000**（8/10） | 2.0s |
| t2_api | key2_qa / math_verify | DeepSeek-V4-Flash | acc=**0.8667**（13/15） | 112.7s |
| t2s_api | key2_qa / **semantic_judge** | 同上 | acc=**1.0000**（15/15） | 387.7s |
| t3_api | key2_q_ma / any_math_verify | 同上 | acc=**0.8333**（5/6） | 31.4s |
| t5_api | key3_q_choices_as / micro_f1 | 同上 | micro_f1=**0.9524** | 63.4s |
| t2_ref_local | key2_qa / math_verify（**预填答案探针**） | —（不生成） | acc=**0.7333**（11/15） | 0.8s |

跨模型对比（同数据同算子，只换 serving）：

| eval_type | 0.5B 本地 | V4-Flash API | 差距 |
|---|---|---|---|
| key2_qa (math_verify) | 1.0000 | 0.8667 | -0.1333 |
| key2_q_ma | 0.8333 | 0.8333 | 0 |
| key3_q_choices_as (micro_f1) | 0.7214 | 0.9524 | +0.2310 |

注意第一行是**度量假阴性主导的反转**：0.5B 的 15/15 里有 3 条仅靠子串包含（见 §4.1），API 的 13/15 里 2 条是 LaTeX 格式假阴性（见 §4.2）——同一把尺子在不同错误类型上灵敏度不同，**跨模型比分数先比口径**。

### 3.1 各类型逐 id 明细（摘）

- **key1_text_score（ppl）**：5 条文本 16.8~249.8。"This is a simple test sentence to measure perplexity..."（249.8）远高于"The capital of France is Paris."（16.8）——ppl 对文本"常见度"敏感，可用来给语料打分（Task5 过滤思路的效果侧对应物）。
- **key3_q_choices_a（ll_choice_acc）**：10 题对 6 题。错的是 mc_0002（`len([1,2,3])` 选 2 而非 3）、mc_0006/0007/0010。ll 抽核 3/3 与算子 argmax 一致（独立前向重算，见 §5）。
- **key3_q_choices_as（micro_f1）**：0.5B 的 10 条里 6 条预测集合 = 全部选项（precision 0.5/recall 1.0/F1 0.667），1 条完全正确；API 模型 9/10 条精确匹配，仅 ms_0003 输出整段解释被字母解析成全选。
- **key3_q_a_rejected（pairwise）**：8/10。输的 pw_0006/pw_0010 都是 better 答案更短/更正式（ll_better=-69.3 vs -32.4），**ll 偏好"长而口语"的劣选答案**——直接用 ll winrate 当人类偏好代理是有偏的。
- **key2_q_ma**：两个模型同样错 ma_0005（光速，targets 都带单位，模型答 `186282` 不带单位 → 无参考命中）。

## 4. 度量口径的三个发现（都有 id 级证据）

### 4.1 math_verify 路径 = 严格等价 OR 双向子串包含（宽松度探针）

算子 `_eval_qa_single` 的判定是三条路径取或：`math_verify(抽答案, gt)`、`包含(抽答案, gt)`、`包含(原文, gt)`，其中"包含"是**双向子串**（`r in p or p in r`，归一化 NFKC/下标/去尾标点/casefold 之后）。`verify_task6.py` 复现三路径并逐 id 归因：

| run | 判对条数 | math_verify 严格 | 仅子串包含 |
|---|---|---|---|
| t2_local（0.5B 生成） | 15 | 12 | 3（2 原文 + 1 抽取答案） |
| t2_api（API 生成） | 13 | 12 | 1 |
| t2_ref_local（官方预填答案） | 11 | 9 | 2 |

典型：math_009（golden `x²/2 + C`），0.5B 答案写 `(1/2)x^2 + C`，抽取答案落到 `2`，而 `2` 是 golden 归一化串 `x2/2 + c` 的子串 → 判对。**短数字标签（0/1/2/9）在这种规则下几乎必对**。

附带发现：多参考 QA（key2_q_ma）两个模型的 5 条判对**全部**走子串包含、0 条走 math_verify（t3_local: 4 原文 + 1 抽取答案；t3_api: 5 原文）——常识问答的答案本就是句子级包含关系，严格数学等价几乎不触发；这类 benchmark 实际由"包含规则"主导。

探针实验（`code/task6/probe_reference_answers.py`）：官方 type2 数据自带 15 条预填 `model_answer`，其中 math_006（顶点）、math_008（log₂8）、math_012（sin90°）、math_014（7×8）四条是故意答错的。把该列复制为 `generated_ans` 直接进同一评测算子 → **11/15，4 条错答案全部判 0**。即：指标对"明确答错"是严格的；15/15 的生成答案里那 3 条子串命中才是宽松所在。**报告分数时必须同时报告命中路径归因**，否则会把度量宽松度读成模型能力。

### 4.2 LaTeX/Unicode 格式差 = 假阴性

t2_api 两条判 0 全是格式问题而非答错：

- math_002：golden `25π`，模型答 `...25\pi...`（抽取答案没落到 `25\pi`，原文也不含 Unicode π）→ 0；**同题在 semantic run 里裁判判 1**。
- math_009：golden `x²/2 + C`，模型答 `(1/2)x^2 + C` → 0；**judge_pass_but_strict_fail=[math_009]**（verify 的口径差异检查逐 id 列出）。

即 math_verify 路径对 `π/²/₂` 与 `\pi/^2/_2` 的互化不完全；LLM 裁判对格式鲁棒。**结论：要审"格式敏感性"就跑一次 semantic 对照，两个口径的差集就是格式造成的争议题。**

### 4.3 多选：prompt 要 JSON，模型不给，fallback 救命

type5 的 prompt 明确要求 `Output JSON only, format: {"choices": ["A","C"]}`，但两个模型的真实输出都是自然语言（`A. 2 and C. 11`、`A, B, D`）。评测算子的 `_parse_multiselect_set` 依次尝试 JSON → 字母集合 → 数字集合，字母解析救回了绝大多数。0.5B 的系统性行为是"全选"（6/10 条预测=全部选项），micro-F1 的 precision 项直接把分数压到 0.667；API 模型 9/10 精确。**这条链路上 prompt 遵循度本身就是被评测的能力**，而指标设计（先 JSON 后字母）决定了"不遵守指令"会被部分救分而非全判 0。

## 5. 核对方法（verify_task6.py，11/11 all_ok=True）

不信任算子输出，每个 run 独立核对：

1. **聚合对账**：`eval_result.jsonl` 的 total/valid/accuracy(或 ppl_mean/micro_f1_mean) 与最终 step jsonl 的 `eval_score` 列独立聚合一致（11/11）。
2. **指标重算**：key2_qa/key2_q_ma 用自实现归一化 + math_verify 复现三路径逐 id 比对（11/11 一致，0 mismatch）；key3_q_choices_as 从 `eval_pred` 的 pred_set/gold_set 独立重算 micro-F1；key3_q_a_rejected 从 ll_better/ll_rejected 独立重算胜者；key1 检查 ppl 有限且 >0。
3. **ll 抽核**（`--ll-spotcheck 3`）：用 transformers 独立前向（无 padding、单条）重算 mc_0001~0003 四个选项的 loglikelihood，argmax 与算子一致 3/3。
4. **完整性**：id 唯一、生成类类型 `generated_ans` 非空、`eval_error` 分布。
5. **API 审计**：4 个 API run 共 61 次请求全部 ok，平均延迟落盘。
6. **semantic run 特殊处理**：裁判口径与 math_verify 不是同一把尺子，不做同口径 pass/fail，改为逐 id 列出 `judge_pass_but_strict_fail`（=格式争议题）。

## 6. 复现

```bash
# vm-60
source ~/dataflow-task6/task6/env.sh          # 端点/密钥(只从 key 文件进进程)/模型路径/限流
cd ~/dataflow-task6/task6
bash run_all.sh                                # 全量：数据集 + 10 个 bench run + 探针 + 核对
bash run_all.sh local                          # 只跑本地 6 类型 + 探针
bash run_all.sh api                            # 只跑 API 4 run
```

产物：每个 run 一个 `cache_*` 目录（`*_step1/2.jsonl` 逐步数据、`eval_result/eval_result.jsonl` 聚合、`run_meta.json` 配置快照、`api_calls.jsonl` 审计），核对输出 `verify_task6.json`；本仓库 [evidence/task6](../evidence/task6) 有全量归档（含 6 张终端风格 runshot，由 `code/task6/render_runshots.py` 从真实日志渲染，非手工截图）。

## 7. 边界与后续

- **样本量小**（官方 example 5~15 条/类型），所有数字只用于验证管线行为与口径，不外推为模型能力结论；要出能力结论需换真实 benchmark（如 GSM8K/MATH/MMLU 同格式 jsonl，算子不用改）。
- **API serving 没有 loglikelihood 接口**，`key3_q_choices_a`/`key3_q_a_rejected` 两个 ll 类指标只能本地模型跑（API run 会 `ll_unavailable` 全判 invalid）——这是管线对 serving 能力的硬依赖，选 serving 时要按指标反推。
- **ppl/ll 都走评测算子内置的 transformers 回退路径**（serving 只提供模型路径），模型被加载两次（生成一份 + 评测一份）；0.5B 无感，大模型需注意显存。
- 官方 `Qwen/Qwen2.5-7B-Instruct` 未跑（vllm 不可装 + 24GB 放不下）；如要跑官方原样配置，需独立 venv 装 `vllm==0.9.2 + torch 2.7.0`（与现有环境隔离），本任务未走该路线，已在 §2 说明。
