# DataFlow Task6 打卡 · 从数据到效果——评估模型能力（模型性能评测 Pipeline）

**任务**：dataflowTask6 从数据到效果——评估模型能力：模型性能评测 Pipeline（官方 unified bench eval 六类型全量 + semantic 变体）
**环境**：vm-60（RTX 4090 D 24GB）· open-dataflow 1.0.11 · torch 2.9.1 · 2026-11-02 实测
**代码**：code/task6/（local_hf_serving 适配器 / bench_eval_task6 主管线 / build_eval_data / probe_reference_answers / verify / render_runshots + env.sh + run_all.sh）
**证据**：evidence/task6/（11 个 run 日志 + 逐步 jsonl + eval_result + verify_task6.json + api_calls 审计 + 6 张 runshot figures/runshot_task6_01~06 + 官方 7 个 pipeline 原型拷贝）

## 一句话结论

官方六个评测类型全部在 vm-60 真实跑通（ppl / math_verify / any_math_verify / ll_choice_acc / micro_f1 / pairwise_ll_winrate，外加 LLM 裁判变体），算子/prompt/metric 与官方逐字一致、只换 serving；用同一套算子评了两个真实模型（本地 0.5B vs AMD 端点 DeepSeek-V4-Flash），并用「官方预填答案探针 + 逐 id 命中路径归因」证明：0.5B 的 QA 15/15 里 3 条仅靠子串包含得分、API 模型 13/15 里 2 条是 LaTeX 格式假阴性——**分数对度量口径敏感，跨模型比能力先比口径**。

## 验证了什么

1. **六种 benchmark 数据格式用一套算子评**：`BenchAnswerGenerator`（按 eval_type 决定是否生成）+ `UnifiedBenchDatasetEvaluator`（写回 eval_score/eval_pred/eval_valid/eval_error + 聚合落盘），`key1_text_score`(ppl) / `key2_qa`(math_verify) / `key2_q_ma`(any_math_verify) / `key3_q_choices_a`(ll_choice_acc) / `key3_q_choices_as`(micro_f1) / `key3_q_a_rejected`(pairwise_ll_winrate) 全量实测。
2. **loglikelihood 类指标不依赖 vllm**：评测算子在 serving 没有 ll 接口时按 `real_model_path` 自己用 transformers 前向，单选/偏好/ppl 三个指标因此原样跑通（算子零修改）。
3. **semantic_judge 变体**：同一批生成答案换 LLM 裁判，15/15 vs math_verify 口径 13/15，差集=格式争议题（math_009）。
4. **度量宽松度探针**：官方 type2 自带 15 条预填答案（4 条故意答错）进同一算子 → 11/15，错答案全部判 0；证明指标对明确错误严格、对短标签宽松。

## 改了什么（相对官方 pipeline）

- serving：`LocalModelLLMServing_vllm`（Qwen2.5-7B）→ 自写 `LocalModelLLMServing_hf`（transformers，本地 Qwen2.5-0.5B-Instruct）。原因：open-dataflow[vllm] 锁 vllm<=0.9.2→torch 2.7.0，与现有 torch 2.9.1 冲突，且 24GB 单卡放不下 7B 双副本；适配器只实现 LLMServingABC 三方法 + 暴露 real_model_path。
- 裁判/生成二合一 API：OpenAI gpt-4o → AMD Radeon 兼容端点 DeepSeek-V4-Flash（2026-11-02 实测 vm-60 直连 200，Task5 的隧道方案已不需要；仍保留 10rpm 客户端限流 + 请求审计，61 次请求全部成功）。
- 解码：评测场景贪心（temperature=0）保证可复现；路径全部环境变量化 + run_meta.json 快照。

## 实际结果（数字）

- 本地 0.5B：ppl_mean=88.47（5 条 16.8~249.8）；QA 15/15；多参考 QA 5/6；单选 6/10；多选 micro-F1 0.7214；偏好胜率 8/10。
- API V4-Flash：QA(math_verify) 13/15；QA(semantic_judge) 15/15；多参考 QA 5/6；多选 micro-F1 0.9524。
- 命中路径归因（verify 独立重算）：0.5B 的 15 条 = 12 严格 + 3 子串；API 的 13 条 = 12 严格 + 1 子串；预填答案 11 条 = 9 严格 + 2 子串。
- 假阴性实例：math_002（`25\pi` vs golden `25π`）、math_009（`(1/2)x^2 + C` vs `x²/2 + C`）被判 0，LLM 裁判判 1。
- 核对 11/11 all_ok：聚合对账、逐 id 重算 0 mismatch、ll 抽核 3/3、API 审计 61/61 ok。

## 踩到的坑（都有证据）

1. **open-dataflow 1.0.11 的 vllm extra 与现有 torch 冲突**：`[vllm]` 锁 `vllm<=0.9.2 → torch==2.7.0`，装它会降级 Task3/4/5 共用的 torch 2.9.1；解法是 transformers 适配器（评测算子的 HF 回退路径让 ll/ppl 指标照样工作）。
2. **`UnifiedBenchDatasetEvaluator` 类默认 `prompt_template=AnswerJudgePrompt`（类不是实例）会被 prompt_restrict 拒掉**，必须显式传 `None`（官方管线正是传 None）；直接 `UnifiedBenchDatasetEvaluator(eval_result_path=...)` 会 TypeError。
3. **NFS 加载模型 91s**（/mnt/nfs 93% 满）：模型复制到本地盘后加载 ~1s，11 个 run 省下 ~10 分钟。
4. **math_verify 口径的双向子串**会把短标签放大（抽取答案 `2` 是 golden `x²/2 + C` 归一化串的子串 → 判对）；跨模型比分数前必须先做命中路径归因，否则把度量宽松度读成模型能力。
5. **多选 prompt 要求输出 JSON，两个模型都不遵守**（输出 `A. 2 and C. 11` 式自然语言），靠评测算子的字母集合 fallback 救分；0.5B 系统性"全选"（6/10 条预测=全部选项），precision 把 micro-F1 压到 0.667。
6. **ll 偏好长答案**：偏好对比里 pw_0006/pw_0010 的 better 更短更正式，ll_better < ll_rejected 直接判负——ll winrate 不能直接当人类偏好代理。

## 下一步修正

- 样本量扩大到真实 benchmark（GSM8K/MATH/MMLU 同格式 jsonl，算子不用改）再谈能力结论；报告分数必须附命中路径归因与 semantic 对照。
- golden 标签统一 LaTeX 或 Unicode（或在评测前过一层格式归一化），消除 `π/\pi`、`x²/x^2` 类假阴性。
- 多选把"是否遵守 JSON 指令"单独计一个遵循度指标，与 micro-F1 分开报告；API serving 需补 loglikelihood 接口才能覆盖 ll 类指标。
