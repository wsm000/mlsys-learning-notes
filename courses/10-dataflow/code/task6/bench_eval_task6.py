#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DataFlow Task6 — 模型性能评测 Pipeline（官方 unified bench eval 6 类型全量，vm-60 实测）

官方原型：dataflow/statics/pipelines/gpu_pipelines/benchmark_eval/unified_bench_eval_type{1..6}.py
          (+ unified_bench_eval_type_semantic.py 的 use_semantic_judge=True 变体)

六个评测类型（eval_type）与默认指标：
  key1_text_score      ppl                  文本困惑度（不需要生成）
  key2_qa              math_verify          单参考问答（生成 + 抽答案 + 比对）
  key2_qa(+semantic)   semantic_judge       单参考问答（LLM 裁判）
  key2_q_ma            any_math_verify      多参考问答（任一参考命中即对）
  key3_q_choices_a     ll_choice_acc        单选（对每个选项算 loglikelihood，取 argmax）
  key3_q_choices_as    micro_f1             多选（解析模型输出集合，算 micro-F1）
  key3_q_a_rejected    pairwise_ll_winrate  偏好对比（better/rejected 的 ll 胜率）

相对官方的改造点（算子 / prompt / metric / storage 用法完全一致，只换 serving 与路径）：
  1) serving：本地跑用 LocalModelLLMServing_hf（transformers 直写，见 local_hf_serving.py，
     原因是 open-dataflow[vllm] 锁 vllm<=0.9.2 -> torch 2.7.0，与 vm-60 现有 torch 2.9.1 冲突）；
     API 跑用官方 APILLMServing_request（AMD Radeon 兼容端点 DeepSeek-V4-Flash），
     加进程内匀速限流 + 请求审计（与 Task4/5 同策略）。
  2) 输入/缓存/前缀/结果路径全部走环境变量（T6_INPUT/T6_CACHE/T6_PREFIX）。
  3) 每步打印 in/out 行数与耗时；run_meta.json 落盘本次配置快照。
  4) key2_qa 额外支持 --semantic-judge：裁判换成 llm_serving（默认同一 serving）。

用法：
  T6_MODEL=local python bench_eval_task6.py --eval-type key2_qa --model local
  T6_MODEL=api   python bench_eval_task6.py --eval-type key2_qa --model api --semantic-judge
"""
import argparse
import json
import os
import threading
import time

from dataflow.operators.core_text import BenchAnswerGenerator, UnifiedBenchDatasetEvaluator
from dataflow.prompts.core_text import FormatStrPrompt
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage

from local_hf_serving import LocalModelLLMServing_hf


def env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


class ThrottledLoggingAPILLMServing_request(APILLMServing_request):
    """官方 APILLMServing_request + 进程内匀速限流 + 请求/响应审计（Task4/5 同款）。

    AMD 端点对同一用户限流 20 requests/minute，官方默认 max_workers 会瞬间打满触发 429。
    每 60/rpm 秒放行一个请求，让 DataFlow 自带退避重试只需处理偶发冲突。
    """

    def __init__(self, *args, **kwargs):
        rpm = int(kwargs.pop("rpm", 10))
        self.log_path = kwargs.pop("log_path", None)
        super().__init__(*args, **kwargs)
        self._min_interval = 60.0 / max(1, rpm)
        self._lock = threading.Lock()
        self._next_slot = 0.0
        self.n_requests = 0
        self._log_lock = threading.Lock()
        print("[THROTTLE] client-side rpm=%d (interval=%.2fs)" % (rpm, self._min_interval), flush=True)

    def _throttle(self):
        with self._lock:
            now = time.time()
            wait = self._next_slot - now
            if wait > 0:
                time.sleep(wait)
                now = time.time()
            self._next_slot = now + self._min_interval
            self.n_requests += 1

    def _audit(self, record):
        if not self.log_path:
            return
        with self._log_lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _api_chat_with_id(self, id, payload, model, is_embedding=False, json_schema=None):
        self._throttle()
        t0 = time.time()
        result = super()._api_chat_with_id(id, payload, model, is_embedding, json_schema)
        dt = time.time() - t0
        resp_text = result[1] if result else None
        try:
            user_msg = payload[-1].get("content", "") if isinstance(payload, list) and payload else str(payload)
        except Exception:
            user_msg = "<unparsable>"
        self._audit({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "id": id,
            "model": model,
            "latency_s": round(dt, 2),
            "prompt_tail": user_msg[-400:],
            "response": resp_text,
            "ok": resp_text is not None,
        })
        return result


# 官方每个类型的 prompt 模板与 run() 键映射（逐字对齐 unified_bench_eval_type*.py）
PROMPT_QA = "Question: {question}\nAnswer:"
PROMPT_CHOICES = "{context}Question:\n{question}\n\nChoices:\n{choices_text}\n\nAnswer:"
PROMPT_PAIRWISE = "{context}Question:\n{question}\n\nAnswer:"

TYPE_CONFIG = {
    "key1_text_score": {
        "prompt": None,
        "gen_keys": dict(input_text_key="text", input_context_key=None, output_key="generated_ans"),
        "eval_keys": dict(input_text_key="text", input_context_key=None, input_pred_key="generated_ans"),
        "default_metric": "ppl",
    },
    "key2_qa": {
        "prompt": PROMPT_QA,
        "gen_keys": dict(input_question_key="question", input_target_key="golden_label",
                         input_context_key=None, output_key="generated_ans"),
        "eval_keys": dict(input_question_key="question", input_target_key="golden_label",
                          input_context_key=None, input_pred_key="generated_ans"),
        "default_metric": "math_verify",
    },
    "key2_q_ma": {
        "prompt": PROMPT_QA,
        "gen_keys": dict(input_context_key="context", input_question_key="question",
                         input_targets_key="targets", output_key="generated_ans"),
        "eval_keys": dict(input_context_key="context", input_question_key="question",
                          input_targets_key="targets", input_pred_key="generated_ans"),
        "default_metric": "any_math_verify",
    },
    "key3_q_choices_a": {
        "prompt": PROMPT_CHOICES,
        "gen_keys": dict(input_context_key="context", input_question_key="question",
                         input_choices_key="choices", output_key="generated_ans"),
        "eval_keys": dict(input_context_key="context", input_question_key="question",
                          input_choices_key="choices", input_label_key="label",
                          input_pred_key="generated_ans"),
        "default_metric": "ll_choice_acc",
    },
    "key3_q_choices_as": {
        "prompt": PROMPT_CHOICES,
        "gen_keys": dict(input_context_key="context", input_question_key="question",
                         input_choices_key="choices", output_key="generated_ans"),
        "eval_keys": dict(input_context_key="context", input_question_key="question",
                          input_choices_key="choices", input_labels_key="labels",
                          input_pred_key="generated_ans"),
        "default_metric": "micro_f1",
    },
    "key3_q_a_rejected": {
        "prompt": PROMPT_PAIRWISE,
        "gen_keys": dict(input_context_key="context", input_question_key="question",
                         output_key="generated_ans"),
        "eval_keys": dict(input_context_key="context", input_question_key="question",
                          input_better_key="better", input_rejected_key="rejected",
                          input_pred_key="generated_ans"),
        "default_metric": "pairwise_ll_winrate",
    },
}


def count_lines(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def build_serving(model_kind, cache_path):
    """model_kind: local | api"""
    if model_kind == "local":
        return LocalModelLLMServing_hf(
            hf_model_name_or_path=os.environ.get("T6_MODEL_PATH") or "/mnt/nfs/models/Qwen/Qwen2.5-0.5B-Instruct",
            hf_cache_dir=os.environ.get("HF_HOME") or None,
            hf_temperature=float(os.environ.get("T6_TEMPERATURE", "0")),
            hf_max_tokens=env_int("T6_MAX_TOKENS", 512),
            hf_max_model_len=env_int("T6_MAX_MODEL_LEN", 2048),
        )
    api_base = (os.environ.get("AMD_API_BASE") or "https://developer.amd.com.cn/radeon/api/v1").rstrip("/")
    return ThrottledLoggingAPILLMServing_request(
        api_url=api_base + "/chat/completions",
        model_name=os.environ.get("DF_MODEL") or "DeepSeek-V4-Flash",
        max_workers=env_int("T6_MAX_WORKERS", 2),
        max_retries=env_int("T6_MAX_RETRIES", 6),
        connect_timeout=10.0,
        read_timeout=180.0,
        rpm=env_int("T6_RPM", 10),
        log_path=os.path.join(cache_path, "api_calls.jsonl"),
    )


def limit_rows(in_path, max_samples, cache_path):
    """官方数据集原样跑；--max-samples 只用于冒烟，截断后落盘 step0 供复现"""
    if not max_samples or max_samples <= 0:
        return in_path
    rows = []
    with open(in_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            line = line.strip()
            if line:
                rows.append(line)
    out = os.path.join(cache_path, "input_limited.jsonl")
    os.makedirs(cache_path, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"[LIMIT] max_samples={max_samples} -> {out} ({len(rows)} rows)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-type", required=True, choices=sorted(TYPE_CONFIG))
    ap.add_argument("--model", required=True, choices=["local", "api"])
    ap.add_argument("--semantic-judge", action="store_true",
                    help="key2_qa 专用：use_semantic_judge=True，裁判用 llm_serving")
    ap.add_argument("--force-generate", action="store_true",
                    help="传给 BenchAnswerGenerator.force_generate（key3_q_choices_a 默认不生成）")
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    cfg = TYPE_CONFIG[args.eval_type]
    if args.semantic_judge and args.eval_type != "key2_qa":
        ap.error("--semantic-judge 只支持 key2_qa")

    input_path = os.environ.get("T6_INPUT")
    cache_path = os.environ.get("T6_CACHE") or "./cache_bench"
    prefix = os.environ.get("T6_PREFIX") or "bench_task6"
    os.makedirs(cache_path, exist_ok=True)

    input_path = limit_rows(input_path, args.max_samples, cache_path)

    storage = FileStorage(
        first_entry_file_name=input_path,
        cache_path=cache_path,
        file_name_prefix=prefix,
        cache_type="jsonl",
    )

    serving = build_serving(args.model, cache_path)
    metric = cfg["default_metric"] if not args.semantic_judge else "semantic_judge"

    prompt_template = FormatStrPrompt(f_str_template=cfg["prompt"]) if cfg["prompt"] else None

    gen_step = BenchAnswerGenerator(
        llm_serving=serving,
        eval_type=args.eval_type,
        prompt_template=prompt_template,
        allow_overwrite=False,
        force_generate=args.force_generate,
    )
    eval_result_path = os.path.join(cache_path, "eval_result", "eval_result.jsonl")
    os.makedirs(os.path.dirname(eval_result_path), exist_ok=True)
    eval_step = UnifiedBenchDatasetEvaluator(
        eval_result_path=eval_result_path,
        llm_serving=serving,
        eval_type=args.eval_type,
        prompt_template=None,
        use_semantic_judge=args.semantic_judge,
        metric_type=None,  # use default metric
    )

    meta = {
        "eval_type": args.eval_type,
        "model_kind": args.model,
        "model": os.environ.get("T6_MODEL_PATH") if args.model == "local" else os.environ.get("DF_MODEL"),
        "semantic_judge": bool(args.semantic_judge),
        "force_generate": bool(args.force_generate),
        "metric": metric,
        "prompt_template": cfg["prompt"],
        "input": input_path,
        "rows": count_lines(input_path),
        "cache": cache_path,
        "prefix": prefix,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(os.path.join(cache_path, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[CONFIG] " + json.dumps(meta, ensure_ascii=False), flush=True)

    t0 = time.time()
    serving.start_serving()
    print(f"[INIT] serving ready ({time.time()-t0:.1f}s)", flush=True)

    def run_op(name, fn, **kwargs):
        step_storage = storage.step()
        n = step_storage.operator_step
        in_path = input_path if n == 0 else os.path.join(cache_path, f"{prefix}_step{n}.jsonl")
        out_path = os.path.join(cache_path, f"{prefix}_step{n + 1}.jsonl")
        before = count_lines(in_path)
        t = time.time()
        fn(storage=step_storage, **kwargs)
        after = count_lines(out_path)
        print(f"[STEP] {name:28s} {str(before):>4s} -> {str(after):>4s}  ({time.time()-t:6.1f}s)", flush=True)

    run_op("BenchAnswerGenerator", gen_step.run, **cfg["gen_keys"])
    run_op("UnifiedBenchDatasetEvaluator", eval_step.run, **cfg["eval_keys"])

    if os.path.exists(eval_result_path):
        with open(eval_result_path, encoding="utf-8") as f:
            stats = json.load(f)
        print("[RESULT] " + json.dumps(stats, ensure_ascii=False), flush=True)

    nreq = getattr(serving, "n_requests", 0)
    print(f"[DONE] total {time.time()-t0:.1f}s requests={nreq}", flush=True)
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["total_s"] = round(time.time() - t0, 1)
    meta["requests"] = nreq
    with open(os.path.join(cache_path, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
