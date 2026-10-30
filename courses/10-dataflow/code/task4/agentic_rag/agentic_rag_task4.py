#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DataFlow Task4 — AgenticRAG 数据合成 pipeline（改造版，vm-60 实测）

改造点（相对官方 dataflow/statics/pipelines/api_pipelines/agentic_rag_pipeline.py）：
  1) first_entry_file_name 从 "../example_data/..."（相对路径，官方要求先 dataflow init 到 run_dataflow/）
     改为安装包内绝对路径，可用 T4_INPUT 覆盖；
  2) cache_path 改为 "./cache_<profile>"（T4_CACHE），file_name_prefix 改为 agentic_rag_task4；
  3) LLM 端点改为 AMD Radeon OpenAI 兼容端点 + DeepSeek-V4-Flash：
     max_workers 500 -> 2，read_timeout 120 -> 180，max_retries -> 6；
  4) **新增客户端限流**（ThrottledAPILLMServing_request）：
     该端点对 OneClick 用户是 20 requests/minute 的全局限流（官方 pipeline 的 500 并发会直接 429 风暴），
     因此按 T4_RPM（默认 12）在进程内做全局匀速放行，把并发压力降到限流线以下。
  5) 暴露 max_per_task / max_question 控制候选任务数与每文档问题数，用于控制 API 调用量。

原始算子与本步数据流完全未改：
  AgenticRAGAtomicTaskGenerator : contents -> identifier/candidate_tasks_str/question/answer/
                                  refined_answer/llm_answer/llm_score/golden_doc_answer/golden_doc_score/optional_answer
  AgenticRAGQAF1SampleEvaluator : refined_answer vs golden_doc_answer -> F1Score
"""
import os
import sys
import threading
import time
import traceback

from dataflow.operators.agentic_rag import (
    AgenticRAGAtomicTaskGenerator,
    AgenticRAGQAF1SampleEvaluator,
)
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage

EXAMPLE_EVAL_DATA = (
    "/home/simin/.venv-df3/lib/python3.10/site-packages/dataflow/"
    "example/AgenticRAGPipeline/eval_test_data.jsonl"
)


class ThrottledAPILLMServing_request(APILLMServing_request):
    """在官方 APILLMServing_request 之上加一个进程内匀速限流器。

    AMD 端点报错原文：
      {"error":{"message":"Rate limit exceeded: maximum 20 requests per minute for this OneClick user."}}
    官方 pipeline 默认 max_workers=500，会瞬间打满并触发 429 风暴；这里改成
    「每 60/rpm 秒放行一个请求」，让 DataFlow 自带的退避重试只需处理偶发冲突。
    """

    def __init__(self, *args, **kwargs):
        rpm = int(kwargs.pop("rpm", 12))
        super().__init__(*args, **kwargs)
        self._min_interval = 60.0 / max(1, rpm)
        self._lock = threading.Lock()
        self._next_slot = 0.0
        self.n_requests = 0
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

    def _api_chat_with_id(self, *args, **kwargs):
        self._throttle()
        return super()._api_chat_with_id(*args, **kwargs)


def env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


class AgenticRAGEval_APIPipeline(object):
    def __init__(self):
        self.input_path = os.environ.get("T4_INPUT") or EXAMPLE_EVAL_DATA
        self.cache_path = os.environ.get("T4_CACHE") or "./cache"
        self.prefix = os.environ.get("T4_PREFIX") or "agentic_rag_task4"

        os.makedirs(self.cache_path, exist_ok=True)

        self.storage = FileStorage(
            first_entry_file_name=self.input_path,
            cache_path=self.cache_path,
            file_name_prefix=self.prefix,
            cache_type="jsonl",
        )

        api_base = (
            os.environ.get("AMD_API_BASE") or "https://developer.amd.com.cn/radeon/api/v1"
        ).rstrip("/")
        self.llm_serving = ThrottledAPILLMServing_request(
            api_url=api_base + "/chat/completions",
            model_name=os.environ.get("DF_MODEL") or "DeepSeek-V4-Flash",
            max_workers=env_int("T4_MAX_WORKERS", 2),
            max_retries=env_int("T4_MAX_RETRIES", 6),
            connect_timeout=10.0,
            read_timeout=180.0,
            rpm=env_int("T4_RPM", 12),
        )

        self.max_per_task = env_int("T4_MAX_PER_TASK", 2)
        self.max_question = env_int("T4_MAX_QUESTION", 1)

        self.task_step1 = AgenticRAGAtomicTaskGenerator(
            llm_serving=self.llm_serving,
            max_per_task=self.max_per_task,
            max_question=self.max_question,
        )
        self.task_step2 = AgenticRAGQAF1SampleEvaluator()

        print("[CONFIG] input        =", self.input_path, flush=True)
        print("[CONFIG] cache_path   =", self.cache_path, flush=True)
        print("[CONFIG] api_url      =", api_base + "/chat/completions", flush=True)
        print("[CONFIG] model        =", self.llm_serving.model_name, flush=True)
        print("[CONFIG] max_workers  =", self.llm_serving.max_workers, flush=True)
        print("[CONFIG] max_retries  =", self.llm_serving.max_retries, flush=True)
        print("[CONFIG] max_per_task =", self.max_per_task, flush=True)
        print("[CONFIG] max_question =", self.max_question, flush=True)

    def forward(self):
        t0 = time.time()
        print("[STEP1] AgenticRAGAtomicTaskGenerator start", flush=True)
        self.task_step1.run(storage=self.storage.step(), input_key="contents")
        t1 = time.time()
        print("[STEP1] done in %.1fs, llm_requests=%d" % (t1 - t0, self.llm_serving.n_requests), flush=True)

        print("[STEP2] AgenticRAGQAF1SampleEvaluator start", flush=True)
        self.task_step2.run(
            storage=self.storage.step(),
            output_key="F1Score",
            input_prediction_key="refined_answer",
            input_ground_truth_key="golden_doc_answer",
        )
        t2 = time.time()
        print("[STEP2] done in %.1fs" % (t2 - t1), flush=True)
        print("[TOTAL] %.1fs, total llm requests=%d" % (t2 - t0, self.llm_serving.n_requests), flush=True)

        final = os.path.join(self.cache_path, self.prefix + "_step2.jsonl")
        if os.path.exists(final):
            n = sum(1 for line in open(final, encoding="utf-8") if line.strip())
            print("[FINAL] %s rows=%d" % (final, n), flush=True)
        else:
            print("[FINAL] MISSING:", final, flush=True)


if __name__ == "__main__":
    try:
        model = AgenticRAGEval_APIPipeline()
        model.forward()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
