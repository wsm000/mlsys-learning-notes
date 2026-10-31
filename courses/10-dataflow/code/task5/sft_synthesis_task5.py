
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DataFlow Task5 — SFT 数据合成（官方 API 版 TextSFTSynthesis pipeline 改造，vm-60 实测）

官方原型：dataflow/statics/pipelines/api_pipelines/text_sft_synthesis_pipeline.py
链路（1.0.11）：
  CondorGenerator  —— 基于预置知识树标签(topic/domain/theme)两阶段从零合成：
                      阶段一生成 [Easy]/[Medium]/[Hard] 三档问题，阶段二逐题生成答案
  CondorRefiner    —— 两阶段优化：先对 (question, answer) 生成 critique，再按 critique 改写答案
  AlpagasusFilter  —— LLM 打分（0-5）过滤，官方阈值 min_score=3, max_score=5

改造点（相对官方）：
  1) LLM 端点改为 AMD Radeon OpenAI 兼容端点 + DeepSeek-V4-Flash（经 Windows 主机隧道转发，
     因 vm-60 直连外网被拦截；密钥只从远端 key 文件进进程，不落盘不外传）；
  2) 官方 max_workers=100 在该端点（20 RPM 用户级限流）会触发 429 风暴，改为进程内匀速限流
     ThrottledLoggingAPILLMServing_request（默认 rpm=10、workers=2，与 Task4 同策略）；
  3) 每个请求/响应对落盘 api_calls.jsonl（不含密钥），用于审计与复现；
  4) random.seed 固定，保证知识树采样可复现；
  5) cache/prefix 改为 T5_CACHE/T5_PREFIX，输入为空文件（合成模式从零开始）。
"""
import json
import os
import random
import threading
import time

from dataflow.operators.text_sft import (
    AlpagasusFilter,
    CondorGenerator,
    CondorRefiner,
)
from dataflow.prompts.general_text import CondorQuestionPrompt, CondorRefinePrompt
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage


def env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


class ThrottledLoggingAPILLMServing_request(APILLMServing_request):
    """在官方 APILLMServing_request 之上加：进程内匀速限流 + 请求/响应审计日志。

    AMD 端点对 OneClick 用户限流 20 requests/minute（Task4 实测），
    官方 pipeline 默认 max_workers=100 会瞬间打满；这里每 60/rpm 秒放行一个请求，
    让 DataFlow 自带退避重试只需处理偶发冲突。
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
        # 只审计 prompt 与响应文本，不记录密钥/头信息
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


class TextSFTSynthesis_Task5:
    def __init__(self):
        self.cache_path = os.environ.get("T5_CACHE") or "./cache_syn"
        self.prefix = os.environ.get("T5_PREFIX") or "sft_synthesis_task5"
        os.makedirs(self.cache_path, exist_ok=True)
        empty = os.path.join(self.cache_path, "empty.jsonl")
        if not os.path.exists(empty):
            open(empty, "w", encoding="utf-8").close()

        self.storage = FileStorage(
            first_entry_file_name=empty,
            cache_path=self.cache_path,
            file_name_prefix=self.prefix,
            cache_type="jsonl",
        )

        api_base = (os.environ.get("AMD_API_BASE") or "https://developer.amd.com.cn/radeon/api/v1").rstrip("/")
        self.llm_serving = ThrottledLoggingAPILLMServing_request(
            api_url=api_base + "/chat/completions",
            model_name=os.environ.get("DF_MODEL") or "DeepSeek-V4-Flash",
            max_workers=env_int("T5_MAX_WORKERS", 2),
            max_retries=env_int("T5_MAX_RETRIES", 6),
            connect_timeout=10.0,
            read_timeout=180.0,
            rpm=env_int("T5_RPM", 10),
            log_path=os.path.join(self.cache_path, "api_calls.jsonl"),
        )

        self.num_samples = env_int("T5_NUM_SAMPLES", 9)
        random.seed(env_int("T5_SEED", 20261101))

        self.generator = CondorGenerator(
            llm_serving=self.llm_serving,
            num_samples=self.num_samples,
            prompt_template=CondorQuestionPrompt(),
        )
        self.refiner = CondorRefiner(
            llm_serving=self.llm_serving,
            prompt_template=CondorRefinePrompt(),
        )
        self.alpagasus_filter = AlpagasusFilter(min_score=3, max_score=5, llm_serving=self.llm_serving)

    def forward(self):
        t0 = time.time()
        print("[STEP] CondorGenerator (num_samples=%d)" % self.num_samples, flush=True)
        self.generator.run(storage=self.storage.step())
        print("[STEP] CondorGenerator done (%.1fs)" % (time.time() - t0), flush=True)

        t1 = time.time()
        print("[STEP] CondorRefiner", flush=True)
        self.refiner.run(
            storage=self.storage.step(),
            input_instruction_key="instruction",
            input_output_key="output",
        )
        print("[STEP] CondorRefiner done (%.1fs)" % (time.time() - t1), flush=True)

        t2 = time.time()
        print("[STEP] AlpagasusFilter (min_score=3, max_score=5)", flush=True)
        self.alpagasus_filter.run(
            storage=self.storage.step(),
            input_instruction_key="instruction",
            input_input_key="input",
            input_output_key="output",
        )
        print("[STEP] AlpagasusFilter done (%.1fs)" % (time.time() - t2), flush=True)
        print("[DONE] api_requests=%d total=%.1fs" % (self.llm_serving.n_requests, time.time() - t0), flush=True)


if __name__ == "__main__":
    print("[CONFIG] cache     =", os.environ.get("T5_CACHE"), flush=True)
    print("[CONFIG] api_url   =", os.environ.get("AMD_API_BASE"), flush=True)
    print("[CONFIG] model     =", os.environ.get("DF_MODEL"), flush=True)
    model = TextSFTSynthesis_Task5()
    model.forward()
