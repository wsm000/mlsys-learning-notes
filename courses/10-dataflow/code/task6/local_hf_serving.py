#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DataFlow Task6 — 本地 transformers 版 LLM serving（官方 LocalModelLLMServing_vllm 的替代实现）

为什么需要它：
  官方 benchmark_eval 管线（gpu_pipelines/benchmark_eval/unified_bench_eval_type*.py）默认用
  LocalModelLLMServing_vllm 生成答案。但 open-dataflow 1.0.11 的 [vllm] extra 锁
  vllm<=0.9.2 -> torch==2.7.0，与 vm-60 上 .venv-df3 的 torch 2.9.1 冲突（强制降级会破坏
  Task3/4/5 共用的环境）。因此本任务用一个 transformers 直写的 serving 顶替同一角色：

    - 接口与官方 serving 对齐：generate_from_input(user_inputs, system_prompt) / start_serving / cleanup
    - 暴露 real_model_path / hf_cache_dir / trust_remote_code，
      官方 UnifiedBenchDatasetEvaluator 的 _ll_batch / _ppl_batch 在 serving 没有
      loglikelihood 接口时会回退到「按 real_model_path 自己用 transformers 加载模型」，
      所以 key3_q_choices_a 的 ll_choice_acc、key3_q_a_rejected 的 pairwise_ll_winrate、
      key1_text_score 的 ppl 都能原样跑通，无需改算子一行代码。
    - 评测场景默认贪心解码（temperature=0），保证可复现；官方 vllm 版默认 0.7。

仅此一处替换，算子 / prompt / metric / storage 与官方管线完全一致。
"""
import os
import threading
import time
from typing import List, Optional

from dataflow import get_logger
from dataflow.core import LLMServingABC


class LocalModelLLMServing_hf(LLMServingABC):
    """transformers 版本地模型 serving（LLMServingABC 接口对齐官方 vllm 版）"""

    def __init__(
        self,
        hf_model_name_or_path: str,
        hf_cache_dir: Optional[str] = None,
        hf_local_dir: Optional[str] = None,
        hf_temperature: float = 0.0,
        hf_top_p: float = 1.0,
        hf_max_tokens: int = 512,
        hf_max_model_len: int = 2048,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        self.logger = get_logger()
        self.hf_model_name_or_path = hf_model_name_or_path
        self.hf_cache_dir = hf_cache_dir
        self.hf_local_dir = hf_local_dir
        self.hf_temperature = hf_temperature
        self.hf_top_p = hf_top_p
        self.hf_max_tokens = hf_max_tokens
        self.hf_max_model_len = hf_max_model_len
        self.trust_remote_code = trust_remote_code

        # 官方 serving 在 start_serving 里决定 real_model_path（本地目录或 snapshot_download）
        self.real_model_path = hf_model_name_or_path

        self._device = device
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self.backend_initialized = False
        self.n_requests = 0

    # ---------------- 生命周期 ----------------
    def start_serving(self) -> None:
        if self.backend_initialized:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.real_model_path,
            cache_dir=self.hf_cache_dir,
            trust_remote_code=self.trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.real_model_path,
            cache_dir=self.hf_cache_dir,
            trust_remote_code=self.trust_remote_code,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._model.eval()
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self.backend_initialized = True
        self.logger.info(
            f"[LocalModelLLMServing_hf] model={self.real_model_path} device={self._device} "
            f"load={time.time()-t0:.1f}s temperature={self.hf_temperature} max_tokens={self.hf_max_tokens}"
        )

    def cleanup(self) -> None:
        model, tok = self._model, self._tokenizer
        self._model = None
        self._tokenizer = None
        self.backend_initialized = False
        del model, tok
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ---------------- 生成 ----------------
    def generate_from_input(self, user_inputs: List[str], system_prompt: str = None) -> List[str]:
        if not self.backend_initialized:
            self.start_serving()

        import torch

        outputs: List[str] = []
        with self._lock:
            for user_input in user_inputs:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_input})
                prompt_text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                enc = self._tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.hf_max_model_len,
                ).to(self._device)
                gen_kwargs = dict(
                    max_new_tokens=self.hf_max_tokens,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
                if self.hf_temperature and self.hf_temperature > 0:
                    gen_kwargs.update(
                        do_sample=True,
                        temperature=self.hf_temperature,
                        top_p=self.hf_top_p,
                    )
                else:
                    gen_kwargs.update(do_sample=False)
                with torch.no_grad():
                    out = self._model.generate(**enc, **gen_kwargs)
                new_tokens = out[0][enc["input_ids"].shape[-1]:]
                text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
                outputs.append(text.strip())
                self.n_requests += 1
        return outputs
