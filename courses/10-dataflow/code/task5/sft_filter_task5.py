
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DataFlow Task5 — SFT 数据过滤（官方 GPU 版 SFTTextFilter pipeline 改造，vm-60 实测）

官方原型：dataflow/statics/pipelines/gpu_pipelines/text_sft_filter.py（SFTTextFilter_GPUPipeline）
改造点（相对官方）：
  1) first_entry_file_name 改为 T5_INPUT（本任务设计语料 sft_input.jsonl，含 id 列）；
  2) cache_path/prefix 改为 T5_CACHE/T5_PREFIX；
  3) 模型缓存指向 T5_HF_CACHE（/mnt/nfs/task5_hf：gpt2 用于 Superfiltering IFD，
     hkust-nlp/deita-quality-scorer 用于 DEITA 质量分）；
  4) 算子、参数与顺序与官方 GPU 版完全一致：
     WordNumberFilter(min_words=20, max_words=1000, key=output)
       -> SuperfilteringFilter(min_score=0.5, max_score=1.0, gpt2 IFD 比值)
       -> DeitaQualityFilter(min_score=2.5, max_score=10000, 1-6 分质量评分)
  5) 每步打印 in/out 行数与耗时；分数列（SuperfilteringScore/DeitaQualityScore）
     保留在 cache step 文件中，供 verify_task5.py 统计分布。
"""
import os
import time

from dataflow.operators.general_text import WordNumberFilter
from dataflow.operators.text_sft import SuperfilteringFilter, DeitaQualityFilter, AlpagasusFilter
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage
import os as _os


def count_lines(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


class SFTTextFilter_Task5:
    def __init__(self):
        self.input_path = os.environ.get("T5_INPUT") or "./sft_input.jsonl"
        self.cache_path = os.environ.get("T5_CACHE") or "./cache_sft"
        self.prefix = os.environ.get("T5_PREFIX") or "sft_filter_task5"
        os.makedirs(self.cache_path, exist_ok=True)
        self.storage = FileStorage(
            first_entry_file_name=self.input_path,
            cache_path=self.cache_path,
            file_name_prefix=self.prefix,
            cache_type="jsonl",
        )
        self.model_cache_dir = os.environ.get("T5_HF_CACHE") or "/mnt/nfs/task5_hf"

        self.word_number_filter_step1 = WordNumberFilter(min_words=20, max_words=1000)
        self.super_filtering_filter_step2 = SuperfilteringFilter(
            min_score=0.5,
            max_score=1.0,
            model_cache_dir=self.model_cache_dir,
        )
        # 官方第三步是 DeitaQualityFilter（hkust-nlp/deita-quality-scorer，Llama-7B fp32 13.5GB）。
        # 若该模型未在离线缓存中就绪（本次因 hf-mirror 限流未能完整拉取），
        # 退化用官方 API 合成管线同款的 AlpagasusFilter（LLM 对 Q/A 打 0-5 分，官方阈值 [3,5]）顶替同一"质量分过滤"角色，
        # 并在日志中明确标注用了哪条路径。
        _deita_bin = _os.path.join(
            self.model_cache_dir, "models--hkust-nlp--deita-quality-scorer", "snapshots", "main", "pytorch_model-00002-of-00002.bin"
        )
        # 两个 shard 的期望字节数（来自 hf 仓库元数据）：9976639678 + 3544121344 = 13520761022
        _deita_bin1 = _os.path.join(
            self.model_cache_dir, "models--hkust-nlp--deita-quality-scorer", "snapshots", "main", "pytorch_model-00001-of-00002.bin"
        )
        self.deita_ready = (
            _os.path.exists(_deita_bin1) and _os.path.getsize(_deita_bin1) == 9976639678
            and _os.path.exists(_deita_bin) and _os.path.getsize(_deita_bin) == 3544121344
        )
        if self.deita_ready:
            self.deita_quality_filter_step3 = DeitaQualityFilter(
                min_score=2.5,
                max_score=10000,
                max_length=512,
                model_cache_dir=self.model_cache_dir,
            )
        else:
            print("[FALLBACK] deita-quality-scorer 不在离线缓存，第三步改用 AlpagasusFilter(LLM 0-5 分, [3,5])", flush=True)
            api_base = (_os.environ.get("AMD_API_BASE") or "https://developer.amd.com.cn/radeon/api/v1").rstrip("/")
            llm = APILLMServing_request(
                api_url=api_base + "/chat/completions",
                model_name=_os.environ.get("DF_MODEL") or "DeepSeek-V4-Flash",
                max_workers=2,
                max_retries=6,
                connect_timeout=10.0,
                read_timeout=180.0,
            )
            self.deita_quality_filter_step3 = AlpagasusFilter(min_score=3, max_score=5, llm_serving=llm)

    def run_op(self, name, fn, **kwargs):
        step_storage = self.storage.step()
        n = step_storage.operator_step
        in_path = self.input_path if n == 0 else os.path.join(self.cache_path, f"{self.prefix}_step{n}.jsonl")
        out_path = os.path.join(self.cache_path, f"{self.prefix}_step{n + 1}.jsonl")
        before = count_lines(in_path)
        t0 = time.time()
        fn(storage=step_storage, **kwargs)
        after = count_lines(out_path)
        print(f"[STEP] {name:32s} {str(before):>4s} -> {str(after):>4s}  ({time.time()-t0:6.1f}s)", flush=True)

    def forward(self):
        self.run_op("WordNumberFilter", self.word_number_filter_step1.run, input_key="output")
        self.run_op(
            "SuperfilteringFilter",
            self.super_filtering_filter_step2.run,
            input_instruction_key="instruction",
            input_input_key=None,
            input_output_key="output",
        )
        if self.deita_ready:
            self.run_op(
                "DeitaQualityFilter",
                self.deita_quality_filter_step3.run,
                input_instruction_key="instruction",
                input_output_key="output",
            )
        else:
            # AlpagasusFilter.run 的签名比 DeitaQualityFilter 多一个 input_input_key
            self.run_op(
                "AlpagasusFilter(fallback)",
                self.deita_quality_filter_step3.run,
                input_instruction_key="instruction",
                input_input_key="input",
                input_output_key="output",
            )


if __name__ == "__main__":
    print("[CONFIG] input    =", os.environ.get("T5_INPUT"), flush=True)
    print("[CONFIG] cache    =", os.environ.get("T5_CACHE"), flush=True)
    print("[CONFIG] hf_cache =", os.environ.get("T5_HF_CACHE"), flush=True)
    t0 = time.time()
    model = SFTTextFilter_Task5()
    print(f"[INIT] operators ready ({time.time()-t0:.1f}s)", flush=True)
    model.forward()
    print(f"[DONE] total {time.time()-t0:.1f}s", flush=True)
