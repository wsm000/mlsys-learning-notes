#!/usr/bin/env python3
"""Case 6: PromptedVQAGenerator on rendered PDF page images via local Ollama VLM.

Env: VLM_MODEL (default moondream:1.8b), CASE6_PROMPT in {ocr,describe,vqa}
"""
import os
from dataflow.operators.core_vision import PromptedVQAGenerator  # 1.0.11 uses LazyLoader; import via package attr
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.utils.storage import FileStorage

PROMPTS = {
    "ocr": "Please recognize and output all text present in the image, including any formulas in LaTeX.",
    "describe": "Describe the image in detail, focusing on any formulas, charts or diagrams.",
    "vqa": "What mathematical content is shown in this image? Answer concisely.",
}

def main():
    model = os.environ.get("VLM_MODEL", "moondream:1.8b")
    kind = os.environ.get("CASE6_PROMPT", "ocr")
    self_prompt = PROMPTS[kind]
    storage = FileStorage(
        first_entry_file_name="./example_data/VQA/pic_path.json",
        cache_path="./cache_case6",
        file_name_prefix="vqa",
        cache_type="json",
    )
    llm_serving = APIVLMServing_openai(
        model_name=model,
        api_url="http://localhost:11434/v1",
        key_name_of_api_key="DF_API_KEY",
        max_workers=2,
        timeout=600,
    )
    vqa = PromptedVQAGenerator(llm_serving, self_prompt)
    vqa.run(storage=storage.step(), input_key="raw_content")
    print("CASE6_DONE prompt=%s model=%s" % (kind, model))

if __name__ == "__main__":
    main()
