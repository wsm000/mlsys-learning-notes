#!/usr/bin/env python3
"""DataFlow Task3: PDF VQA Extraction Pipeline (local adaptation).

Official pipeline: dataflow/statics/pipelines/api_pipelines/pdf_vqa_extract_pipeline.py
Local adaptations (vm-60, no route to huggingface/modelscope/mineru.net/github):
  1. MinerU: FileOrURLToMarkdownConverterLocal, backend "vlm-transformers"
     (pure transformers engine, no vllm/sglang needed); local model dir via ~/mineru.json
     -> replaces FileOrURLToMarkdownConverterFlash (vllm) / API (needs MINERU_API_KEY)
  2. Output-dir shim: mineru 2.6.8 hardcodes parse_method="vlm" for every vlm-* backend,
     so results land in intermediate/<pdf>/vlm/ while the operator chain expects
     intermediate/<pdf>/<backend>/. LocalMinerUExecutor symlinks vlm -> vlm-transformers
     after each parse so all upstream operators run unmodified.
  3. LLM: APILLMServing_request -> local Ollama OpenAI-compatible endpoint
     (http://localhost:11434/v1/chat/completions, qwen3:1.7b) instead of gemini-2.5-pro API
  4. max_chunk_len reduced to fit the 40k context of the local model
"""
import os
from dataflow.operators.knowledge_cleaning import FileOrURLToMarkdownConverterLocal
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage
from dataflow.operators.pdf2vqa import MinerU2LLMInputOperator, LLMOutputParser, QA_Merger, PDF_Merger, VQAFormatter
from dataflow.operators.core_text import ChunkedPromptedGenerator
from dataflow.pipeline import PipelineABC
from dataflow.prompts.pdf2vqa import QAExtractPrompt
from pathlib import Path
import re

MODEL_DIR = os.environ.get("MINERU_VLM_PATH", "/home/simin/df3_assets/MinerU2.5-2509-1.2B")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:1.7b")
LLM_URL = os.environ.get("LLM_URL", "http://localhost:11434/v1/chat/completions")


class LocalLLMOutputParser(LLMOutputParser):
    """Local adaptation (env TASK3_PARSER_FALLBACK=1): the upstream parser maps
    comma-separated numeric ids back to layout text. A small local LLM (qwen3:1.7b)
    tends to copy the original text instead of emitting ids, which makes every
    question/solution resolve to an empty string. When a field contains no valid
    numeric id, fall back to the raw string so the QA content survives.
    Trade-off: figure placement (<pic> tags) is lost, because images are referenced
    by id only."""

    def _id_to_text(self, input_ids, input_json, image_prefix="images"):
        text = super()._id_to_text(input_ids, input_json, image_prefix)
        if text.strip():
            return text
        raw = (input_ids or "").strip()
        if raw and not re.fullmatch(r"[\d,\s]*", raw):
            return raw
        return text


class LocalMinerUExecutor(FileOrURLToMarkdownConverterLocal):
    """FileOrURLToMarkdownConverterLocal + output-dir shim for mineru>=2.6.

    mineru 2.6.x hardcodes the output subdirectory to "vlm" for every vlm-* backend
    (mineru/cli/common.py: parse_method = "vlm"), but this operator composes the
    expected markdown path as intermediate/<pdf>/<backend>/. After the CLI run we
    symlink <backend> -> vlm so the recorded path resolves.
    """

    def _batch_parse_pdf_with_mineru(self, pdf_files: list):
        results = super()._batch_parse_pdf_with_mineru(pdf_files)
        for item in pdf_files:
            pdf_name = Path(item["raw_path"]).stem
            real_dir = os.path.abspath(os.path.join(self.intermediate_dir, pdf_name, "vlm"))
            alias_dir = os.path.join(self.intermediate_dir, pdf_name, self.mineru_backend)
            if os.path.isdir(real_dir) and not os.path.exists(alias_dir):
                os.symlink(real_dir, alias_dir)  # absolute target: symlink resolves relative to its own dir
        return results


class PDF_VQA_extract_local_pipeline(PipelineABC):
    def __init__(self):
        super().__init__()
        self.storage = FileStorage(
            first_entry_file_name="./example_data/PDF2VQAPipeline/vqa_extract_test.jsonl",
            cache_path="./cache",
            file_name_prefix="vqa",
            cache_type="jsonl",
        )
        self.llm_serving = APILLMServing_request(
            api_url=LLM_URL,
            key_name_of_api_key="DF_API_KEY",
            model_name=LLM_MODEL,
            max_workers=2,
            max_tokens=8192,
            read_timeout=900.0,
        )
        self.vqa_extract_prompt = QAExtractPrompt()
        self.pdf_merger = PDF_Merger(output_dir="./cache")

        # Local MinerU (vlm-transformers engine, no vllm needed)
        self.mineru_executor = LocalMinerUExecutor(
            intermediate_dir="intermediate",
            mineru_backend="vlm-transformers",
            mineru_source="local",
            mineru_model_path=MODEL_DIR,
            mineru_download_model_type="vlm",
        )
        self.input_formatter = MinerU2LLMInputOperator()
        self.vqa_extractor = ChunkedPromptedGenerator(
            llm_serving=self.llm_serving,
            system_prompt=self.vqa_extract_prompt.build_prompt(),
            max_chunk_len=12000,
        )
        parser_cls = LocalLLMOutputParser if os.environ.get("TASK3_PARSER_FALLBACK") == "1" else LLMOutputParser
        self.llm_output_parser = parser_cls(output_dir="./cache", intermediate_dir="intermediate")
        self.qa_merger = QA_Merger(output_dir="./cache", strict_title_match=False)
        self.vqa_format_converter = VQAFormatter(output_json_file="./cache/data/qa.json")

    def forward(self):
        self.pdf_merger.run(
            storage=self.storage.step(),
            input_pdf_list_key="input_pdf_paths",
            input_name_key="name",
            output_pdf_path_key="merged_pdf_path",
        )
        self.mineru_executor.run(
            storage=self.storage.step(),
            input_key="merged_pdf_path",
            output_key="vqa_markdown_path",
        )
        self.input_formatter.run(
            storage=self.storage.step(),
            input_markdown_path_key="vqa_markdown_path",
            output_converted_layout_key="converted_vqa_layout_path",
        )
        self.vqa_extractor.run(
            storage=self.storage.step(),
            input_path_key="converted_vqa_layout_path",
            output_path_key="extracted_llm_vqa_path",
        )
        self.llm_output_parser.run(
            storage=self.storage.step(),
            input_response_path_key="extracted_llm_vqa_path",
            input_converted_layout_path_key="converted_vqa_layout_path",
            input_name_key="name",
            output_qalist_path_key="extracted_vqa_path",
        )
        self.qa_merger.run(
            storage=self.storage.step(),
            input_qalist_path_key="extracted_vqa_path",
            input_name_key="name",
            output_merged_qalist_path_key="output_merged_vqalist_path",
            output_merged_md_path_key="output_merged_md_path",
            output_qa_item_key="vqa_pair",
        )
        self.vqa_format_converter.run(
            storage=self.storage.step(),
            input_qa_item_key="vqa_pair",
            output_messages_key="messages",
            output_images_key="images",
        )


if __name__ == "__main__":
    pipeline = PDF_VQA_extract_local_pipeline()
    pipeline.compile()
    pipeline.forward()
