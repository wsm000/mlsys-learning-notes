"""KBC PDF -> text -> real model-generated QA -> Alpaca training data.

Adaptation for a born-digital PDF: PyMuPDF replaces OCR/MinerU;
the existing local Ollama server replaces the template's multi-GPU vLLM.
DataFlow FileStorage and PromptedGenerator perform the pipeline processing.
"""
import hashlib
import json
import re
from pathlib import Path

import fitz
import requests
from dataflow.core import LLMServingABC
from dataflow.operators.core_text import PromptedGenerator
from dataflow.utils.storage import FileStorage

ROOT = Path(__file__).resolve().parent


class OllamaServing(LLMServingABC):
    def start_serving(self):
        pass

    def cleanup(self):
        pass

    def generate_from_input(self, user_inputs, system_prompt, **kwargs):
        results = []
        for i, text in enumerate(user_inputs):
            response = requests.post(
                "http://127.0.0.1:11434/api/chat",
                json={"model": "qwen3:0.6b", "stream": False, "think": False,
                      "format": {"type": "object", "properties": {"pairs": {"type": "array", "items": {"type": "object", "properties": {"question": {"type": "string"}, "answer": {"type": "string"}}, "required": ["question", "answer"]}}}, "required": ["pairs"]},
                      "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
                      "options": {"temperature": 0, "num_predict": 500, "num_ctx": 2048}},
                timeout=180,
            )
            response.raise_for_status()
            result = response.json()["message"]["content"]
            json.loads(result)
            results.append(result)
            print(f"QA_CHUNK_DONE {i + 1}/{len(user_inputs)}", flush=True)
        return results


if __name__ == "__main__":
    pdf = ROOT / "bitter_lesson.pdf"
    document = fitz.open(pdf)
    pages = []
    for number, page in enumerate(document, 1):
        raw = page.get_text()
        assert len(raw.strip()) > 200
        # Strip the repeated browser print header/footer, retain the article text.
        lines = [s for s in raw.splitlines() if not re.match(r"^(2/3/22|www\.incompleteideas|[12]/2$)", s)]
        text = re.sub(r"\s+", " ", " ".join(lines)).strip()
        pages.append({"page": number, "text": text})
        page.get_pixmap(matrix=fitz.Matrix(1, 1)).save(ROOT / f"pdf_page_{number}.png")
    (ROOT / "extracted_pages.json").write_text(json.dumps(pages, indent=2, ensure_ascii=False))
    chunks = []
    for record in pages:
        sentences = re.split(r"(?<=[.!?])\s+", record["text"])
        for start in range(0, len(sentences), 5):
            content = " ".join(sentences[start:start + 5])
            if len(content) > 150:
                chunks.append({"page": record["page"], "chunk_id": len(chunks), "text": content})
    (ROOT / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    storage = FileStorage(first_entry_file_name=str(ROOT / "chunks.json"), cache_path=str(ROOT / "cache_local"), cache_type="jsonl")
    generator = PromptedGenerator(
        llm_serving=OllamaServing(),
        system_prompt="Create exactly two factual question-answer pairs supported ONLY by the passage. Answers must be short direct quotations from the passage, at most 35 words. Return JSON with a pairs array containing question and answer fields. Do not invent facts.",
    )
    generator.run(storage=storage.step(), input_key="text", output_key="qa_json")
    output = sorted((ROOT / "cache_local").glob("*.jsonl"), key=lambda p: p.stat().st_mtime)[-1]
    records = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    pairs, rejected = [], []
    seen = set()
    for row in records:
        for pair in json.loads(row["qa_json"]).get("pairs", []):
            question, answer = pair.get("question", "").strip(), pair.get("answer", "").strip()
            normalized = lambda value: re.sub(r"\s+", " ", value).lower().strip(" .\"'")
            valid = bool(question and answer and len(answer.split()) <= 45 and normalized(answer) in normalized(row["text"]) and question not in seen)
            evidence = {"instruction": question, "input": "", "output": answer, "source_page": row["page"], "chunk_id": row["chunk_id"]}
            if valid:
                seen.add(question)
                pairs.append(evidence)
            else:
                rejected.append(evidence)
    assert len(pairs) >= 6, f"Only {len(pairs)} grounded pairs survived validation"
    (ROOT / "qa_evidence.json").write_text(json.dumps(pairs, indent=2, ensure_ascii=False))
    (ROOT / "rejected_qa.json").write_text(json.dumps(rejected, indent=2, ensure_ascii=False))
    alpaca = [{key: row[key] for key in ("instruction", "input", "output")} for row in pairs]
    (ROOT / "qa.json").write_text(json.dumps(alpaca, indent=2, ensure_ascii=False))
    summary = {"pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(), "pages": len(pages), "chunks": len(chunks), "accepted_qa": len(pairs), "rejected_qa": len(rejected), "backend": "Ollama qwen3:0.6b, think=false", "extraction": "PyMuPDF born-digital text"}
    (ROOT / "preparation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
