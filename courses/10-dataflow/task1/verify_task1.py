"""Focused end-to-end checks; writes a portable completion record."""
import argparse
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent
pdf_root = ROOT / "pdf2model"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-adapter", action="store_true")
    args = parser.parse_args()
    input_rows = read(ROOT / "workspace/example_data/ReasoningPipeline/pipeline_math_short.json")
    steps = sorted((ROOT / "workspace/cpu_pipelines/cache_local").glob("*.jsonl"))
    assert len(steps) == 3
    counts = [len([s for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]) for p in steps]
    assert len(input_rows) >= counts[0] >= counts[1] >= counts[2] > 0
    assert (ROOT / "init.log").is_file()
    assert (pdf_root / ".cache/train_config.yaml").is_file()
    preparation = read(pdf_root / "preparation_summary.json")
    summary = read(pdf_root / "training_summary.json")
    assert summary["status"] == "PASS" and summary["steps"] == 48
    train, evaluation = read(pdf_root / "train.json"), read(pdf_root / "eval.json")
    assert not ({r["instruction"] for r in train} & {r["instruction"] for r in evaluation})
    if args.require_adapter:
        assert (pdf_root / "adapter/adapter_model.safetensors").stat().st_size > 1000
    losses = read(pdf_root / "training_losses.json")
    assert len(losses) >= 12 and losses[-1]["step"] == 45
    assert summary["max_loaded_tensor_difference"] == 0.0
    result = {
        "status": "PASS", "python": platform.python_version(),
        "dataflow": "1.0.11 (recorded experiment version)",
        "verification_scope": "published evidence plus local adapter" if args.require_adapter else "published evidence only; original GPU verification is in verification.json",
        "cpu_pipeline_input_rows": len(input_rows), "cpu_pipeline_step_rows": counts,
        "pdf_preparation": preparation, "training": summary,
        "scope": "Sections 1.1 through 1.2.4, with documented local-backend adaptations",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
