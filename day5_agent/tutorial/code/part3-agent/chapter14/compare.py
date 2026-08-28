"""Repeat a paired candidate-versus-incumbent comparison in fresh processes."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if args.runs < 3:
        raise SystemExit("--runs must be at least 3")

    results = [
        evaluate(args.task, args.candidate, args.incumbent) for _ in range(args.runs)
    ]
    failed = [result for result in results if result.get("status") != "ok"]
    if failed:
        print(json.dumps({"status": "failed", "results": results}, indent=2))
        raise SystemExit(1)

    candidate_medians = [float(result["latencyMs"]) for result in results]
    incumbent_medians = [
        float(result["comparison"]["incumbentLatencyMs"]) for result in results
    ]
    improvements = [
        float(result["comparison"]["improvementFraction"]) for result in results
    ]
    report = {
        "status": "ok",
        "runs": args.runs,
        "candidateMediansMs": candidate_medians,
        "incumbentMediansMs": incumbent_medians,
        "pairedImprovementFractions": improvements,
        "medianCandidateMs": statistics.median(candidate_medians),
        "medianIncumbentMs": statistics.median(incumbent_medians),
        "medianPairedImprovementFraction": statistics.median(improvements),
        "results": results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

