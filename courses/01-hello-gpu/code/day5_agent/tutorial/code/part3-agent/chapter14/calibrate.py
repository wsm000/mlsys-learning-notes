"""Repeat the frozen baseline to measure benchmark variation."""

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
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if args.runs < 3:
        raise SystemExit("--runs must be at least 3")

    results = [
        evaluate(args.task, args.candidate, args.candidate) for _ in range(args.runs)
    ]
    failed = [result for result in results if result.get("status") != "ok"]
    if failed:
        print(json.dumps({"status": "failed", "results": results}, indent=2))
        raise SystemExit(1)

    medians = [float(result["latencyMs"]) for result in results]
    paired_noise = [
        abs(float(result["comparison"]["improvementFraction"])) for result in results
    ]
    center = statistics.median(medians)
    max_paired_noise = max(paired_noise)
    report = {
        "status": "ok",
        "runs": args.runs,
        "medianLatencyMs": center,
        "runMediansMs": medians,
        "absolutePairedNoiseFractions": paired_noise,
        "maxPairedNoiseFraction": max_paired_noise,
        "conservativeThresholdFraction": max_paired_noise * 2,
        "note": "The threshold is an experiment policy derived from this run, not a hardware constant.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

