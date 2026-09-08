#!/usr/bin/env python3
"""Print a screenshot-friendly one-dimensional HIP thread mapping."""

from __future__ import annotations

import argparse
import sys


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show the Grid -> Block -> Thread mapping for Vector Add."
    )
    parser.add_argument("--n", type=positive_int, default=10, help="array length")
    parser.add_argument(
        "--block", type=positive_int, default=4, help="threads per block"
    )
    args = parser.parse_args()

    grid = (args.n + args.block - 1) // args.block
    print("=== Vector Add Thread Mapping ===")
    print(f"N: {args.n}")
    print(f"block_size: {args.block}")
    print(f"grid_size: {grid}")
    print("formula: i = blockIdx.x * blockDim.x + threadIdx.x")
    print()
    print("block | thread | global_i | action")
    print("------+--------+----------+--------------------------------------")

    for block_index in range(grid):
        for thread_index in range(args.block):
            index = block_index * args.block + thread_index
            if index < args.n:
                action = f"valid: c[{index}] = a[{index}] + b[{index}]"
            else:
                action = "OUT OF BOUNDS: return; no memory access"
            print(f"{block_index:5d} | {thread_index:6d} | {index:8d} | {action}")

    tail_threads = grid * args.block - args.n
    print()
    print(f"summary: {args.n} valid threads, {tail_threads} out-of-bounds threads")
    if args.n == 10 and args.block == 4:
        print("required check: Block 0 -> 0--3; Block 1 -> 4--7; Block 2 -> 8--11")
        print("required check: i=10 and i=11 are OUT OF BOUNDS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
