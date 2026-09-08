"""Deliberately unfused Triton baseline for the masked-softmax task."""

import triton
import triton.language as tl


@triton.jit
def _row_max_kernel(
    x,
    row_max,
    scale,
    cols: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    valid = offsets < cols
    causal = offsets <= row % cols
    values = tl.load(x + row * cols + offsets, mask=valid, other=0.0)
    values = values.to(tl.float32) * scale
    values = tl.where(valid & causal, values, -float("inf"))
    tl.store(row_max + row, tl.max(values, axis=0))


@triton.jit
def _exp_sum_kernel(
    x,
    exp_values,
    row_max,
    row_sum,
    scale,
    cols: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    valid = offsets < cols
    causal = offsets <= row % cols
    values = tl.load(x + row * cols + offsets, mask=valid, other=0.0)
    values = values.to(tl.float32) * scale
    maximum = tl.load(row_max + row)
    numerator = tl.exp(tl.where(valid & causal, values, -float("inf")) - maximum)
    tl.store(exp_values + row * cols + offsets, numerator, mask=valid)
    tl.store(row_sum + row, tl.sum(numerator, axis=0))


@triton.jit
def _normalize_kernel(
    exp_values,
    row_sum,
    output,
    cols: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    valid = offsets < cols
    numerator = tl.load(exp_values + row * cols + offsets, mask=valid, other=0.0)
    denominator = tl.load(row_sum + row)
    tl.store(output + row * cols + offsets, numerator / denominator, mask=valid)


def launch(x, output, exp_values, row_max, row_sum, scale, rows, cols):
    """Write the complete result to output using preallocated workspace."""

    block_size = triton.next_power_of_2(cols)
    num_warps = 8 if block_size >= 2048 else 4
    grid = (rows,)
    _row_max_kernel[grid](
        x,
        row_max,
        scale,
        cols=cols,
        block_size=block_size,
        num_warps=num_warps,
    )
    _exp_sum_kernel[grid](
        x,
        exp_values,
        row_max,
        row_sum,
        scale,
        cols=cols,
        block_size=block_size,
        num_warps=num_warps,
    )
    _normalize_kernel[grid](
        exp_values,
        row_sum,
        output,
        cols=cols,
        block_size=block_size,
        num_warps=num_warps,
    )

