"""Triton baseline for vector add（chapter15 教学用，故意朴素）。"""

import triton
import triton.language as tl


@triton.jit
def _vadd_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def launch(x, y, output, n_elements):
    block_size = 256  # 故意偏小，给优化留空间（可试更大 block / 向量化）
    grid = (triton.cdiv(n_elements, block_size),)
    _vadd_kernel[grid](x, y, output, n_elements, block_size)

