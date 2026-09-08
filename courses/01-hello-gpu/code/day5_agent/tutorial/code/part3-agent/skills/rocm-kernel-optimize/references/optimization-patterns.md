# 优化套路库（按 bound 类型）

## memory-bound（多数 elementwise / 带宽型算子）

- 合并访存（coalescing）：相邻 lane 访问相邻地址
- 向量化加载/写回（一次搬更宽，如一次处理 4 个元素）
- 增大 block / 提高占用，摊薄 launch 开销
- 减少中间结果写回；能寄存器完成就别落全局内存
- kernel 融合（多步合一）
- 避免读放大（共享内存 / LDS 缓存复用）

## compute-bound

- 消除冗余计算（循环不变量外提）
- 更高吞吐数据类型/指令（fp16/bf16、dot）
- 指令级并行（展开独立计算）

## vector_add 常见改法（教学）

1. 增大 `block_size`（256 → 1024）
2. 每个 program 处理多段（grid-stride 或向量化）
3. 保持合并访存，避免额外临时缓冲

## 踩坑

- occupancy：寄存器/LDS 过多会降低活跃 wave
- N 非 2 的幂时注意 LDS bank conflict
- cache 带宽 ≠ 主存带宽：小数据反复命中会高估带宽
- **性能数字只认 `bench_kernel` / `accept_candidate`，`run_code` 自测不算数**
