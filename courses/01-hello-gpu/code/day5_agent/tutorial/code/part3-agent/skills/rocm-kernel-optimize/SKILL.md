# ROCm Kernel Optimize Skill

## 何时使用

当需要把 GPU 算子（Triton / 转成 Triton 后）在本机硬件上做**正确且可信**的性能优化时启用。

## 优化 SOP（摘要）

1. 理解任务（shape / dtype / 入口）
2. `measure_peak` 测硬件峰值
3. `profile_kernel` 定 memory/compute-bound
4. 按 bound 选套路（见 `references/optimization-patterns.md`）
5. 生成候选 → `compile_kernel` → `bench_kernel` → `accept_candidate`
6. 反思迭代 → 报告（数字只认权威工具）

## 纪律

- 工具结果即事实；禁止用 `run_code` 自测数字当结论
- 正确性先于性能；晋升只认 `accept_candidate`
- 一次只改一个主要机制
