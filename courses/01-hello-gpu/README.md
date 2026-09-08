# hello-gpu

本目录收纳 `datawhalechina/hello-gpu` 学习记录，按 Day1–Day5 排列。

## 内容索引

| 笔记 | 说明 |
|---|---|
| [Day1_认识GPU并跑通程序.md](notes/day01-认识GPU并跑通程序.md) | GPU 并行模型、Vector Add、线程映射与 Day 1 打卡材料。 |
| [Day2_学会可信计时.md](notes/day02-学会可信计时.md) | warmup、同步、GPU Event、统计口径与 Day 2 打卡材料。 |
| [Day2_OpenMP_学习笔记.md](notes/day02-OpenMP学习笔记.md) | Day 2 延伸：OpenMP CPU 侧并行对照。 |
| [Day3_解释性能瓶颈.md](notes/day03-解释性能瓶颈.md) | rocprof Kernel Trace、Grid 划分、合并访存与 Roofline 读图。 |
| [Day4_实现Vector_Add.md](notes/day04-实现Vector_Add.md) | Triton Vector Add、边界 mask、BLOCK_SIZE 单变量实验。 |
| [Day5_Agent接手优化.md](notes/day05-Agent接手优化.md) | Agent 优化闭环、TaskSpec、baseline、Oracle 与门禁。 |
| [心得体会_五日GPU学习总结.md](notes/day01-05-五日学习总结.md) | Day 1–5 综合心得。 |

## 代码与证据

- `code/day1_vector_add/`
- `code/day2_timing/`
- `code/day3_rerun/`
- `code/day5_agent/`
- `evidence/打卡材料/`

> 当前阶段先建立统一课程入口；原文件保留以避免旧 GitHub 链接失效。
