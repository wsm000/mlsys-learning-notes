# 状态：这是一次**未完成**的扩样尝试，结论以 exp10-1b-adv 为准

**这个目录是什么**

`exp_b_extension.py` 用一套 35 题的新陷阱集（B01-B35，ground truth 全部来自本机真实执行）
重测 §10.2 表 10-1。它只跑完了 **G（生成）阶段**：

- `generation.json`：35 条生成记录
- `review_B1.json` / `review_B2.json`：**不存在**，两个审查臂从未运行

所以这里**没有任何配对比较**，不能用来支持或反对任何结论。

**它唯一还能贡献的一件事**

32 道可判分题目上，生成者 **32/32 全对（100%）**。这 32 个样本并入功率分析后，
"生成者在筛选任务上不出错"的证据从 110 道增加到 **142 道，错误数仍为 0**
（见 `../exp10-1b-power-analysis.json` 的 `screening.exp10_1b_ext_generation_only`）。

**为什么被取代**

最终采用的扩样是 `exp10-1b-adv/`：38 题 × 3 轮 = 113 对，两个审查臂都完整，
并且修掉了两个评分缺陷（"没有产出答案"被当成答错、输出预算对 B1 的系统性歧视）。
结论与机制解释见 `../../notes/task05-ch10-shared-context.md` §2.3 与
`../../notes/task05-ch10-practice.md` §2。

**复现方式**

```bash
cd courses/08-ai-agent-book/code/task5
python exp_b_extension.py            # 会重跑 G 阶段（幂等，按 task_id 跳过）
python exp_b_extension.py --rescore  # 只读已有 JSON 确定性重判
```

保留这个目录与脚本，是为了让"扩样这件事试过两条路、其中一条没走完"有据可查，
而不是只留下成功那条。
