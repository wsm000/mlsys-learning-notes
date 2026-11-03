# 同伴复现协议与互评清单（Task8 · Basic/Advanced 档）

> 用途：任何拿到本仓库的人，按下面步骤**独立复现** Quanty Task8 Demo；复现者按互评清单给作者打分/反馈。
> 设计原则来自课程方法论：**固定输入 + 明确预期**——复现者不需要理解全部机制，只需要能判定"通过/不通过"。

## A. 同伴复现协议（目标：≤ 3 分钟，含安装约 5 分钟）

```bash
# 0) 拿代码（任选其一）
git clone <本仓库地址> && cd <仓库>/courses/09-deepagents-in-action
# 或：直接拿到 courses/09-deepagents-in-action/ 目录

# 1) 建环境（Python 3.12+；无需 GPU / API Key / 网络）
py -3.12 -m venv .venv                        # Windows
.venv\Scripts\pip install -r requirements.lock.txt
# python3.12 -m venv .venv && .venv/bin/pip install -r requirements.lock.txt   # Linux/macOS

# 2) 一条命令跑 Demo
.venv\Scripts\python task8\run_demo.py         # .venv/bin/python task8/run_demo.py

# 3) 判定
#    通过标准（三条同时满足）：
#      a) 输出含 22 行 [PASS]、0 行 [FAIL]
#      b) 最后打印 "总结：22/22 项检查通过"
#      c)  shell 退出码为 0（PowerShell: echo $LASTEXITCODE; bash: echo $?）
```

**复现记录表**（复现者填写，可附截图/日志）：

| 项 | 值 |
|---|---|
| 复现者 | |
| 日期时间（含时区） | |
| 操作系统 / Python 版本 | |
| deepagents 版本（`pip show deepagents`） | |
| 命令与退出码 | |
| PASS/FAIL 计数 | |
| 与作者留证的差异（若有） | |
| 卡点与耗时 | |

## B. 互评清单（10 项，每项 是/否 + 一句话依据）

**可复现性（对应"陌生人 3 分钟能跑通吗"）**

1. README 是否**一句话讲清**项目是什么、Demo 验什么？
2. 是否只靠 README 的 3 条命令就跑通，无需问作者？
3. 失败时是否有**明确信号**（FAIL 行 + 非零退出码），而不是静默通过？

**方法论（固定输入 + 明确预期）**

4. 固定输入是否写死并可核对（文件指纹 / 话语 / 数据）？
5. 预期值是否**运行前声明**（EXPECTED 表），而非跑完再写？
6. 证据是否自洽（`summary.json` 中逐项"预期 vs 实际"可逐条核对）？

**课程要求（Basic 档：≥3 能力、≥3 改造）**

7. Demo 是否综合**至少 3 项**课程能力且能在输出中对应上？（工具调用 / 长期记忆 / 人工审批 / 风险分层 / Skills）
8. 作者是否列出相对原模板**至少 3 项实质改造**（见 Task8 打卡笔记"改了什么"），且每项有证据指向？

**诚实度（Advanced 档：失败处理与能力边界）**

9. 失败与边界是否如实呈现（含"工具异常会冲出 invoke""权限是工具级"这类对作者不利的事实）？
10. 作者是否说清了**不证明什么**（scripted ≠ 真实模型语义；n=1；非安全证明）？

## C. 反馈模板

```
复现者 / 时间 / 环境：
通过情况：__/22 PASS，退出码 __
最有价值的一点：
最困惑的一点：
按清单第 __ 项不认可，原因：
建议（可选）：
```

## D. 作者本轮自留证

- 正式运行：`../evidence/task8/demo-run-01/`（作者本机，venv + 锁定依赖）
- "陌生人视角"运行：`../evidence/task8/demo-run-02-stranger/`（新进程、默认参数、陌生 cwd，只凭 README 命令）
- 两次运行 22/22 一致；指纹与退出码见各目录 `registration.json`
