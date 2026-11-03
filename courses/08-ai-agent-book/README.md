# AI Agent：从入门到工程实践

学习主线：**Agent = LLM + Context + Tools**。

- 课程资料：[bojieli/ai-agent-book](https://github.com/bojieli/ai-agent-book)
- 打卡平台：[Datawhale 活动 77](https://www.datawhale.cn/activity/77)
- 学习周期：活动通知为 10 月 12 日至 10 月 30 日；具体任务截止时间以活动公告为准。

## 笔记索引

| Task | 内容 | 状态 |
|---|---|---|
| [Task0](notes/task00-agent-basics-and-environment.md) | Agent 基础、环境准备、官方离线演示、MiMo 上下文对照 | 已记录实跑结果 |
| [Task1](notes/task01-context-engineering-reflection.md) | Chapter 2：上下文工程 | 学习心得、注意力图、六组提示消融与资料快照压缩实验 |
| [Task2](notes/task02-memory-rag/Task2_十二项实验机制总览.md) | Chapter 3：用户记忆与 RAG | 十二项机制学习与等价实验；[全量记忆验收说明](notes/task02-memory-rag/从238到240_全量验收补充报告.md)：原配置238/240，恢复后240/240 |
| [Task3](notes/task03-tools-and-mcp.md) | Chapter 4：Tools 与 MCP | 工具分类、MCP 接口与执行工具实测；已注明未验证范围 |
| [Task4](notes/task04-coding-agent-and-general-agent.md) | Chapter 5：Coding Agent 与通用 Agent | 闭环、Harness 与代码元能力图解；折扣校验和销售报告两个相关实验及复现代码 |
| Task5 | Chapter 10：Multi-Agent 协作 | 待学习 |

## 目录

- `notes/`：学习笔记与实验分析。
- `code/task0/`：课程原始 Agent 的 MiMo 提供商适配入口。
- `evidence/task0/`：真实截图、对照汇总和从运行记录提取的证据。

Task2阅读入口：[十二项机制总览](notes/task02-memory-rag/Task2_十二项实验机制总览.md) · [基础科普报告](notes/task02-memory-rag/从记忆到证据.md)。机制运行完成不等于官方十二项原样全量验收或所有回答通过。
