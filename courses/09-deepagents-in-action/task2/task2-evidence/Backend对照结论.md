# Backend 生命周期对照

执行日期：2026-10-15；机器 vm-60；详细版本见原始 JSON。证据目录：task2-backend-compare-20261015T074342Z。

## 验证什么

同一虚拟路径、同一 UTF-8 内容在新进程中是否能读回，以及读回依赖什么存储。

## 改了什么

使用课程 ScriptedChatModel 指定同一文件工具调用，真实 create_deep_agent 执行。未调用远端模型，无 Checkpointer，每个阶段创建新进程和新 Agent。State 恢复只通过 invoke 的 files 字段注入，不传旧聊天。JSON 导出与加载由实验程序完成。

## 实际结果

| 条件 | 结果 |
|---|---|
| State 新进程，不传文件状态 | 工具报告 File not found，符合预期 |
| State 新进程，传入原 files 状态 | 成功读回 |
| Filesystem 新进程，沿用同一 root_dir | 成功读回 |

写入阶段分别在独立进程完成。五个阶段 PID 各不相同。State 内容和磁盘内容逐字节等价，成功读回后的内容哈希与写入时一致。

## 证据与解释

summary.json 是总验收；其余阶段 JSON 包含输入字段、真实工具请求和返回。state-files.json 是实验程序显式导出的状态，不是 StateBackend 自动落盘的证明。disk/reports/raw.md 是 Filesystem 的真实磁盘文件。

结论：相同虚拟路径不代表同一存储。State 需要原文件状态；Filesystem 需要保留磁盘文件并使用相同路径映射。实验未验证数据库 Checkpointer、Store、机器重启或真实模型遵从能力。

## 下一步修正

整理完整 Task2 打卡时，将这些机制对照与前面的真实模型实验区分；基础的资料分析和 Agent 修改闭环仍应单独验收，不能以本对照替代。
