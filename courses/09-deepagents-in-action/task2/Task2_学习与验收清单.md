# Task2：让成果留在文件里

状态：已开始准备，尚未完成本期实验或提交。

## 学习范围

课程第 3 章：虚拟文件系统与上下文管理、可插拔存储后端。
教程：https://datawhalechina.github.io/deepagents-in-action/

本期先完成 Basic，再做 Advanced。遵循《项目执行规则.md》的基础打卡优先顺序。

## 已有基础

Task1 本地归档位于 publish-agent-task1/courses/09-deepagents-in-action/。
其中 evidence/backend-experiment.md 记录了 StateBackend 写入和读回；
evidence/persistence-experiment.md 记录了 FilesystemBackend 跨进程读回。
这些是历史记录，本次仅阅读，未重新执行。现有记录尚不能证明 Task2 的修改闭环完成。

## Basic 验收

- [x] 核对用户提供的三档目标和课程目录。
- [ ] 确认主模板实际 Backend 配置、运行环境和依赖版本。
- [ ] Agent 写入资料与报告，保留原始工具调用及返回。
- [ ] Agent 读回并引用文件中的内容。
- [ ] Agent 修改一个可定位的字段或段落，再次读回。
- [ ] 在项目真实目录找到文件，记录虚拟路径、root_dir 和磁盘路径。
- [ ] 保存修改前后内容与差异，检查其他内容是否意外丢失。
- [ ] 按“验证什么、改了什么、实际结果、证据与解释、下一步修正”整理打卡。
- [ ] 材料达到可立即提交状态。

建议实验：沿用 Task1 模拟量价报告，先保存 window=3 的结果，再根据 window=5 的新计算结果修改报告。数值必须来自工具返回，并保留模拟数据说明。

## 三天安排

1. 第一天：确认配置，完成写入、读回和磁盘定位。
2. 第二天：完成修改与再次读回，整理 Basic 证据和打卡文字。
3. 第三天：基础材料就绪后，对比 State 与 Filesystem；有余力再加入 Store。

## Advanced 设计（基础验收后进行）

使用相同内容和虚拟路径，比较同次调用、新调用、新线程和新进程的读回结果。记录是否配置 Checkpointer，以及是否沿用原状态；不能把 StateBackend 简化成“一结束就消失”。

Filesystem 记录是否沿用同一 root_dir；Store 记录具体实现、namespace 及存储服务生命周期。后端名称本身不足以证明持久化。

## 证据要求

每轮保存输入、版本、Backend 配置、工具调用参数与返回、落盘内容和验证结果。失败也保留，不能用手动写入的文件充当 Agent 成功写入的证据。
