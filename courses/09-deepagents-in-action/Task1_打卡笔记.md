# Task1：我的第一个工具 Agent

## 验证什么
验证 Deep Agent 能根据自然语言选择自定义 Python 工具、填写参数、执行函数，并基于返回结果回答。业务场景是本地模拟数据的量价比较；不涉及真实投资结论。

## 改了什么
参考课程第1–2章和 SDK 快速上手，实现 analyze_price_volume(symbol, window)。用 ChatOpenAI 适配 AMD 的兼容接口，由 create_deep_agent 提供调用循环。环境在 vm-60：Python3.11.15、Deep Agents0.7.24。MiMo网关请求失败后，仅更换模型为 DeepSeek-V4-Flash，凭据、工具和数据不变。

进一步用课程AgentSeek default模板启动主应用，锁定环境Python3.13.13/Deep Agents0.7.8，接入相同工具。针对100条请求擅自缩为5条的失败，人工修改系统提示词，要求先说明限制并征求同意。增加本地执行Trace回调与跨进程文件持久化实验。

## 实际结果
第一轮：模型请求 DEMO、window=5；工具计算涨幅6%、前后均量1000/1600、比值1.6；模型正确引用结果并说明模拟数据局限。

第二轮：问题改为最近3条。模型请求 window="3"，框架解析为整数3后执行；工具返回涨幅2.912621%、前后均量1300/1700、比值1.307692；回答与返回值一致。这说明模型能随需求调整参数，但不能据此断言模型严格遵守整数类型。

主应用真实调用正常结束，四组件已对应到源码。空输入和4001字符在CLI入口被拒绝；100条请求修正后会询问替代方案，但一次主应用Trace仍发现多余ls。正常请求约28.4秒，100条约28.65秒，工具执行均不到1毫秒，耗时主要在模型API往返。

课程第1–2章三份原始notebook按锁定Python3.12.13/Deep Agents0.7.22环境全部通过（offline），51项对应官方测试通过。离线脚本模型只验证框架机制，真实模型能力以独立DeepSeek实验为证。源码提交f93fe0a8be843eae77efaf85448035ddc54ff7ea。

## 证据与解释
- [两个模型稳定性选做对照](evidence/model-comparison-report.md)：DeepSeek与GLM各3次，同一API及任务；均2次成功、1次超时，未发现谁更稳定。
- [课程三份notebook执行报告](evidence/course-notebooks/report.json)
- [Notebook运行说明与验收记录](evidence/course-notebooks-run.md)
- [完整本地执行Trace与边界对照](evidence/execution-trace-report.md)
- [四组件定位](组件定位.md)
- [Backend写入读回实验](evidence/backend-experiment.md)：StateBackend保存于返回状态，检查路径未落盘。
- [磁盘跨进程持久化实验](evidence/persistence-experiment.md)：FilesystemBackend写入磁盘，旧进程退出后，新进程只凭路径读回，文件哈希一致。
- [首次成功消息轨迹](evidence/deepseek-success-trace.json)
- [3条记录对照轨迹](evidence/deepseek-window3-trace.json)
- [工具结果](evidence/tool-result.json)
- [工具测试](evidence/tool-tests.txt)
- [远端实验记录](evidence/vm60-progress.md)

请求中的工具调用ID与ToolMessage中的tool_call_id一致，证明返回对应本次请求；返回数值与独立执行工具一致。最终消息轨迹不等同于包含时间跨度的完整LangSmith Trace。

模型生成工具请求，Harness执行调用循环，Python函数负责读取与计算，模型再解释结果。框架提供搭建组件与执行支持；Harness进一步组合上下文管理、文件工具等通用能力。

## 下一步修正
减少无必要的ls和计划调用；用更多未见输入重复测量，防止单例改好却损伤其他任务。CLI长度限制尚未同步验证AG-UI入口。Agent修改工具代码并验证改进属于后续拓展，尚未实现。



## 参考资料
- [课程教程](https://datawhalechina.github.io/deepagents-in-action/)
- [课程源码](https://github.com/datawhalechina/deepagents-in-action)
- [Deep Agents快速上手](https://docs.langchain.com/oss/python/deepagents/quickstart)
- [AgentSeek](https://github.com/ob-labs/agentseek)


