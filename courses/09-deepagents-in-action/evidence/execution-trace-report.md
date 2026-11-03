# Advanced：本地执行Trace与边界对照

## 本轮验证
在vm-60课程主应用的build_agent入口，使用固定DeepSeek-V4-Flash和同一工具，测试3条与100条问题。每个请求使用新Agent实例，不传旧消息。采用LangChain回调记录chain/model/tool跨度、UTC开始结束时间、单调时钟耗时、父调用ID、工具请求和返回；不依赖LangSmith账户。

| 用例 | 模型调用 | 工具调用 | 总耗时 | 结果 |
|---|---:|---|---:|---|
| 最近3条 | 2 | analyze_price_volume一次，约0.77毫秒 | 28.40秒 | 参数与数值正确，未破坏原功能 |
| 最近100条 | 2 | ls一次，约0.76毫秒 | 28.65秒 | 说明限制并询问，未替代分析；多余ls使效率验收未通过 |

单次模型调用耗时分别为6.90/21.47秒与11.52/17.10秒。它们包括API往返与服务端处理，不等于纯模型推理耗时。本轮主要时间在模型调用，不在CSV计算。样本各一次，不能推断稳定性或平均性能。

## Trace核验
verify_execution_traces.py检查跨度ID唯一、父ID可解析、开始结束闭合、耗时非负、正常调用请求与返回相符、100条无替代分析或修改工具。人工检查最终100条回答说明仅10条数据、需要200条，并询问是否接受5条。

原始100条记录中的behavior_check=false是当时“必须零工具调用”的严格标准结果，保留未改；之后分别评价业务规则与多余调用，避免把效率问题与擅自替代混为一谈。

## 证据
- execution-trace-normal3.json
- execution-trace-unsupported100.json
- execution-trace-summary.json
- 空/超长CLI入口：input-boundaries.json
- 提示词修正前后：window100-before.json、window100-after.json

## 下一步
多余ls证明“提示无需额外操作”不是硬约束。后续可评估精简工具暴露或显式路由，再用独立测试集验证。本轮只是人工修正与验证，未实现Agent自动递归改进。完整回调Trace不等同于完整供应商内部链路或LangSmith云端记录。

课程资料：https://datawhalechina.github.io/deepagents-in-action/
SDK资料：https://docs.langchain.com/oss/python/deepagents/overview
