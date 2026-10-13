# 课程9 · vm-60 实验记录（2026-10-13）

## RSI扩展
- 新增calculate_rsi工具并保留analyze_price_volume；明确区分period与window。
- vm-60执行6项测试全部通过，覆盖Wilder手算平滑、上涨100/下跌0/平盘50、非法输入和默认14期数据不足。
- DeepSeek按用户明确指定的3期调用calculate_rsi，返回RSI约94.43，最终回答引用结果并注明模拟数据与短历史局限。
- 成功消息轨迹：deepseek-rsi3-trace.json。
- 默认14期不足仅通过确定性工具测试，尚未另跑该分支的真实模型调用。

## 最新结果：DeepSeek 调用成功
- 参数对照实验：用户改问最近3条记录，模型实际请求 window=3，1次成功返回。涨幅2.912621%，前后均量1300/1700，比值1.307692，与直接工具计算一致。证据：deepseek-window3-trace.json。
- 类型细节：原始模型参数实际为字符串 "3"，工具返回的 window 为整数3。框架参数解析接受并转换了数字字符串。本次不能声称模型严格输出了符合schema的整数。已核对调用ID、返回类型与上述数值。
- 使用同一AMD凭据，仅更换模型为 DeepSeek-V4-Flash。
- 截图名称 DeepSeek-V4-Flash-0731 返回404；通过 models.list 确认真实API ID后修正。
- 1次 analyze_price_volume 请求，1次匹配的成功工具返回。
- 最终回答正确引用6%涨幅、1000/1600均量、1.6比值，并注明模拟数据与描述性局限。
- 消息轨迹：deepseek-success-trace.json。该文件是最终消息轨迹，不是完整LangSmith跨度Trace。
- 本项目自己的工具调用闭环已完成；课程第1–2章全部示例及主应用尚未完成。
- 单次成功不足以比较两模型调用稳定性；MiMo此前网关失败不能归因于模型推理能力。

- 远端目录：/home/simin/course09-quanty
- 独立环境：.venv，Python 3.11.15；Deep Agents 0.7.24。
- 硬件已确认：RTX 4090 D。当前模型使用 MiMo API，不占用 GPU 推理。
- 复用课程8远端凭据文件，仅在内存读取，不复制回本地。
- 远端3项工具测试通过。
- 第一次真实 Agent 请求：HTTP 504 Gateway Time-out。
- 第二次真实 Agent 请求：HTTP 502 Bad Gateway。
- 两次均在模型阶段失败，没有成功工具调用证据，Lite 未完成。
- 增加 check_vm60.py 做普通小请求连通性对照；结果另存 connectivity.json。
- 对照结果：普通模型小请求也发生 APITimeoutError（30秒超时）。问题不只出现在 Agent 请求中；尚不能确定网关或上游故障原因。

## 重新运行
```text
ssh vm-60
cd /home/simin/course09-quanty
.venv/bin/python run_vm60.py
```
成功后检查 outputs 中消息轨迹：analyze_price_volume 请求及对应 tool_call_id 的成功返回。还需人工核对回答与数值一致。
