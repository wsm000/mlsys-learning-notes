# 流式恢复的证据核验

原配置基线的240份检查点字节均未改变，其中238份完成、2份未完成。新目录中的恢复记录补齐两项，含模型不合格回答及其0分，不替换成更好的答案。

新请求均记录为stream=true；实际SSE分片分别保存。以下核验不依赖拼接器自身的判断，重新从原始分片构造content与reasoning，检查结束原因、模型身份和usage字段。

| 调用 | 模型 | 最终内容字符 | 推理字段字符 | usage可用 |
|---|---|---:|---:|---|
| 3-1/3-2 answer layer2_07_multiple_medications advanced_json_cards | DeepSeek-V4-Flash | 1646 | 0 | True |
| 3-1/3-2 memory update layer2_07_multiple_medications advanced_json_cards session 3 | DeepSeek-V4-Flash | 15428 | 0 | True |
| 3-1/3-2 independent judge layer3_13_emergency_medical_cascade json_cards | Qwen3.8-Flash-Next | 2957 | 15253 | True |
| 3-1/3-2 independent judge layer2_07_multiple_medications advanced_json_cards | Qwen3.8-Flash-Next | 3189 | 6383 | True |

尽管请求reasoning_effort=none，本次Qwen返回流仍包含reasoning字段。这是本次观测，不能据此断言所有模型或接口均如此；预算不能仅按可见JSON长度估计。

流式接收与超时延长同时发生，因此这里只能说明组合恢复成功，未分别验证两项改动的因果贡献。提高评审上限没有修改原始答案或评分标准，但混合预算的汇总不应称为统一配置对照。

原始官方实验指针仍保留partial；恢复实验有单独的combined/validation/latest.json。两者不互相冒充。