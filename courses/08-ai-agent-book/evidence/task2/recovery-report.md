# 3-1／3-2定向恢复实验

恢复后覆盖：240/240；状态：passed。

原配置238/240结果完全保留。本次只恢复两项技术失败，不重跑已完成的低分答案。

多药物第三次记忆更新改用流式接收，记忆预算仍为6000 tokens。紧急医疗既有答案保持不变，评审预算由2400提高为8192。新调用读取超时600秒，模型、官方提示和评分标准不变。

这是混合配置下的恢复覆盖，不是原配置统一预算的240项重跑。质量分数为独立模型判断，流程通过也不等于每题正确。流式接口若未返回usage，新增调用的token与成本记为不可用，不作零成本解释。

| 模式 | 完成数 | 质量通过率 | 平均奖励 | 幻觉否决率 |
|---|---:|---:|---:|---:|
| notes | 60/60 | 55.0% | 0.723 | 13.3% |
| enhanced_notes | 60/60 | 70.0% | 0.789 | 10.0% |
| json_cards | 60/60 | 76.7% | 0.789 | 16.7% |
| advanced_json_cards | 60/60 | 66.7% | 0.773 | 13.3% |

## 恢复结果

- layer2_07_multiple_medications / advanced_json_cards：completed
  {"scores": {"precision": 2, "recall": 1, "reasoning": 1, "proactivity": 2}, "hallucination_veto": true, "passed": false, "reward": 0.0}
- layer3_13_emergency_medical_cascade / json_cards：completed
  {"scores": {"precision": 1, "recall": 2, "reasoning": 2, "proactivity": 2}, "hallucination_veto": true, "passed": false, "reward": 0.0}