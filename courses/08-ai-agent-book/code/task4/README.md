# Chapter 5：两个最小相关实验

博客：[从一行折扣公式，到一个能完成任务的 Agent](../../notes/task04-coding-agent-and-general-agent.md)。

要求：Python 3.9+，仅标准库，无需模型 API、密钥或第三方依赖。以下命令均从本目录执行。

## 一次复现全部关键观察

```powershell
python verify_experiments.py
```

验证器在临时副本中执行，不修改当前源码、CSV 或已保存报告。应依次观察：

1. 无范围校验：七项中三项异常测试失败。
2. 两个越界比较：仅 NaN 测试失败。
3. 闭区间校验：七项全部通过。
4. 销售草稿：已知地区 440 元、待确认 60 元、总额 500 元，状态未完成。
5. 应用教学模拟确认：华东 120、华南 300、华北 80、待确认 0，总额仍为 500 元；原始 CSV 哈希不变。

关键结果写到 `verification.json`（默认不跟踪）。`published-verification.json` 是上传准备时的实际复验记录。

## 单独运行

```powershell
python -m unittest discover -s coding -v
python sales/build_draft.py
python sales/build_confirmed.py
```

后两条会重新生成对应检查文件与报告，但不改原始 CSV。v2 依赖已给定的教学模拟确认，不能据此认为程序自行查明了 C002 的真实地区。

## 文件与证据

- `coding/cart.py`、`coding/test_cart.py`：最终实现与七项测试。
- `coding/*before.txt`、`coding/*after.txt`：早期失败与修复记录。
- `coding/comparison-or.txt`、`coding/comparison-restored.txt`：NaN 对照原始输出。
- `sales/data/`：七条模拟记录，原始数据只读。
- `sales/业务口径.md`：退款、币种与地区规则。
- `sales/inspect_data.py`：源内校验；执行成功不代表业务校验通过，须读取 `final_ready` 与 `errors`。
- `sales/build_draft.py`：保留待确认金额并生成未完成草稿。
- `sales/region-confirmations.json`：绑定源版本的独立修订层，仅为教学模拟。
- `sales/build_confirmed.py`：核对版本、应用修订、重新汇总并保存 v2。
- `sales/*checks.json`：对账、版本哈希、状态与修订依据。

旧测试日志中的本机文件路径已替换为 `<workspace>`，断言、测试名和通过/失败状态保留；历史终端中的中文乱码不影响测试结论。本实验没有实现独立 Agent 调度器、事务数据库、邮件发送或真实权限沙箱。正文中的相关设计不属于已实测能力。
