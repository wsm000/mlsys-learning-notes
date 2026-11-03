# Task2 实验源码快照

这些脚本保留 vm-60 实际执行版本，用于核对实验过程；并非开箱即用的软件包。`probe.py` 可通过环境变量 `RADEON_API_KEY` 或隐藏输入测试 AMD 接口。运行教学脚本前设置 `RADEON_API_KEY`、按需设置 `RADEON_MODEL`，安装所需依赖，并调整脚本中的实验目录。旧版凭据文件路径仅为运行环境记录，仓库不包含凭据。

全量验收依赖教程固定版本 `dbc046eb896ac4e39aa19c7774c8bf49583b89a6` 的原始用例、提示和评审逻辑。来源与实际运行源码、检查点和哈希见 `../../evidence/task2/Task2学习与实验证据.zip`。恢复脚本只补跑两项，不能替代从头执行。请使用自己的 API 凭据；重新执行会产生 API 用量。

## 十二项机制教学补充

`experiment03_equivalent.py`至`experiment12_equivalent.py`及对应followup/control脚本为本轮实际运行快照，按编号执行。脚本保留vm-60目录，跨环境需调整ROOT；输入与原始结果见`../../evidence/task2/mechanisms/`。核心依赖包括torch、transformers、numpy、scikit-learn、PyYAML、annoy、hnswlib；本地缓存模型为BAAI/bge-small-zh-v1.5与Qwen/Qwen3-0.6B。脚本强制离线时需要先准备模型缓存。

3-3读取固定教程版本的脱敏样本；3-9、3-11读取官方YAML；3-11还依赖此前3-1的真实卡片检查点。固定源码版本与原检查点在既有证据归档中说明。其他部分采用明确标注的自建教学数据、规则或模型替代。机制运行和回答质量分开记录，不以失败后的人工说明替换模型原始答案。
