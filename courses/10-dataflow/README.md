# DataFlow · AI-Ready Data 实战（18 天七段任务）

> 教程主仓：https://github.com/OpenDCAI/DataFlow
> 图文教程（飞书）：https://wcny4qa9krto.feishu.cn/wiki/I9tbw2qnBi0lEakmmAGclTysnFd
> 视频教程（B站）：https://space.bilibili.com/3546929239689711
> 官方文档：https://opendcai.github.io/DataFlow-Doc/zh/

按任务组织笔记、代码与证据。所有实验均在 vm-60（RTX 4090 D 24GB）真实执行；本地资源不可达的部分（外网 API、HF/modelscope 下载）如实标注替代方案。

| Task | 主题 | 目录 |
|---|---|---|
| Task 1 | PDF→模型（1.1–1.2.4） | `~/dataflow-task1`（vm-60），归档见 `evidence/task1` |
| Task 2 | PDF→结构化文本→QA | `~/dataflow-task2`（vm-60），归档见 `evidence/task2` |
| Task 3 | 多模态数据：公式/图表/版式 VQA 处理（3天，已完成） | [notes/task03](notes/task03-多模态数据公式图表版式VQA.md) · [code/task3](code/task3) · [evidence/task3](evidence/task3) · [打卡](evidence/task3/dataflow-task3-checkin.md) |
| Task 4 | 构建 RAG 与 Text2SQL 数据：AgenticRAG 数据合成 + Text2SQL refine/gen（已完成） | [notes/task04](notes/task04-构建RAG与Text2SQL数据.md) · [场景数据需求](notes/task04-不同AI应用场景数据需求.md) · [code/task4](code/task4) · [evidence/task4](evidence/task4) · [打卡](evidence/task4/dataflow-task4-checkin.md) |
| Task 5 | 构建高质量训练数据：预训练数据过滤 + SFT 数据过滤 + SFT 数据合成（已完成） | [notes/task05](notes/task05-构建高质量训练数据.md) · [code/task5](code/task5) · [evidence/task5](evidence/task5) · [打卡](evidence/task5/dataflow-task5-checkin.md) |

## Task3 结论速览

- **解析层（公式/图表/版式）**：MinerU2.5-2509-1.2B 本地 VLM 解析真实可用——LaTeX 公式、HTML 表格、带题注插图、版面框全覆盖（4 页 40 项 / 2 页 64 项）。
- **抽取层（QA）**：本地 qwen3:1.7b 无法遵守 id 协议，产物结构有效但内容错误；教程原版需 gemini-2.5-pro。链路 22/22 验收通过，抽取质量如实标注为不可用。
- 网络限制下的完整替代方案：Verge 代理下载模型权重 + tiktoken 缓存 → scp 上台；MinerU 本地 vlm-transformers 后端替代 API/Flash。

## Task5 结论速览

- **预训练数据过滤（官方 GPU 25 步）**：44 条设计语料 -> 14 条存活；规则过滤器砍 26 条、PairQual 质量分砍 7 条（干净百科 2.6-5.7 vs 垃圾/乱码 -7，官方 min_score=2 贴近分布顶部）、fasttext 砍 1 条西班牙语、MinHash 砍 3 条（近似重复 Jaccard=0.97）。缺陷清单 id 级吻合 40/44，4 条差异全部可解释。
- **SFT 数据过滤（官方 GPU 3 步）**：16 -> 11；词数砍 4 条、IFD 砍 1 条（指令不对应 1.25 > 1.0）；deita 13.5GB 模型因镜像限流拿不齐，按缓存完整性自动退化为官方 API 管线同款 AlpagasusFilter（LLM 0-5 分）——LLM 裁判对重复/敷衍回答也打 >=3，重复度需专用指标兜底。
- **SFT 数据合成（官方 API 3 步）**：知识树采样 3 prompt -> 9 条三档难度样本，critique-refine 两阶段改写（6 条变长/3 条变短），Alpagasus 全打 4.0；44 次请求 14.5 分钟（20 RPM 限流 + rpm=10 客户端限速，≈4.3 请求/分钟）。
- **两个可复现的管线行为发现**：① 官方把 RemoveExtraSpacesRefiner 放最前会压平换行，三个"行比例"过滤器只剩查行首的 bullet 有效（单元核对固定证据）；② 同一 LLM 既生成又裁判会自利（合成数据零方差高分），质量门槛要外部信号。
- **环境约束**：vm-60 外网被劫持（假 IP/门户页 + FlClash 474% CPU 空转），改用 Windows 主机 CONNECT 隧道 + hf-mirror + 离线 HF cache；deita 因镜像限流 + xet 不支持 Range 未能齐装，走了文档化的退化路径。

## Task4 结论速览

- **AgenticRAG 数据合成**：4 条真实文档 → 4 条 QA（13 字段全非空），F1=[1.0,1.0,0.7826,0.0]（均值 0.696），`llm_score` 全 0 + `golden_doc_score` 1~2，把「必须检索 + 文档可答」双过滤固化成数据条件；66 请求/654.7s 是 20 RPM 端点的真实吞吐。
- **Text2SQL 数据流水线**：refine 10 步全通（4 条输入 → 一致性过滤后 3 条、9/10 列非空、0 次 429）；gen 模式 3 个 SQLite 库 ×1 条 SQL → 一致性过滤后 2 条，9 步全通。
- **端点限流是最大工程约束**：AMD 端点账号级 20 RPM + 模型级并发上限；smoke 的 LLM 最密集 step10 曾被 429 中断，正式 run 靠缩采样/低并发/退避 0 次 429 完成。token-F1 对日期/中文格式敏感（`2017-08-30` vs `August 30, 2017` 判 0），不能直接当 RAG 的 RL 奖励。
