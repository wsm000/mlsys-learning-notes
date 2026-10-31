# DataFlow Task5 打卡 · 构建高质量训练数据（Text pipeline）

**任务**：dataflowTask5 构建高质量训练数据 —— Text pipeline：预训练数据过滤 / SFT 数据过滤 / SFT 数据合成
**环境**：vm-60（RTX 4090 D 24GB）· open-dataflow 1.0.11 · 2026-10-31 实测
**代码**：code/task5/（build_corpora / pt_filter / sft_filter / sft_synthesis / verify / unit_check + env.sh + run_all.sh）
**证据**：evidence/task5/（44 个文件：3 条 run 日志、verify_task5.json、unit_check_line_filters.json、语料清单、三条管线逐步 jsonl）

## 一句话结论

三条官方 pipeline 全部在 vm-60 真实跑通：预训练过滤 44 条设计语料 -> 14 条存活（25 步漏斗，每一刀的落点都有 id 级记录）；SFT 过滤 16 条 -> 11 条（词数 -> IFD -> LLM 质量分三级）；SFT 合成用知识树从零生成 9 条三档难度数据、critique-refine 改写后 LLM 打分全过。最大的工程约束是 vm-60 外网被劫持，用 Windows 主机 CONNECT 隧道 + hf-mirror + 离线 HF cache 解决。

## 验证了什么

1. **预训练数据过滤**（官方 GPU 版 25 步全通）：3 个 refiner 先改写内容，19 个规则过滤器逐级删减，PairQual 质量分（BERT 回归）、fasttext 语言过滤（西班牙语文档在此被砍）、MinHash 5-gram 去重（精确重复 + 只换 1 个词的近似重复 Jaccard=0.97 都被砍，保留首条）。
2. **SFT 数据过滤**（官方 GPU 版 3 步）：WordNumberFilter(output 20-1000 词) -> SuperfilteringFilter(gpt2 IFD，官方 [0.5,1.0]，指令不对应样本 IFD=1.25 被砍) -> 质量分过滤（deita 模型因镜像限流拿不齐，按缓存完整性自动退化为官方 API 管线同款的 AlpagasusFilter，LLM 0-5 分 [3,5]）。
3. **SFT 数据合成**（官方 API 版 3 步）：CondorGenerator 按知识树(topic/domain/theme + 任务场景)采样生成 [Easy]/[Medium]/[Hard] 三档问题并逐题生成答案 -> CondorRefiner 先 critique 再 refine 两阶段改写 -> AlpagasusFilter 打分保留 3-5。

## 改了什么（相对官方 pipeline）

- 端点：OpenAI -> AMD Radeon 兼容端点 DeepSeek-V4-Flash（经 Windows 隧道）；官方 max_workers=100 在 20 RPM 限流下会 429 风暴，改进程内匀速限流（rpm=10、workers=2）+ 请求/响应审计日志（不含密钥）。
- 路径：官方相对路径 example_data/... 改为本任务设计语料；PairQualFilter 官方没传 model_cache_dir（默认 ./dataflow_cache 会现拉权重），显式指向离线预置缓存。
- 证据：每步打印 in/out 行数；verify_task5.py 把 25 步 cache 还原成 id 级"被谁砍掉"对照表；unit_check_line_filters.py 对三个行比例过滤器做 refiner 前后对照的单位核对。

## 实际结果（数字）

- PT：44 -> 14；规则过滤器砍 26 条、PairQual 砍 7 条、语言过滤砍 1 条、MinHash 砍 3 条；缺陷清单吻合 40/44（4 条差异全部有解释）；PairQual 分数 n=18, min=2.02, max=5.69, mean=3.95。
- SFT：16 -> 11；词数砍 4 条、IFD 砍 1 条（不对应样本 1.25）、质量分砍 0 条（LLM 裁判对重复/敷衍回答也打 >=3，退化路径下的实测边界）。
- 合成：3 个 prompt -> 9 条（Easy/Medium/Hard 各 3），refine 后 6 条变长 3 条变短，44 次 API 请求（39 成功、5 次 504 被重试吸收）、14 分 33 秒、均延迟 29.7s。

## 踩到的坑（都有证据）

1. vm-60 直连外网被劫持（HTTPS 重置/假 IP -> 10.0.31.130 门户页），本机 FlClash 代理 474% CPU 空转（kill 后负载 14->1.4）；解法：Windows 主机 Node CONNECT 代理 + ssh -R 反向隧道 + hf-mirror 下载 + HF_HUB_OFFLINE 离线加载。
2. 官方 PT 管线把 RemoveExtraSpacesRefiner 放最前，" ".join(text.split()) 压平换行，导致三个"行比例"过滤器实际看到单行文本——省略号/javascript 缺陷文档逃过本命过滤器改落 PairQual（bullet 查行首仍能砍）。已用算子级单元核对（refiner 前后对照）固定证据。
3. PairQual-en 输出区间约 [-8, 5.7]，官方 min_score=2 贴近分布顶部：干净百科 2.6-5.7 可过，营销垃圾/乱码 -7 被砍，但带 URL（1.28）或词表式文本（-3.7）被误伤。
4. hf-mirror 的 xet 后端不支持 HTTP Range（断点续传 416），且批量下载后对本 IP 限流（0.2MB/s）；deita-quality-scorer 13.5GB 拿不齐，SFT 过滤第三步按缓存完整性自动退化为 AlpagasusFilter。
5. 同一 LLM 既当生成器又当裁判员会自利：Condor 合成 + refine 后的 9 条样本 Alpagasus 全打 4.0（零方差），质量门槛需要外部信号兜底。

## 下一步修正

- 行比例过滤器要么前置于 refiner，要么改不依赖换行的判据；PT 管线可预置 NLTK_DATA 消除 init 的 394s 下载超时。
- PairQual 阈值按语料分位数定并固定评分器版本；URL/表情文档先过 refiner 再评分。
- 合成数据加独立裁判/执行验证/人工抽检；中文场景需另配中文知识树（官方 Condor 标签以英文为主）。