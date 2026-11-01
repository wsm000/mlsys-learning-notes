# Task5 · 构建高质量训练数据（Text pipeline：预训练过滤 / SFT 过滤 / SFT 合成）

> 教程主仓：https://github.com/OpenDCAI/DataFlow
> 官方文档：[预训练数据过滤](https://opendcai.github.io/DataFlow-Doc/zh/guide/pt_filter_pipeline/)、[SFT 数据过滤](https://opendcai.github.io/DataFlow-Doc/zh/guide/sft_filter_pipeline/)、[SFT 数据合成](https://opendcai.github.io/DataFlow-Doc/zh/guide/sft_synthesis_pipeline/)
> 环境：vm-60（Linux x86_64，RTX 4090 D 24GB）· open-dataflow 1.0.11 · Python 3.10.12 · 2026-10-31 实测
> 模型：AMD Radeon 兼容端点 DeepSeek-V4-Flash（SFT 合成/评分）· 本地评分器 PairQual-Scorer-en / gpt2(Superfiltering) / fasttext-lid
> 本任务产出：三条官方 pipeline 实测 + 设计语料漏斗证据 + 算子级单元核对 · [代码](../code/task5) · [证据](../evidence/task5) · [打卡](../evidence/task5/dataflow-task5-checkin.md)

## 0. 环境与证据边界（先说能说什么、不能说什么）

| 项 | 实际情况 |
|---|---|
| 机器 | vm-60：Linux x86_64，NVIDIA RTX 4090 D（24GB），Python 3.10.12，open-dataflow 1.0.11 |
| 网络 | **vm-60 直连外网被劫持**（HTTPS 被重置/劫持到 10.0.31.130 门户页；本机 FlClash 代理进程 474% CPU 空转，kill 后负载从 14 降到 1.4）；改用 Windows 主机 CONNECT 隧道：`vm-60:18899 --ssh -R--> Windows:18888(node CONNECT proxy) --> 目标站点`，AMD API 经隧道实测 200 |
| 模型获取 | HF 直连不可达 → hf-mirror.com 经隧道下载到 `/mnt/nfs/task5_hf`（NFS 共享盘；预置 HF cache 布局 + `HF_HUB_OFFLINE=1` 离线加载，gpt2/PairQual/fasttext 三个评分器共 ~2.2GB） |
| 生成模型 | DeepSeek-V4-Flash（AMD Radeon OpenAI 兼容端点），经隧道调用；密钥只从远端 key 文件进进程，不落盘不外传 |
| 评分器 | PairQual-Scorer-en（BERT 回归，GPU）、gpt2（Superfiltering IFD，GPU）、fasttext lid.176（语言识别，CPU） |
| deita 让步 | 官方 SFT 过滤第三步的 hkust-nlp/deita-quality-scorer（Llama-7B fp32 13.5GB）因 hf-mirror 对本次 IP 限流（0.2MB/s）未能完整拉取，**退化用官方 API 合成管线同款的 AlpagasusFilter**（LLM 0-5 分，官方阈值 [3,5]）顶替同一"质量分过滤"角色，脚本按缓存完整性自动二选一（见 §4.3） |
| 本任务**不报告** | 任何"训完模型后的效果提升"数字；只报告数据流水线本身的可运行性、漏斗行为与产物质量边界 |

**为什么这类数据值得单独做**：预训练语料清洗是"规则 + 模型"多级漏斗（C4/G/FineWeb 一脉：先用廉价规则删大头，再用贵模型打分），SFT 数据质量由"指令-回答"关系决定（IFD 比值/复杂度/质量分），SFT 合成则是"从零生成 + 打分过滤"的闭环。DataFlow 把这三件事拆成可复用算子；本任务逐条跑通，并用**带缺陷清单的设计语料**把"每一级漏斗砍掉了谁"变成可核对的证据。

## 1. 实验设计：带缺陷清单的语料

官方 example_data 只有 10 条干净文本，跑过滤漏斗全是"全过"，验证不了任何过滤器。因此用 `build_corpora.py` 本地生成两份语料（不下载外部数据），每条样本带 `id` 与缺陷标注：

| 语料 | 行数 | 构成 |
|---|---|---|
| pt_input.jsonl | 44 | 12 干净百科文档 + 4 边界对照（恰好 20 词/19 词/3 句/2 句）+ 26 缺陷文档（每条形靶向一个过滤器）+ 3 refiner 演示文档（emoji/URL/多余空格） |
| sft_input.jsonl | 16 | 8 干净指令对 + 8 缺陷/边界（过短/过长/空输出/重复输出/指令不对应/低质量/恰好 20 词/19 词） |

设计原则：每条文档标注 `defect`（预期触发的过滤器）与 `expect`（keep/drop），事后用 `verify_task5.py` 做 id 级核对——**漏斗的实际落点 vs 设计意图**的差异本身就是发现（见 §3.1、§4）。边界文档贴阈值构造（如恰好 20 词、恰好 3 句），用来验证阈值语义。

## 2. 三条官方 pipeline 与算子账本

### 2.1 预训练数据过滤（官方 GPU 版，25 步，本任务改造后全通）

```
input(raw_content)
 -> RemoveEmojiRefiner / HtmlUrlRemoverRefiner / RemoveExtraSpacesRefiner   # 3 个 refiner：只改内容不过滤
 -> BlocklistFilter(屏蔽词计数<=1)
 -> WordNumberFilter(20<=词数<100000)
 -> ColonEndFilter(不以冒号结尾)
 -> SentenceNumberFilter(3<=句数<=7500)
 -> LineEndWithEllipsisFilter(省略号结尾行比例<0.3)
 -> ContentNullFilter(非空)
 -> MeanWordLengthFilter(3<=平均词长<10)
 -> SymbolWordRatioFilter(#/.../… 与词数比<0.4)
 -> HtmlEntityFilter(无 HTML 实体)
 -> NoPuncFilter(标点间最大词数<=112)
 -> SpecialCharacterFilter(无 �/{/U}/u200e/U+1F600 等模式)
 -> WatermarkFilter(无 Copyright/Watermark/Confidential)
 -> CurlyBracketFilter(花括号/文本长<0.025)
 -> CapitalWordsFilter(全大写词比例<=0.2)
 -> LoremIpsumFilter(lorem ipsum 占比<=3e-8)
 -> UniqueWordsFilter(唯一词比例>0.1)
 -> CharNumberFilter(去空白字符数>=100)
 -> LineStartWithBulletpointFilter(项目符号行比例<=0.9)
 -> LineWithJavascriptFilter(非 javascript 行数>=3 或总行数<=3)
 -> PairQualFilter(min_score=2, BERT 质量回归)
 -> LanguageFilter(fasttext, __label__eng_Latn)
 -> MinHashDeduplicateFilter(threshold=0.9, 5-gram, 保留首条)
```

### 2.2 SFT 数据过滤（官方 GPU 版，3 步）

```
input(instruction, input, output)
 -> WordNumberFilter(output 20<=词数<1000)
 -> SuperfilteringFilter(gpt2 算 IFD = PPL(指令+回答)/PPL(回答)，官方保留 [0.5,1.0])
 -> DeitaQualityFilter(hkust-nlp/deita-quality-scorer 对 Q/A 打 1-6 分，官方保留 >=2.5)  # 本次退化为 AlpagasusFilter
```

IFD（Instruction Following Difficulty）语义：回答在"指令条件下"与"单独"的困惑度之比。比值低 = 指令强烈预测回答（数据对模型有用）；比值 >1 = 指令对回答几乎没有信息量（不匹配）。官方保留区间 [0.5,1.0] 是"中等"带宽。

### 2.3 SFT 数据合成（官方 API 版，3 步）

```
empty.jsonl
 -> CondorGenerator(num_samples//3 个 prompt；每个 prompt 随机采样知识树 topic/domain/theme + 任务场景，
    阶段一生成 [Easy]/[Medium]/[Hard] 三档问题，阶段二逐题生成答案)
 -> CondorRefiner(阶段一 critique：对 Q/A 写优缺点；阶段二 refine：按 critique 改写答案，剥离开放标记)
 -> AlpagasusFilter(LLM 对 Q/A 打 0-5 分，官方保留 [3,5])
```

## 3. 实测结果

> 全部数字来自 vm-60 真实运行（2026-10-31 21:35-22:03 CST），原始日志与逐步 jsonl 见 [evidence/task5](../evidence/task5)。

### 3.1 预训练数据过滤：44 → 14（25 步漏斗）

| 步骤 | 行数 | 本步砍掉（id 级） |
|---|---|---|
| input | 44 | — |
| RemoveEmoji/Url/Spaces Refiner | 44→44→44 | 0（只改内容：emoji_doc 去掉 🎆、html_url_doc 去掉 2 个 URL、extra_spaces_doc 压缩连续空格，均 changed=True） |
| BlocklistFilter | 44→42 | blocklist（2 个屏蔽词）、null_content（**注意**：三个空格被 refiner 压成空串，死在这一步的 `if text` 判空而不是 ContentNullFilter） |
| WordNumberFilter | 42→40 | too_short（7 词）、boundary_words_19（19 词，验证 `>=20` 语义） |
| ColonEndFilter | 40→39 | colon_end（冒号结尾） |
| SentenceNumberFilter | 39→37 | one_sentence（1 句）、boundary_sentences_2（2 句，验证 `>=3` 语义） |
| LineEndWithEllipsisFilter | 37→37 | **0**——ellipsis_lines 逃过（见 §4.1） |
| ContentNullFilter | 37→37 | 0 |
| MeanWordLengthFilter | 37→35 | mean_len_long（化学长词均长 15.5）、mean_len_short（单字母词均长 1.0） |
| SymbolWordRatioFilter | 35→34 | symbol_heavy（20 个 # / 24 个 token ≈ 0.83） |
| HtmlEntityFilter | 34→33 | html_entity（&quot;/&amp;/&nbsp;） |
| NoPuncFilter | 33→32 | no_punc（120 词无标点长句 > 112） |
| SpecialCharacterFilter | 32→31 | special_char（含 &#247; 与 {/U}） |
| WatermarkFilter | 31→30 | watermark（Copyright/Confidential） |
| CurlyBracketFilter | 30→29 | curly_bracket（JSON 花括号比 0.11 > 0.025） |
| CapitalWordsFilter | 29→28 | capital_words（22 个缩略语全大写，比例 1.0） |
| LoremIpsumFilter | 28→27 | lorem_ipsum |
| UniqueWordsFilter | 27→26 | unique_words_low（同句重复 30 遍，唯一词比 0.03） |
| CharNumberFilter | 26→26 | 0（被前面的过滤器预先淘汰） |
| LineStartWithBulletpointFilter | 26→25 | bullet_lines（5 行全 • 开头，压平换行后仍查行首，照样砍） |
| LineWithJavascriptFilter | 25→25 | **0**——javascript_lines 逃过（见 §4.1） |
| PairQualFilter | 25→18 | seo_spam(-7.07)、gibberish(-7.46)、ellipsis_lines(-4.86)、javascript_lines(-6.51)、html_url_doc(-5.11)、boundary_words_20(-3.68)、boundary_sentences_3(-2.44) |
| LanguageFilter | 18→17 | non_english（西班牙语，PairQual-en 得 2.02 过质量关后被 fasttext 判为非 eng_Latn） |
| MinHashDeduplicateFilter | 17→14 | dup_exact_a、dup_exact_b（与 pt_clean_00 完全相同；**首条 pt_clean_00 保留**）、near_dup（只换 1 个词，实测 5-gram Jaccard=0.97 > 0.9） |

**存活 14/44；缺陷清单吻合 40/44**。4 条不吻合全部有解释：boundary_words_20 / boundary_sentences_3 是"词表式"文本，PairQual 质量分只有 -3.7/-2.4（词表不像百科散文）；html_url_doc 因正文带 URL 被 PairQual 压到 1.28（<2）；dup_exact_a 预期保留但它与 pt_clean_00 同文本、输入序在后，被 MinHash 当次条砍掉。

**PairQualScore 分布**（进入该步的 18 条）：min=2.02, max=5.69, mean=3.95, median=4.06。干净百科文本实测 2.6-5.7——官方 min_score=2 的门槛贴着分布上沿，**这条阈值在本语料上等于"只留高质量散文"**（营销垃圾/乱码 -7、带 URL 文本 1.3、词表 -3.7 全被砍）。

### 3.2 SFT 数据过滤：16 → 11

| 步骤 | 行数 | 砍掉（id 级） |
|---|---|---|
| input | 16 | — |
| WordNumberFilter(output 20-1000 词) | 16→12 | sft_short_output（4 词）、sft_long_output（~1800 词）、sft_empty_output（空）、sft_boundary_19（19 词） |
| SuperfilteringFilter(IFD∈[0.5,1.0]) | 12→11 | sft_mismatch（问水沸点答罗马史，IFD=1.25 > 1.0） |
| AlpagasusFilter(LLM 0-5 分, [3,5]) | 11→11 | 0（最低分 3.0，mean 4.27） |

**SuperfilteringScore 分布**（进入该步的 11 条）：min=0.65, max=0.98, mean=0.86。官方 [0.5,1.0] 区间把"指令与回答不对应"（IFD=1.25）精准挡在外面；干净指令对 0.81-0.95。

### 3.3 SFT 数据合成：0 → 9（知识树从零生成）

| 指标 | 实测值 |
|---|---|
| prompt 数 | 3（num_samples=9 → 9//3=3 个知识树采样 prompt，seed=20261101 固定） |
| CondorGenerator | 3 prompt → 9 行（每个 prompt 解析出 Easy/Medium/Hard 三档问题各 1 条 + 逐题答案），difficulty 分布 3/3/3 |
| CondorRefiner | 9 行；critique+refine 两阶段改写后 output 长度 6 条变长（如 Hard 6445→11223 字符）、3 条变短，全部非空 |
 | AlpagasusFilter | 9 行全部得分 4.0，[3,5] 区间内 9/9 保留 |
| API 成本 | 44 次请求（39 成功、5 次 504 被 DataFlow 指数退避重试吸收）、14 分 33 秒、平均延迟 29.7s/请求（≈4.3 请求/分钟，20 RPM 用户级限流 + rpm=10 客户端限速下的真实吞吐） |
| 采样到的主题 | Environment/Pollution Control（Daily Conversation）、Military（Role Playing）、Sports/Fitness（Problem Solving）等，见 api_calls.jsonl |

合成样例（Easy，得分 4）：*"I was reading the news and saw that my city is planning to introduce a new rule about not burning leaves in the backyard..."* —— Condor 的"场景化问题"（带上下文、不突兀）确实按 prompt 要求生成了。

## 4. 质量边界与失败分析

### 4.1 refiner 压平换行，三个"行比例"过滤器名不副实

官方 PT 管线把 3 个 refiner 放在最前面，其中 `RemoveExtraSpacesRefiner` 用 `" ".join(text.split())` 标准化空白——**换行符一起被压掉**。于是后面三个按行统计的过滤器看到的都是单行文本：

| 缺陷文档 | 原始文本上的判定（unit_check） | 压平后的判定 | 管线实际落点 |
|---|---|---|---|
| ellipsis_lines（3/4 行以 ... 结尾） | LineEndWithEllipsisFilter **drop** | keep（末行以句号结尾，比例 0） | 逃过，改落 PairQual |
 | bullet_lines（5 行全 • 开头） | LineStartWithBulletpointFilter **drop** | **drop**（查行首，压不掉） | 正常砍掉 |
| javascript_lines（5 行全含 javascript） | LineWithJavascriptFilter **drop** | keep（1 行 ≤3） | 逃过，改落 PairQual |

结论：这三个过滤器在官方管线顺序下只有"查行首"的 bullet 还能发挥作用；ellipsis/javascript 变体要么把 refiner 后置，要么改用不依赖换行的判据。算子本身无误（单元核对全部符合设计），是**管线顺序**削弱了它们。

### 4.2 PairQual-en 的分数区间与官方阈值

PairQual-Scorer-en 实测输出约 [-8, 5.7]（BERT 回归头，无界），官方 `min_score=2` 靠近分布顶部：干净百科 2.6-5.7 过、SEO 垃圾/乱码 -7 砍、带 URL 1.28 砍、词表 -3.7 砍、 casual 新闻体 -1.8 砍。**把它当"质量高分过滤器"用是合理的，但当"语言/主题过滤器"用会误伤**（URL、表情、词表都被压分）。阈值应按语料分位数定，并固定评分器版本。

### 4.3 deita 模型不可得时的退化路径

官方 SFT 过滤第三步 DeitaQualityFilter 需要 13.5GB 的 Llama-7B 评分器；本次 hf-mirror 对 IP 限流（0.2MB/s，ETA > 4 小时）拿不齐，且其 xet 后端不支持 HTTP Range 续传。脚本按"缓存文件字节数 == 仓库元数据"自动二选一：齐则跑官方 DeitaQualityFilter，缺则退化 AlpagasusFilter（官方 API 合成管线的同款 LLM 质量过滤器，0-5 分，阈值 [3,5]），日志明确标注走了哪条。**退化后的实测边界**：LLM 裁判对本模型自己生成/润饰过的文本很松——sft_repetitive_output（同句重复 40 遍）和 sft_low_quality（8 句敷衍回答）都拿到 ≥3 分存活；而规则/IFD 类过滤器对这两类缺陷是能砍的。说明 LLM 质量分适合做"粗筛下限"，重复度/信息量需要专用指标（如 UniqueWordsFilter、IFD）兜底。

### 4.4 合成数据的裁判自利偏差

Condor 合成 + CondorRefine 改写后的 9 条样本，Alpagasus 全部打 4.0 分（零方差）。同一套裁判对"设计缺陷语料"的打分有 3-5 的区分度，但对"本管线自己生成的数据"一律从宽——**用同一个 LLM 既当生成器又当裁判员，分数会系统性偏高**，合成数据的质量门槛需要外部信号（执行验证、人工抽检、独立裁判）兜底。

### 4.5 其他工程约束

- **端点限流**：AMD 端点 20 requests/min 用户级限流 + 偶发 504（本次 5 次，全部被指数退避吸收）；客户端 rpm=10 匀速放行后 0 次 429，有效吞吐 ≈4.3 请求/分钟。
- **PT 管线 init 394s**：BlocklistFilter 初始化时尝试 `nltk.download('punkt')`，vm-60 无外网时每次 SSL 超时；punkt 本身不被使用（use_tokenizer=False），属无害但耗时的初始化（可预置 NLTK_DATA 消除）。
- **官方小瑕疵**：CondorGenerator 的答案 prompt 有笔误（"Please answer this questiong truthfully"）；CondorRefiner 输出依赖 `[Improved Answer Start/End]` 标记剥离；AlpagasusSampleEvaluator 的 `float(score_line.split()[0])` 遇到非数字首行会炸（本端点实测首行就是裸分数，未触发）。

## 5. 复现

```bash
# Windows 主机侧（vm-60 外网被劫持时的隧道；Node CONNECT 代理 + ssh 反向隧道）
node _tmp_connect_proxy.js 18888 &
ssh -N -R 18899:127.0.0.1:18888 vm-60 &

# vm-60 侧
source ~/dataflow-task5/task5/env.sh    # 代理、HF 离线缓存、AMD 端点配置（密钥只从 key 文件进进程）
cd ~/dataflow-task5/task5
bash run_all.sh          # 语料 -> PT 过滤 -> SFT 过滤 -> SFT 合成 -> 核对 + 行过滤器单元核对
bash run_all.sh pt|sft|syn   # 只跑某一段
```

本地归档：`code/task5/`（7 个脚本 + env.sh + run_all.sh）；证据：`evidence/task5/`（44 个文件：3 条 run 日志、verify_task5.json、unit_check_line_filters.json、语料与清单、三条管线逐步 jsonl）。

## 6. 材料索引

| 文件 | 内容 |
|---|---|
| evidence/task5/pt_filter_run.log | 预训练过滤 25 步漏斗（每步 in/out 行数） |
| evidence/task5/sft_filter_run.log | SFT 过滤 3 步日志 |
| evidence/task5/sft_synthesis_run.log | SFT 合成 3 步日志 |
| evidence/task5/verify_task5.json | id 级去向核对 + 分数分布 + API 统计 |
| evidence/task5/unit_check_line_filters.json | 三个行比例过滤器的算子级单元核对（refiner 前后对照） |
| evidence/task5/corpora/ | 设计语料与缺陷清单 |
| evidence/task5/steps_pt/ · steps_sft/ · steps_syn/ | 三条 pipeline 的逐步产物（含分数列） |
| evidence/task5/figures/runshot_task5_0{1..6}_*.png | 运行截图：PT 漏斗 / id 级核对 / SFT 漏斗 / 合成 / 行过滤器单元核对 / 打卡总览 |
| evidence/task5/dataflow-task5-checkin.md | 打卡文案 |
| code/task5/render_runshots.py | runshot 渲染器（从真实日志/JSON 生成上图） |
| code/task5/build_corpora.py | 设计语料构造器（缺陷清单 + 边界对照） |
| code/task5/pt_filter_task5.py · sft_filter_task5.py · sft_synthesis_task5.py | 三条官方 pipeline 的改造版 |
| code/task5/verify_task5.py · unit_check_line_filters.py | 核对脚本 |