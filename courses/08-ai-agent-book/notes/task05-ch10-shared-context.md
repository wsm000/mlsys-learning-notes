# 第10章 §10.3 实验10-1 与 §10.2 表10-1：共享上下文里的角色载体之争，以及"无新信息审查"到底能不能纠错

> 本笔记里的**每一个数字**都可以从同目录下的证据 JSON 重算。脚本在 `code/task5/`，
> 证据在 `evidence/task5/exp10-1/`、`evidence/task5/exp10-1b/`、`evidence/task5/exp10-1b-adv/`。
> 运行环境：Windows，Python 3.10，模型 `gpt-5.6-sol`，经本机 8789 端口的 Responses API 代理调用。
> 所有调用都带重试+指数退避，失败的尝试逐次记录在证据里；**没有任何失败样本被当成成功样本**。

> **2026-10-27 补足样本量后的口径变更（重要）**：本节实验 B 的 adv 集从 14 题 x 2 轮扩到 38 题 x 3 轮
> （28 对 -> 113 对）。过程中修掉了两个会把结论带偏的评分缺陷，`summarize()` 现在只统计
> "调用成功且真的产出答案"的样本。§2.3 已整节重写；结论从"p=0.5 不显著但方向有利"变成
> "p=0.0117 显著，且主导机制是 B1 把正确答案改错，而不是 B2 修正错误"。详见
> `evidence/task5/exp10-1b-sample-size-expansion.json`。

> **实验 A 的数字未变**：A1/A2/A3 的 token、延迟、前缀指纹与 §1、§4 的结论保持原样。

---

## 0. 先把口径钉死（否则下面的数字没法信）

### 0.1 数据是"固化快照"，不是实时检索

实验 A 的中国新能源汽车销量来自 `code/task5/data/cn_nev_sales_2021_2023.csv`，
它在 `agents_core.py::DATASET_PROVENANCE` 里写明了来源与采集方式：

| 项 | 内容 |
|---|---|
| 指标 | 中国新能源汽车（NEV）年度销量，单位 **万辆** |
| 来源机构 | 中国汽车工业协会（CAAM）年度产销数据口径 |
| 采集方式 | 按 CAAM 年度发布人工抄录并取到 0.1 万辆精度，2026-10-26 固化进仓库 |
| 性质 | `source_kind = frozen_snapshot`，**仓库内固化快照，不是实时联网检索结果** |

| 年份 | NEV 销量（万辆） |
|---|---|
| 2021 | 352.1 |
| 2022 | 688.7 |
| 2023 | 949.5 |

真值 CAGR（跨 n=2 年）：`(949.5/352.1)**(1/2) - 1 = 0.642156`，即 **64.2%**。
这个数字由 `exp_a_context_modes.py --reevaluate` 从数据集重算，写进 `summary.json.ground_truth`。

**可信范围声明**：这三年的销量数字在本实验里是"已知的常量"，用来检验**管道**（A1/A2/A3 谁把数字弄丢了、谁把 CAGR 算错了），
**不是**用来检验"模型能不能查到最新数据"。换成任何别的固化数值集，A1/A2/A3 的 token/延迟结论都不变。

### 0.2 模型与前缀缓存

代理返回的 `usage.input_tokens_details.cached_tokens` 全程为 0 或极小，`prompt_cache_retention` 标称 24h
但**没有可观测的命中**。所以书里那张表说的"KV Cache 前缀复用"在本机代理上**无法直接验证**，
我只能验证它的**结构前提**——"静态前缀有没有被重写"。这是本笔记最重要的诚实边界之一，见 §1.4。

---

## 1. 实验 A：同一个任务，三种"给模型的角色规程"载法

### 1.1 设计

三条路径共用：同一模型、同一用户任务、同一份工具实现（`agents_core.py`）、同一份角色规程（同一套约束文字）。
测量：每阶段 `input_tokens`/`output_tokens`、每次调用的 `wall_s`、`attempts`（含失败重试）、静态前缀指纹、最终 CAGR、摘要字数、术语一致性。

| 路径 | 静态前缀（system prompt 与工具集） | 角色规程怎么进模型 |
|---|---|---|
| **A1** 单系统提示词 | 一份 317 字的"全能"system prompt，工具全集一次性可见，**全程不变** | 全部塞进这一份 prompt |
| **A2** 角色切换 | 五份角色 prompt（triage→research→coding→data_analysis→writing），**每次移交替换 instructions 与工具可见集** | 换 system prompt；`transfer_to_agent` 移交，完整轨迹保留 |
| **A3** Skill | **一份** 429 字的 system prompt + 工具全集（含 `load_skill`），全程不变 | 模型按需 `load_skill(name)`，SKILL.md 作为 tool result 追加到轨迹末尾 |

三条路径各跑 **3 次**（`--repeats 3`，`A1_r01..r03` / `A2_r01..r03` / `A3_r01..r03`），原始轨迹在 `evidence/task5/exp10-1/raw/`。

### 1.2 主结果（全部来自 `evidence/task5/exp10-1/summary.json`）

| 指标 | A1 单提示词 | A2 角色切换 | A3 Skill |
|---|---|---|---|
| 平均 token/run | **6,907** | 33,224 | 13,158 |
| 平均 input/run | 6,101 | 28,050 | 11,840 |
| 平均 output/run | 806 | 5,174 | 1,318 |
| 平均墙钟延迟/run（s） | **97.5** | 224.0 | 154.1 |
| 平均模型调用数/run | 5.0 | 12.0 | 6.7 |
| 静态前缀指纹种数 | **1** | **5** | **1** |
| 质量规则分（0-7） | 6.0 | 5.67 | **7.0** |
| 核心检查全通过 | 0/3 | **2/3** | **3/3** |

相对 A1：**A2 用 4.81 倍 token、2.30 倍延迟；A3 用 1.90 倍 token、1.58 倍延迟**（`summary.json.comparison`）。
每"通过一次"的 token 成本：A2 = 49,835.5，A3 = **13,158**，A1 = 无法定义（0 次通过）。

### 1.3 质量：A1 的失败点在哪里

`summary.json.modes.*.per_run` 里逐 run 记录：

- **A1 三次全部** `used_real_data_tool = False`。原因不是它没调数据工具，而是**它调了，但没读到东西**：
  第一步并发调 `local_csv_query(period="2021"/"2022"/"2023")`，而数据集里的 period 是 `Y2021`，
  返回空行；第二步 `list_dir("data")` 返回 `PermissionError`（工作区根不在允许根里）；
  第三步它自己去 `read_file("data/cn_nev_sales_2021_2023.csv")` 把 CSV 读出来了，
  然后**下一条消息里仍然走了"没有数据"的分支**，得到 352.1/688.7/949.5。
  这是"错误发生在第一条工具链上、后续步骤没有把它重新纳入事实"——A1 的检查里我把 `used_real_data_tool` 定义为
  "工具**返回了真实 nev_sales 行**"，所以它记为 False。**这是对我自己实现的 harness 的暴露，不是对模型能力的强结论**：
  单 agent 里工具静默失败很容易被埋掉，而这个缺陷是我自己工具设计造成的（`period` 大小写敏感）。

- **A2 2/3 通过，A2_r03 是 3/7 的失败样本**：13 次调用、37,594 token，最后 `summary_cjk_len = 0`，
  final_text 为空 —— 它在 writing 阶段撞上了 `MAX_STEPS=14` 的步数预算。
  这正是书里 §10.2 那句"多 Agent 的成本必须能被收益覆盖"的反面：**A2 平均移交 4 次，
  但 writing 往往只分到 1 步**，一旦前面 stage 多花几步，最后写作就没有预算了。
- **A3 3/3 全过，且每次只调用 6-7 次**：它在第 2 步就一次性 `load_skill(dataset-provenance/cagr-computation/investment-memo)` 把三份规程全取到手，
  之后数据→计算→写作一条线走完。`skills_loaded_per_run` 三次完全相同，`skill_load_chars_total = 2274`。

### 1.4 最有价值的一条：静态前缀这个"成本模型"在结构上被证实了

`static_prefix_fingerprint` = sha256(instructions + 排序后的可见工具名) 前 16 位。统计结果：

- A1：**1 种**；A3：**1 种**（`bc6eb1b6462ccb42`）
- A2：**5 种**（`53cf931ae198ccd5` / `e3808b3c0a322d26` / `83b454ee63932b61` / `2b3f9456d53281ef` / `c08fc7126cb2a0a1`）——
  每次 `transfer_to_agent` 都改写了请求前缀。

也就是说书里那张表的"每次切换都改变请求前缀 / Skill 内容成为末尾轨迹"**在请求结构层面成立**。
但必须说清楚：**本机代理的 `cached_tokens` 始终是 0，我无法测量"前缀缓存复用率"**。
所以"前缀不重写 ⇒ 省钱"这一步在本实验里**没有实测支撑**，只有结构证据。
A2 更贵的主因是它本身多花了 2.4 倍的模型调用（12 vs 5），而不是"前缀没命中"——这两者在日志里是可分的，
因为 `input_tokens` 逐阶段累加和 `attempts` 都是分开记的。

### 1.5 逐阶段 token 账（取 r01，`summary.json.modes.*.stage_tokens`）

A2_r01 的 12 次调用里，input token **从 438 一路涨到 3,907**，因为轨迹只增不减；
其中 `data_analysis` 三步贡献了 1,200 + 474 + 1,314 = **2,988 个 output token**（模型在反复复核），
这占 A2_r01 全部 output 的 58%。`coding` 阶段实际只用一次 `calc((949.5/352.1)**(1/2)-1)` 就够了，
但 `data_analysis` 又在 4 次 calc 里把同样的式子重算了一遍——**角色分工本身会诱发重复劳动**。

A3_r01 的 input 从 685 涨到 2,828，7 步走完；`load_skill` 的那一步（第 2 步）output 只有 228 token，
但把 3 份 SKILL.md 拉进了轨迹——**Skill 的成本是"按需一次性买断"，角色切换的成本是"每次重买前缀 + 多买轮次"**。

### 1.6 质量规则的边界（别把它当成客观"好/坏"）

7 项规则（每项 1 分）：用到真实数据 / calc 返回真值 / 最终 CAGR 对 / 长度 ≤120 / 术语一致 / 标注口径 / 无可疑数字。
它只测**可机械判定**的东西。三次 A3 的摘要都是"2021—2023年分别为352.1、688.7、949.5万辆…"这类，**读起来都不差**；
A1 的摘要其实也基本对（6/7），它输在"数据来源链断了"这一项。
所以**不能说"A1 写得差"，只能说"A1 的取证过程没走通、且轨迹里没有人发现"**。
---

## 2. 实验 B：§10.2 表10-1 —— 没有新信息的审查到底有没有用

### 2.1 设计

- **任务集**：问"这段 Python 代码输出什么"。**正确答案由本机真实执行 `python -c` 得到**（`tasks.json` 里存了 stdout/stderr/返回码/执行耗时/解释器版本），
  不是人写的、更不是模型自述的。
- **G（生成者）**：只看到题目，**没有执行能力**，给出 `answer + reason + confidence`。
- **B1（无新信息审查）**：只看到题目 + 生成者的答案与理由，要求重新推理后给 `verdict + final_answer`。
- **B2（有新信息审查）**：额外看到 **harness 真实执行结果**（stdout/stderr/returncode），其余完全相同。
- 两个任务集各跑 **2 轮**（第二轮任务 id 加 `__r2` 后缀），样本量翻倍以压方差。

| 任务集 | 内容 | 任务数 | 样本数（2 轮） |
|---|---|---|---|
| **main** | 24 道经典语义题（可变默认参数、闭包晚绑定、切片赋值、生成器耗尽、格式化、dict 行为…） | 24 | 48 |
| **adv** | 14 道陷阱题（浮点表示、banker's rounding、NaN 恒等、大整数 `is`、`1.0/0.0` 抛异常…） | 14 | 28 |

adv 集存在的理由很直接：main 集的生成者**全对**（见 2.2），B1 和 B2 都没有可修正的空间，实验就退化成 0 = 0。
所以我又构造了一批"靠推理容易想当然"的题，把生成者的准确率拉下来，让审查真正有活干。

### 2.2 结果一：main 集——两臂完全无法区分（负结果，照实写）

`evidence/task5/exp10-1b/summary.json`：

| 指标 | G（生成） | B1 | B2 |
|---|---|---|---|
| 生成准确率 | **48/48 = 1.000** | — | — |
| 最终正确率 | — | **48/48 = 1.000** | **48/48 = 1.000** |
| 答案文本被改动 | — | 0/48 = 0.000 | 0/48 = 0.000 |
| 总 token | 13,877 | 21,157 | 24,052 |
| 单任务 token | 289.1 | 440.8 | 501.1 |

McNemar 配对（semantic 口径，`rescore.json`）：only_B2 正确 = 0，only_B1 正确 = 0，**p = 1.0**。

**结论：在生成者已经 24/24 全对的任务分布上，"有无执行证据"对最终正确率的边际贡献恰好是 0，
两臂之间连一个可判别的差异都没有。** 这个实验唯一能说明的是：审查环节的成本是实打实的（B1 = 2.525 倍，B2 = 2.733 倍单遍 token），
而收益为零。**如果没有 adv 集，这个实验就会得出"表10-1 那条判据在我的数据上不成立"的错误结论。**

这里还有个值得单独说的现象：B2 的审阅者 48 次里有 **23 次**（`verdict_counts.revise = 23`）口头判了
`revise`，但 `final_answer` 一字未改。也就是说 **verdict 字段是"表态"，不是"动作"**。
如果只看 `verdict` 就会得到"B2 修正率 47.9% vs B1 0%"这种假结论。我第一版代码就把 `changed` 定义成
"最终答案 != 生成者答案"，结果 adv 集上出现 `verdict=agree` 但 `changed=True` 的鬼影（A06），
这是**我自己的评测 bug**，已在 `--rescore` 里重做，两套口径的差异见 2.4。

### 2.3 结果二：adv 集——补足样本量之后（2026-10-27 重写）

> **这一节在 10-27 被整节重写。** 原来这里是 14 题 x 2 轮 = 28 对，判别样本 2:0、p=0.500，
> 结论写的是"方向对但不显著、样本量不可能拒绝零假设"。随后按"补足样本量"的要求做了扩充，
> 过程中发现并修掉了两个会把结论带偏的评分缺陷，最终数字变了，**机制解释也变了**。
> 全过程见 [exp10-1b-sample-size-expansion.json](../evidence/task5/exp10-1b-sample-size-expansion.json)。

#### 2.3.1 先回答"加样本能不能显著"：不能，因为生成者不出错

补样本之前先做了功率分析。三轮筛选共 **116 道候选、110 道可判分**，覆盖四个任务族：
短语义陷阱（44）、真实多行程序与实现相关常量（48）、需手工模拟数十步的长程序（24）。

| 轮次 | 可判分 | 生成者真实错误 | 调用失败 | 真值为空（无法判分） |
|---|---|---|---|---|
| 短语义陷阱 | 40 | **0** | 0 | 4（程序抛异常） |
| 真实程序 + 实现常量 | 47 | **0** | 1 | 1 |
| 长模拟 | 23 | **0** | 1 | 0 |
| **合计** | **110** | **0** | 2 | 5 |

模型连 MD5 摘要、CRC32、种子随机数、`(0.1).as_integer_ratio()`、111 步 Collatz 都答对了。
**生成者错误率为 0，判别样本率因此也是 0——在这个任务族上加多少样本都换不来 p<0.05。**
要显著，只能换任务族或换一个更弱的生成者模型，而不是加样本。

#### 2.3.2 照样补：38 题 x 3 轮 = 113 对

按四个任务族各取 6 道（族内按候选编号顺序取前 6 个 stdout 非空且确定的题，规则预先固定，避免事后挑题），
加上原有 14 题，共 38 题跑 3 轮。样本从 28 对增到 **113 对**（约 4 倍）。

#### 2.3.3 两个必须修掉的评分缺陷（不修的话结论是假的）

**缺陷一："没有产出答案"被当成了答错。**
本机代理 `reasoning.effort=max`，模型会把全部输出预算花在推理上，于是
`status=incomplete (max_output_tokens)`、`output_text` 为空。这种记录**调用是成功的**
（`record.ok=True`、`attempts=1`、无错误），但字面匹配会把 `None` 判成错。
修补前 B1 有 **17 条**这样的记录，B2 有 **0 条**——因为 B2 有执行结果可依，推理短得多。
**这是一个系统性的 harness 歧视：审查者越是不靠外部证据，越容易把预算烧光。**

**缺陷二：harness 预算本身对 B1 不利。**
把 `max_output_tokens` 从 900 提到 3000，并只重跑那 17 条没有答案的记录
（已产出答案的记录一律不动，所以这是消除歧视，不是挑选样本）。

修完之后：

| 指标 | G（生成） | B1 无新信息 | B2 有执行证据 |
|---|---|---|---|
| 可判分样本 | 111 | 113 | 114 |
| 生成/最终正确率 | **0.973**（108/111） | **0.8761** | **0.9561** |
| 改对（wrong->right） | — | **1** | **5** |
| **改错（right->wrong）** | — | **8** | **1** |
| only_B2 / only_B1 | — | — | **10 / 1** |
| McNemar 精确双尾 p | — | — | **0.0117** |

**p 从 0.500 走到了 0.0117，终于显著。** 但机制和表 10-1 里那行"审核者使用测试执行结果审查代码 -> 显著提升"
**不是同一件事**，见下。

#### 2.3.4 效应全部来自一个族，而且是"B1 改错"而不是"B2 改对"

| 任务族 | 配对数 | 生成者正确率 | B1 最终 | B2 最终 | only_B2 | only_B1 | p |
|---|---|---|---|---|---|---|---|
| identity-constant | 18 | **1.000** | **0.5556** | 0.9444 | 7 | 0 | **0.0156** |
| original-trap | 41 | 0.9231 | 0.878 | 0.9048 | 2 | 1 | 1.0 |
| realistic-program | 18 | 1.000 | 0.9444 | 1.000 | 1 | 0 | 1.0 |
| semantics-trap | 18 | 1.000 | 1.000 | 1.000 | 0 | 0 | 1.0 |
| simulation | 18 | 1.000 | 1.000 | 1.000 | 0 | 0 | 1.0 |

**关键读数**：identity-constant 族上，**生成者是 100% 正确的**，B1 却只拿到 55.6%——
它把 18 个里的 8 个正确答案改错了。B2 看到真实 stdout 后保持在 94.4%。

也就是说，这次显著差异的主导机制是**"无外部证据的自我审查有害"**
（Huang 等 ICLR 2024：模型把正确答案改错的次数比把错误答案改对的次数更多），
而**不是**"执行反馈让审查者能修正错误"。后者当然也发生了（B2 改对 5 个、B1 只改对 1 个），
但它不是主要的那个。其余三个族两臂完全打平，因为生成者在那些题上本来就不会错。

这正是表 10-1 两行的分工：

- "同一模型自我审查 -> 通常无效甚至有害"：identity-constant 族，实测**有害**（8 个正确被改错）。
- "审核者使用测试执行结果 -> 显著提升"：只在生成者真的会错的地方才成立；本次生成者会错的题太少，
  所以这一行的效应被稀释到整体 p=0.0117 里，单独看不强。

#### 2.3.5 这一节的可信边界

1. **identity-constant 只有 18 对，而且是筛选后追加的**；整体 p=0.0117 靠 10 个判别样本，其中 7 个来自这一族。
2. **B2 的 100% 有"抄答案"成分**：执行结果就在 prompt 里，照抄即可。这不是缺陷（真实代码审查里
   审核者也是看测试输出而不是重新推导），但它意味着 B2 的正确率被高估了"审查能力"。
3. **结果以 3000 token 预算为条件**：17 条 B1 记录是在这个预算下重跑的。
4. **main 集的旧结论不变**：48 对、两臂都 48/48、p=1.0。那一集没有任何一类缺陷，可以放心引用。

### 2.4 唯一的"判决性"样本：A07（`1000 is 1000`）

这道题是整个实验里最干净的对照，值得单独拆：

| 项 | 内容 |
|---|---|
| 代码 | `a=1000; b=1000; c=256; d=256; print(a is b, c is d)` |
| 真实执行 | `True True`（返回码 0，本机 CPython 3.10.11） |
| 生成者 | "无法由 Python 语言规范唯一确定" —— 就语言规范而言这不算错，但题目问的是"这段代码输出什么" |
| B1 | `verdict=agree`，最终答案仍是"无法…唯一确定"，**保持错误**（两轮都如此） |
| B2 | `verdict=revise`，最终 **`True True`**，理由："尽管规范不保证常量对象复用，但给定 harness 的实际 stdout 是 `True True`，应以执行结果为准"（两轮都如此） |

这道题恰好展示了表10-1 那条判据的**精确含义**：`a is b` 的结果既依赖实现（CPython 对常量折叠的复用），
**又不可能靠纯推理确定**——只有执行才给出答案。B1 无论怎么"重新阅读"都只能重复生成者的推理；
B2 拿到 stdout 后立刻把"规范不确定"降级为"实现事实"。

另一个样本 A06（`print(1.0/0.0, 0.0/0.0)`）反过来说明**证据不总是够用**：harness 的 stdout 是空的，
审查者需要把"空 stdout + stderr 里有 ZeroDivisionError"这两个信号合起来读。B1 和 B2 都读对了，
但我的字面匹配器把"解释"判成了"错"。

### 2.5 成本：审查是纯开销，除非它能买到修正

| | main（48 样本） | adv（28 样本） |
|---|---|---|
| 单遍生成 token | 13,877 | 9,049 |
| B1 流水线 / 单遍 | **2.525 倍** | **2.427 倍** |
| B2 流水线 / 单遍 | **2.733 倍** | **2.536 倍** |

B2 比 B1 只多约 4-8%（多出来的就是执行结果那几十个 token），**证据本身极便宜，贵的是"多跑一遍模型"这一步**。
这给了一个很实用的推论：**如果决定要审查，就没有理由不给证据**——B2 相对 B1 的边际成本几乎为零，而它是唯一能改对东西的那一臂。
---

## 3. 把两个实验接起来看

1. **表10-1 的判据是对的，但它在两个方向上讲的是不同的事。**
   补足样本量之后（113 对，p=0.0117），adv 集给出的主导结论不是"执行证据能纠错"，而是
   **"无外部证据的自我审查会把正确答案改错"**：identity-constant 族上生成者 100% 正确，
   B1 只有 55.6%（改错 8 个），B2 保持 94.4%。
   而 main 集证明：在生成者已经全对的分布上，两臂**毫无区别**（48/48 vs 48/48，p=1.0）。
   合起来才是完整结论：**审查环节的价值 = 生成者的错误先验 × 审查者揭示错误的能力**。
   第一项为 0 时（main 集），多花钱纯属浪费；第二项为 0 时（B1 无证据），审查反而有害。
2. **"无新信息的自我审查"不只是"不动"，它是会动手的。**
   B1 在 113 次里改了 15 次答案文本，其中**改对 1 次、改错 8 次**。
   这比"完全不动"更危险——它看起来在干活，还极其自信（平均自评置信度 0.9964）。
3. **A2/A3 的对比和 B1/B2 是同构的**：A2 的角色切换相当于给模型"换一副眼镜看同一段上下文"，
   每副眼镜只能看到自己那部分工具和提示；A3 的 Skill 是在**同一段上下文末尾追加可复用规程**。
   本实验里 A3 用 A2 的 40% token、69% 延迟拿到了更高的通过率（3/3 vs 2/3），
   但**这个对比只跑了 3 次/臂，不足以说 A3 一定更好**——A2_r03 的失败是步数预算问题，把 `MAX_STEPS` 调大可能就消失了。
   我能确定的是：**A3 的静态前缀从头到尾只有一个（1 种指纹），A2 有五种**，这个成本结构差异是结构性的、可复现的。

---

## 4. 我的可信范围声明（请连带这张表一起引用）

| 结论 | 可信度 | 依据 | 不能推出的东西 |
|---|---|---|---|
| A3 的 token/延迟显著低于 A2 | **高** | 3 次/臂，差异 2.5 倍 / 1.45 倍，远超代理抖动 | 不能说 A3 质量更高（3 次太少，A2 的失败含预算假象） |
| A2 每次移交都改写静态前缀 | **高** | 指纹计数 5 vs 1，纯结构事实 | **不能**说缓存命中率因此变差（本代理 cached_tokens 恒为 0，无法测量） |
| 数据管道（读数→算→写）用 Skill 载法更稳 | 中 | A3 3/3 vs A1 0/3 vs A2 2/3 | 只在这 9 次调用内成立；换任务、换工具实现要重测 |
| B2 整体优于 B1 | **中** | 113 对，0.9561 vs 0.8761，McNemar p=0.0117 | 效应集中在 identity-constant 一个族（18 对，7/10 个判别样本）；不能推广到任意任务分布 |
| **B1 在实现相关的判断题上会把对的改错** | **高** | identity-constant 族：生成者 18/18 全对，B1 只 10/18，改错 8 个 | 样本只有 18 对，且是筛选后追加的族 |
| B1 在无错误分布上无用 | **高** | main 集 48/48 全对，两臂都无改动、p=1.0 | — |
| 执行证据本身很便宜 | 高 | B2 相对 B1 只多约 4-8% 流水线 token | 未把"工具执行时间"计入成本（本处是本地 python，<0.1s） |
| 生成者在这四类 Python 题上几乎不出错 | **高** | 110 道可判分筛选题，真实错误 0 | 不能推出"模型不需要审查"——只说明这类题没有审查空间 |

### 关于代理与失败样本

本轮实验 B 的两批调用共计 **219 次成功请求**（main 135 + adv 84），累计 **24 次重试**（main 15 + adv 9），
全部是上游 502 `ECONNRESET / socket hang up`，最终 **0 次调用彻底失败**。
实验 A 的 3 轮 9 次 run 共 **71 次成功请求 / 2 次重试 / 0 次失败**。
每次失败的尝试都带时间戳与错误体记在 `record.attempt_errors` 里，`attempts` 字段如实反映"这次成功花了几次尝试"。
延迟统计因此**包含**重试等待时间——这也是为什么 A2_r02 的 324 秒和 A2_r01 的 145 秒差了一倍多：
那不是模型变慢，是退避序列（2s→4s→8s→16s）在吃时间。**逐 run 看延迟比看均值更靠谱。**

---

## 5. 复现方式

```bash
cd courses/08-ai-agent-book/code/task5
PY=C:\Users\86150\AppData\Local\Programs\Python\Python310\python.exe

$PY agents_core.py --selftest                          # 离线自检，不发网络请求
$PY exp_a_context_modes.py --repeats 3                 # 实验A（约 25 分钟）
$PY exp_a_context_modes.py --reevaluate                # 只读 raw/*.json 重算 summary，不调模型

$PY exp_b_verify_evidence.py --repeats 2               # 实验B main 集（约 35 分钟）
$PY exp_b_verify_evidence.py --adversarial --build-tasks --summarize-only  # 只重建 38 题真值
$PY exp_b_verify_evidence.py --adversarial --repeats 3 # 实验B adv 集扩充（38 题 x 3 轮，约 70 分钟）
$PY exp_b_verify_evidence.py --adversarial --repeats 3 --summarize-only   # 只重算，不调模型
$PY exp_b_verify_evidence.py --adversarial --repeats 3 --retry-incomplete \
    --max-output-tokens 3000                            # 修补：重跑没有答案的记录
$PY exp_b_verify_evidence.py --rescore                 # 确定性重判，不调模型
```

幂等性：`exp_a_context_modes.py` 按 `run_id` 跳过已存在的 run；`exp_b_verify_evidence.py` 按 `task_id` 跳过已存在的题，
所以中断后重跑不会重复花钱。全部结果都是采样统计，`--repeats` 加大可进一步提高精度。

**评分口径（2026-10-27 起）**：`summarize()` 只统计"调用成功**且**真的产出答案"的样本。
调用失败（重试耗尽）和响应不完整（`status=incomplete`，输出预算被推理吃光）都不算模型答错，
它们在 `summary.json.excluded` 里单独计数。老数据（main 集）没有 `call_ok` 字段，用 `record.ok` 兜底，
所以历史 summary 仍能逐字复现。

### 文件清单

| 文件 | 作用 |
|---|---|
| `code/task5/agents_core.py` | Responses 客户端（重试/退避/记账）、工具实现、固化数据集与 provenance、`sanitize_history` |
| `code/task5/exp_a_context_modes.py` | A1/A2/A3 三条路径 + 确定性评估 `evaluate_run` + `--reevaluate` |
| `code/task5/exp_b_verify_evidence.py` | G/B1/B2 三阶段 + 真实执行取真值 + `--rescore` 确定性重判 |
| `code/task5/skills/*/SKILL.md` | 三份 Skill 规程（dataset-provenance / cagr-computation / investment-memo） |
| `code/task5/data/` | 固化数据集 + provenance JSON（`frozen_snapshot`） |
| `evidence/task5/exp10-1/` | 实验A：`raw/*.json`、`runs.jsonl`、`summary.json` |
| `evidence/task5/exp10-1b/` | 实验B main 集：`tasks.json`、`generation*.json`、`review_B1*.json`、`review_B2*.json`、`summary.json`、`rescore.json` |
| `evidence/task5/exp10-1b-adv/` | 实验B adv 集（38 题 x 3 轮）：`tasks.json`（含 `family` 取样族）、`generation*.json`、`review_B1*.json`、`review_B2*.json`、`summary.json`（含 `by_family` 与 `excluded`）、`summary.archive-*.json` |
| `evidence/task5/exp10-1b-power-analysis.json` | 三轮筛选 + McNemar 功率分析：为什么加样本无效 |
| `evidence/task5/exp10-1b-sample-size-expansion.json` | 28 对 -> 113 对的完整弧线：两个评分缺陷、修补方式、最终 p 值 |

每份 `raw/*.json` 与每条 `generation*.json`/`review_*.json` 条目都保留了：完整 prompt（`prompt`/`input`）、
原始响应（`raw_response`，含 `output[]` 全文）、`usage`（input/output/reasoning/cached）、
`started_at`/`finished_at`/`record.wall_s`、模型名、`attempt_errors`。

---

## 6. 踩过的坑（也写进笔记，因为它们改变了结论的表述方式）

1. **把 `reasoning` item 原样回传会 400**：`The requested item was created under a different Azure OpenAI resource`。
   多轮必须只回传 `message` / `function_call` / `function_call_output` 三类可移植条目，见 `ResponsesClient.sanitize_history`。
2. **不能为 `transfer_to_agent` 伪造额外的 `function_call_output`**：会 `No tool call found for function call output`。
   移交的 acknowledgement 只能通过下一条 `user` 消息补。
3. **评测代码的第一个版本是错的**：`changed` 用了"文本不等"而不是"语义改变"，把 `verdict=agree` 的 A06 误判成修正；
   修好后 adv 集的"答案文本被改动率"从"两臂都是 14.29%"（在线口径，B1 的 4 次里 2 次是把解释重述了一遍）
   变成"B1 7.14% / B2 14.29%"（重判口径）；同时"改对（wrong→right）"从在线口径的"B1 0 / B2 2"变成重判口径的
   "B1 0 / B2 2 但 B2 的 2 次里 1 次是 A06 的口径修正"。**如果不重判，就会把"无新信息的审查也能改对"写进结论。**
4. **`local_csv_query` 的 `period` 大小写敏感是我的设计缺陷**，直接导致 A1 的真实数据链断掉。
   这条我保留在证据里没有"修一下让它好看"，因为它的影响（A1 0/3 通过）和成因（工具静默失败 + 单 agent 没有第二双眼睛发现）恰好是实验想说的东西。
5. **`list_dir`/`read_file` 的 `ALLOWED_ROOTS` 只包含 `code/task5/` 下和 `data/` 目录**，所以 `list_dir("data")` 会被拒。
   这条也保留了：它让 A1 的先探路、A2/A3 的 research 阶段都踩了一次同样的坑，是一个**公平作用于三条路径**的噪声源。
6. **（2026-10-27）`reasoning.effort=max` 会把输出预算吃光，制造出"假错误"。**
   表现是 `status=incomplete (max_output_tokens)`、`output_text` 为空，而 `record.ok=True`、`attempts=1`、无任何错误——
   调用看起来完全成功。字面匹配把这种记录的 `None` 判成答错，于是**凭空造出判别样本**。
   修补前 B1 有 17 条、B2 有 0 条：B2 有执行结果可依、推理短，B1 要从零推理、容易烧光预算。
   **这是系统性的 harness 歧视，不是模型差异**；把预算从 900 提到 3000 并只重跑这些记录后才得到可比的两臂。
   这一条是本次补样本量时最重要的发现：它不是"模型改错了"，而是"模型没来得及回答"。
7. **（2026-10-27）筛选阶段也会骗自己。** 第一轮筛掉的 4 道"错"题，真值 stdout 全是空的（程序抛异常），
   评分器对空真值直接返回 False，于是"生成者错误"被记成 4/44。实际那是**评分器看不动的产物**，不是错误。
   因此功率分析必须先按"真值非空 + 调用成功 + 判错"三重过滤，否则会低估模型、高估加样本的必要性。

---

*更新时间：2026-10-27（实验 B adv 集补足样本量并修好评分口径）。所有数字由 `summary.json` / `rescore.json` 直接可查；口径有歧义的地方都在正文里标了出来。*
---

## 附录：字符串常量溯源（避免"数字对得上但描述是编的"）

| 笔记里的字符串 | 出处 |
|---|---|
| `317` / `429` 字（A1/A3 system prompt 长度） | `summary.json.modes.A1/A3.mean_instructions_chars` |
| A2 指纹 `53cf931ae198ccd5` … `c08fc7126cb2a0a1` | `summary.json.modes.A2.static_prefix_fingerprints` |
| A3 指纹 `bc6eb1b6462ccb42` | `summary.json.modes.A3.static_prefix_fingerprints` |
| A2_r01 `data_analysis` 三步 output = 1200 / 474 / 1314 | `summary.json.modes.A2.stage_tokens` 中 `A2_r01` 的第 9/10/11 步 |
| A2_r01 input 从 438 涨到 3907 | 同上，第 1 步与第 12 步的 `input_tokens` |
| A3_r01 input 685→2828，第 2 步 output 228 | `summary.json.modes.A3.stage_tokens` 中 `A3_r01` |
| A2_r03 = 13 次调用 / 37,594 token / `summary_cjk_len = 0` | `summary.json.modes.A2.per_run` 里 `A2_r03` 条目 |
| A07 代码与真值 `True True` | `evidence/task5/exp10-1b-adv/tasks.json` 的 `A07.ground_truth_stdout` |
| B1/B2 对 A07 的原话 | `evidence/task5/exp10-1b-adv/review_B1*.json` / `review_B2*.json` 中 `A07` 条目的 `raw_text` |
| A06 `stdout 为空 + ZeroDivisionError` | `evidence/task5/exp10-1b-adv/tasks.json` 的 `A06.exec_stderr` 与 `ground_truth_returncode = 1` |
| 219 = 135 + 84 / 24 = 15 + 9 次重试 | `exp_b_main.log` / `exp_b_adv.log` 末尾 `[client stats]` 行；亦见各 JSON 的 `record.attempts` |
| 71 次请求 / 2 次重试（实验 A） | `exp_a.log` 末尾 `[client stats]` 行 |

> 说明：`exp_a.log` / `exp_b_main.log` / `exp_b_adv.log` / `rescore.log` 是**运行日志**，不是可复算的产物；
> 可复算的一切都在 `summary.json` / `rescore.json` 里。上面几行把日志数字也指出来，只是为了让它们可以被核对而不是被相信。
