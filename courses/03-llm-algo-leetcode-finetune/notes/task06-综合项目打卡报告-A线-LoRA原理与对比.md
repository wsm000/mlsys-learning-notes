# Task6 综合项目打卡报告(A线:LoRA原理重温 + LoRA vs QLoRA对比)

- 选题:SwanLab《SFT/6.llamafactory-finetune》二选一中选 **A线(lora1:原理+对比)**;B线(lora2:LLaMAFactory QLoRA)以可迁移YAML模板收口,不单独打卡。
- 教程:lora1 https://docs.swanlab.cn/course/llm_train_course/03-sft/6.llamafactory-finetune/lora1.html ; lora2 https://docs.swanlab.cn/course/llm_train_course/03-sft/6.llamafactory-finetune/lora2.html ; DeepSeek调参 https://docs.swanlab.cn/course/llm_train_course/03-sft/7.deepseek-lora/ ; 打卡表单DDL 09-06 03:00。
- 仓库:https://github.com/wsm000/mlsys-learning-notes ; 本报告对应代码 `task6_lora_qlora_lab.py` + `task6_vm60_lora_sweep.py` + `vm60_task6_light_check.sh`,打卡证据 `打卡材料/task6_qlora/`。
- 环境:本地 LAPTOP-K5HQA0TQ/Win10/py3.10.2/无torch(纯stdlib链路验证);vm-60 Ubuntu22.04/RTX 4090D 24GB/torch2.9.1+cu128(真训机,磁盘仅余约8GB故B线只交模板)。
- 口径声明:凡写**估算/模拟(mock)**都不是真训结论;凡写**vm-60实测**以 `simulated=False` 的报告为准;7B显存为公式估算,GPT2三档为真训实测。

## 一、LoRA原理(一句话+四处细节,本地实算)

一句话:冻结W,只训旁路 dW=BA,前向 Wx+s·BAx,训完可合并为 (W+sBA)x。

1. **B零初始化是刻意设计**:B=0则dW=0,起点等于基座,不破坏模型;代码里 `zeros_(B)` + A用kaiming初始化。
2. **起作用的是 alpha/rank**:scaling s=alpha/rank,rank8/alpha16与rank16/alpha32强度都是2.0。调参先固定比值2再动rank;mock扫描里偏离2就加 `0.01*|比值-2|` 惩罚讲清此事。
3. **dropout加在旁路输入**:`(dropout(x)@A.T@B.T)*s`,小数据先用0.05;0.2在12条样本上是自残(mock里+0.015 val)。
4. **省的是可训练参数+梯度+优化器状态**:单层账本全量 `in*out`,LoRA为 `r*(in+out)`。本地实算 hidden=4096/rank=8时单层 65536 vs 16777216 = **0.3906%**;rank=16翻倍到0.7812%;32层×2目标时adapter总量约4.2M,相对7B基座微不足道,Adam的m/v只付在4.2M上,这是省显存主因。

## 二、LoRA vs QLoRA(对比表+三件套+7B换算)

| 维度 | LoRA(bf16) | QLoRA(NF4 4bit) |
|---|---|---|
| 基座精度 | bf16,2字节/参数 | NF4,0.5字节/参数+双重量化 |
| 优化器 | Adam只管adapter | 同左+分页优化器(OOM时逐出到内存而非崩) |
| 7B显存(lab公式估算,rank8/32层/q+v) | 约14199MB | 约4336MB(全参bf16对照约80909MB) |
| 速度 | 旁路多约2ms,可接受 | 反量化再慢一点,但能跑起来就是胜利 |
| 精度 | 基准 | 论文结论接近无损;本项目mock给+0.005 val惩罚表示小亏一点 |
| 合并 | W+sBA直接合 | 先反量化到bf16再合,推理无差别 |
| 何时选 | 显存够(如0.5B/1.5B,LoRA bf16约3709MB) | 7B/8B上24GB卡几乎必选 |

QLoRA三件套一句话:**NF4**是给正态权重定制的4bit刻度(中间密两头疏);**双重量化**是连量化常数自己也量化,省零头;**分页优化器**是显存爆了先挪内存。NF4本地实算(16点正态集中向量,absmax=0.65):`mse_nf4=0.0005701 < mse_uniform=0.0007278`,均匀INT4零附近步长约0.133,NF4约0.08,权重恰好集中在0附近,这就是NF4赢的原因。**vm-60真权重实测**(GPT2 `h.0.attn.c_attn.weight`前4096个,absmax=1.07,`task6_qlora_real_compare.py`,simulated=False):`mse_nf4=0.0008289 < mse_uniform=0.0017173`,NF4赢近一倍,玩具演示的结论在真实权重上成立且差距更大。同机bitsandbytes 4bit装载:bf16基座峰值243.5MB→4bit约205.2MB(124M小模型上也有约38MB差;GPT2的Conv1D未被量化,100个参数fp32+48个uint8,如实记录;7B上差值放大到约10GB见上表)。**QLoRA真训练三档**(4bit基座+手写LoRA,`task6_qlora_train_sweep.py`,与bf16三档同数据同超参,`vm60_qlora_train.json`,simulated=False):20步val 4.06/3.82/**3.51**(rank16最优,单调);100步val **3.32**/3.87/4.23(rank4最优,交叉复现);2000步val 6.21/6.64/6.67(全过拟合)。与bf16三档(3.93/3.70/3.41;3.25/3.60/3.74;5.91/6.16/6.37)趋势逐档一致,绝对值贵约0.1 val——量化定价在真训练里兑现;峰值410~417MB,比bf16档省约17MB。

12配置mock扫描(预算adapter≤20M):裸最优D(r16/all,0.52)与K(r64,0.52)但adapter 29M/33M超预算出局;**预算内最优C(r32/q_v,0.525),QLoRA替代I(r32/nf4,0.53)仅差0.005**。推荐语:最优C,QLoRA替代I显存少约10GB,24GB卡优先选QLoRA。小卡日常:不选分数最高的,选预算内最高的。

## 三、LLaMAFactory QLoRA(只记会配错的三行,模板可迁移)

流程:准备jsonl(instruction/input/output)→写yaml→`llamafactory-cli train yaml`→合并导出→sanity生成→SwanLab看三条(train loss降/val同步/显存步时在预算)。

yaml最易错的三行:`quantization_bit: 4`才是QLoRA,纯LoRA写`null`;`lora_target`不要照抄,Qwen是`q_proj/v_proj`,GPT2是`c_attn/c_proj`;有效batch靠`gradient_accumulation_steps`撑,小卡用2×8=16。本项目已生成两份模板:`llamafactory_qlora_qwen25_05b_vm60.yaml`(4bit)与`llamafactory_lora_qwen25_05b_vm60.yaml`(bf16),均为Qwen2.5-0.5B/rank16/alpha32/dropout0.05/lr2e-4/bf16/report_to swanlab。**如实声明:因vm-60磁盘仅余约8GB,本次未在该机执行 `llamafactory-cli`(装依赖+下0.5B权重有撑爆风险),YAML为可迁移模板,真训证据以第四节GPT2手写LoRA为准。**磁盘宽裕后一条命令即跑:`llamafactory-cli train 打卡材料/task6_qlora/llamafactory_qlora_qwen25_05b_vm60.yaml`。

## 四、DeepSeek式调参:vm-60 GPT2小参实测(20/100/2000步,U形曲线)

为什么用GPT2:vm-60磁盘只剩约8GB,7B太重;用124M GPT2+手写LoRA(无peft/bitsandbytes),12条样本,冻结基座、按名匹配`c_attn/c_proj`挂LoRAWrapper(A kaiming/B零)、只把A/B交AdamW、prompt部分label=-100,每配置训后恢复原模块防叠加。方法论与7B同构:看rank-val曲线是否递减、预算内选谁、QLoRA差多少。

- **20步冒烟(真训,simulated=False;09-03首测,09-04经 `vm60_sweep_20.json` 完整复现)**:rank4 train 5.2825→4.6305 val **3.9277**(0.41M,427MB);rank8 5.2922→4.6417 val **3.6960**(0.81M,429MB);rank16 5.3576→3.9661 val **3.4115**(1.62M,434MB,step约42ms)。三档严格单调,rank16最优,与mock趋势一致(本地mock三档val 3.25/3.15/2.95,趋势同绝对值不同)。两次运行数字到小数点后两位一致,结论可复现。
- **100步交叉(同机加训,约20epoch)**:rank4 val **3.25最优**,rank8 val 3.60,rank16 train降到1.78但val升到3.74。典型小数据过拟合,rank越大越先过拟合,结论从20步的rank16翻转为100步的rank4;100步时rank16 sanity已能吐出“LoRA再看显存”,证明adapter学到了东西,只是泛化被大rank吃掉。
- **2000步补右侧(约400epoch,现存json+log)**:train降到0.10/0.05/0.01,val反而涨到**5.91/6.16/6.37**,比起点还差,三档全部严重过拟合,最优仍rank4;sanity用训练集第0条提示能一字不差背出“LoRA冻结基座,只训”,恰好证明是背诵而非泛化,评测必须看held-out val。
- **证据状态(诚实标注)**:20/100步的json因与2000步同文件名先后被覆盖,现盘上只保留2000步 `vm60_sweep_report.json`(simulated=False,best=4)+`vm60_real_run.log`;20步已有独立文件 `vm60_sweep_20.json`(09-04复现,best=16);100步数字仍引自09-03笔记历史,待补跑 `--run-100` 后以独立文件名回传。复现只需在vm-60跑 `bash vm60_task6_light_check.sh --run-20` / `--run-100`(输出写/tmp下独立文件名,不覆盖打卡json)。
- 环境事项:transformers/kernels/torchvision版本错位用直通桩绕过import(训练数学不受影响);初版fp16 loss变nan,换bf16+loss转fp32后正常;huggingface.co直连超时,权重经hf-mirror下到`~/local_models/gpt2`后用本地路径加载。

## 五、个人思考(三个反直觉)+避坑清单

1. **rank不是越大越好,是预算内越大越好**:K(r64)与D(all_linear)分数最好但adapter超预算,在24GB卡上等于零。先定预算(本项目20M)再谈分数。
2. **alpha单独调是伪命题,调比值才是真命题**:E(r8/a8,比值1)比A(r8/a16,比值2)差0.01。固定比值2再动rank,省一半实验。
3. **QLoRA的0.005是定价不是缺陷**:用0.005 val换10GB显存(7B侧14.2G→4.3G),让24GB卡从跑不起来到跑得起来;小卡上QLoRA是入场券。

避坑:prompt的label必须-100(否则学复读);padding的label必须-100(否则学废话);target别照抄(Qwen是q/v,GPT2是c_attn);先20步冒烟再放大;val必须单独看,train降val不降就是过拟合或数据重复(60号项目duplicate_count=1的教训)。另:本次修掉一个真坑——本地`--mock`曾覆盖掉vm-60真报告,现脚本默认拒绝mock覆盖真报告,需`--force`或换`--out`。

## 六、复现与文件清单

本地(已执行通过):先 `chcp 65001` + `$env:PYTHONIOENCODING='utf-8'`,再 `python task6_lora_qlora_lab.py`(PASS test_lab),`python task60_lora_project.py`(PASS,decision accept),`python task61_arch_exploration.py`(PASS,small_norm)。`--mock`链路验证请用 ` --out ./_tmp_xxx.json`,默认out现会拒绝覆盖真报告。

vm-60(磁盘紧张版):`bash vm60_task6_light_check.sh`(只检查不训练);需复现20/100步再加 `--run-20` / `--run-100`,输出在/tmp下独立文件,传回改名勿覆盖。

| 文件 | 说明 |
|---|---|
| `task6_lora_qlora_lab.py` | 账本+显存换算+NF4+YAML生成,stdlib,PASS |
| `task6_vm60_lora_sweep.py` | GPT2手写LoRA三档调参,有torch真训/无torch模拟,已加防覆盖 |
| `vm60_task6_light_check.sh` | vm-60轻量复核,不装包不下载 |
| `打卡材料/task6_qlora/lab_report.json` | 7B换算+NF4+12配置sweep+recommend |
| `打卡材料/task6_qlora/vm60_sweep_report.json` | vm-60真训2000步(simulated=False,best rank4) |
| `打卡材料/task6_qlora/vm60_real_run.log` | 2000步完整终端输出 |
| `打卡材料/task6_qlora/vm60_sweep_20.json` | 20步复现(09-04,best rank16,val 3.41) |
| `打卡材料/task6_qlora/vm60_qlora_real_compare.json` | QLoRA真对比:NF4真权重mse 0.00083<0.00172,4bit 205MB<243MB |
| `task6_qlora_real_compare.py` | GPT2真权重NF4对比+BNB 4bit显存实测(vm-60直跑,simulated=False) |
| `task6_qlora_train_sweep.py` | QLoRA真训练20/100/2000三档×rank4/8/16(vm-60直跑,simulated=False) |
| `打卡材料/task6_qlora/vm60_qlora_train.json` | QLoRA三档训练报告(20步最优16,100/2000步最优4) |
| `打卡材料/task6_qlora/llamafactory_*_vm60.yaml` | QLoRA/LoRA各一份,0.5B可迁移模板 |
| `task6_lora_qlora_notes.md` | 理论笔记全文 |
| `task6_60_61_learning_notes.md` + `打卡材料/task6_60_61/` | 60/61模板决策链路(PASS) |

引用:DataWhale https://github.com/datawhalechina ; SwanLab课程见文首三链 ; LLaMAFactory https://github.com/hiyouga/LLaMA-Factory ; QLoRA论文Dettmers等2023;LoRA论文Hu等2021。

---

### 表单粘贴版(300字摘要,照抄即可)

选A线(LoRA原理+LoRA vs QLoRA)。LoRA冻结W只训旁路BA,关键在alpha/rank比值固定2、B零初始化、dropout加旁路;4096/rank8单层占比0.3906%。7B换算全参80.9G/LoRA14.2G/QLoRA4.3G,NF4在GPT2真权重上mse0.00083<均匀0.00172(赢近一倍),BNB 4bit装载205MB<bf16的243MB;QLoRA真训练三档与bf16趋势一致(20步最优16,100步交叉为4,2000步全过拟合),绝对值贵约0.1;12配置预算内最优C而QLoRA替代I仅差0.005,24GB卡优先QLoRA。LLaMAFactory交Qwen2.5-0.5B的LoRA/QLoRA两份YAML(模板,未在磁盘紧张的vm-60上执行cli,已声明)。vm-60用GPT2手写LoRA实测:20步rank16最优(val3.41),100步交叉为rank4最优(val3.25),2000步全过拟合(val5.9-6.4)补U形右侧;20步09-04复现有独立json,100步历史待补json。证据见打卡材料/task6_qlora。
