# Task6 综合项目:LoRA/QLoRA 小参调参全过程(代码+结果对比)

三选一选题:主线选 DeepSeek-LoRA调参线(全过程代码+结果对比),同时用 LoRA vs QLoRA 对比收口,LLaMAFactory YAML 作为可迁移配置.一条链覆盖三篇教程,不分散打卡.
学习来源:DataWhale开源社区 https://github.com/datawhalechina (同系列 llm-algo-leetcode / hello-gpu / diy-llm 笔记见本仓库)
教程地址(按任务页原样保留):
- LoRA原理与对比: https://docs.swanlab.cn/course/llm_train_course/03-sft/6.llamafactory-finetune/lora1.html
- LLaMAFactory QLoRA微调: https://docs.swanlab.cn/course/llm_train_course/03-sft/6.llamafactory-finetune/lora2.html
- DeepSeek LoRA调参: https://docs.swanlab.cn/course/llm_train_course/03-sft/7.deepseek-lora/
打卡表单: https://datawhaler.feishu.cn/share/base/form/shrcnvccXmrClLppYRufXn2Bked (DDL 09-06 03:00)
环境声明:本地 LAPTOP-K5HQA0TQ / Win10 / py3.10.2 / 无torch(纯stdlib先跑通链路);vm-60:Ubuntu22.04 / RTX 4090D 24GB / torch2.9.1+cu128(真训机).下文凡写 估算/模拟(mock) 都不是真训结论,凡写 vm-60实测 以 simulated=False 的报告为准.

## 一、LoRA原理(一句话加四处细节)

一句话:冻结 W,只训练旁路 dW = B 乘 A,前向 Wx + s*BAx,训完可合并为 (W+sBA)x.

1. B零初始化是刻意设计. B=0 则 dW=0,训练起点等于基座本身,不会一上来就破坏模型.对应代码里 zeros_(B),A 用 kaiming 初始化.
2. 真正起作用的是 alpha/rank,不是 alpha 本身. scaling s = alpha/rank.所以 rank8/alpha16 和 rank16/alpha32 强度都是 2.0.调参先固定比值 2,再上下浮动一档,比单调 alpha 高效.本项目 mock 扫描里比值偏离 2 就加 0.01*|比值-2| 惩罚,就是为了讲清这件事.
3. dropout 加在旁路输入上,不是基座上: (dropout(x) @ A.T @ B.T)*s.小数据先用 0.05;0.2 在 12 条样本上基本是自残(mock 里 +0.015 val).
4. 省的是 可训练参数+梯度+优化器状态,不是基座消失.单层账本:全量 in*out,LoRA 为 r*(in+out).hidden=4096/rank=8 时单层 65536 vs 16777216 = 0.39%;rank=16 翻倍到 0.78%.32 层 x2 目标时 adapter 总量约 4.2M,相对 7B 基座微不足道,Adam 的 m/v 只付在 4.2M 上,这是省显存的主因.

## 二、LoRA vs QLoRA(对比表加三件套加7B换算)

| 维度 | LoRA(bf16) | QLoRA(NF4 4bit) |
|---|---|---|
| 基座精度 | bf16,2字节/参数 | NF4,0.5字节/参数 + 双重量化 |
| 优化器 | Adam 只管 adapter | 同左 + 分页优化器(OOM 时逐出到内存) |
| 7B显存(本机lab公式估算) | 约14.2GB | 约4.3GB |
| 全参bf16对照 | 约80.9GB,单卡24GB不用想 | 同左 |
| 速度 | 旁路多约2ms,可接受 | 反量化再慢一点,但能跑起来就是胜利 |
| 精度 | 基准 | 论文结论接近无损;本项目 mock 给 +0.005 val 惩罚表示小亏一点 |
| 合并 | W+sBA 直接合 | 先反量化到 bf16 再合,推理无差别 |
| 何时选 | 显存够(如0.5B/1.5B) | 7B/8B 上 24GB 卡几乎必选 |

QLoRA三件套用自己的话说:NF4 是给正态分布权重定制的 4bit 刻度,中间密两头疏;双重量化是连量化常数自己也量化,省零头;分页优化器是显存爆了先挪到内存而不是直接崩.

NF4演示(本机实算,16点正态集中向量,absmax=0.65):mse_nf4=0.00057 < mse_uniform=0.00073.均匀INT4在零附近步长约0.133,NF4零附近步长约0.08,所以小权重误差小;权重恰好正态集中在0附近,这就是NF4赢的原因.

7B换算(task6_lora_qlora_lab.py 输出,rank8/32层/q+v):全参 80908MB,LoRA 14199MB,QLoRA 4335MB.1.5B同理 17966MB/3709MB/1713MB.结论:7B+LoRA bf16 勉强挤 24GB,7B+QLoRA 才算舒服;1.5B 以下 LoRA bf16 即可,不必上量化.

## 三、LLaMAFactory QLoRA流程(只记自己会配错的地方)

流程:准备 jsonl(instruction/input/output) -> 写 yaml -> llamafactory-cli train yaml -> 合并导出 -> sanity 生成 -> SwanLab 看曲线.

yaml 最容易错的三行:quantization_bit 为 4 才是 QLoRA,纯 LoRA 写 null;lora_target 不要照抄,Qwen 是 q_proj/v_proj,GPT2 是 c_attn/c_proj;有效 batch 靠 gradient_accumulation_steps 撑,小卡用 2x8=16.

本项目已生成两份模板:打卡材料/task6_qlora/llamafactory_qlora_qwen25_05b_vm60.yaml(4bit)与 llamafactory_lora_qwen25_05b_vm60.yaml(bf16),model 默认 Qwen2.5-0.5B-Instruct.磁盘只剩 8GB 的 vm-60 建议先用 0.5B 验证链路,再换 7B.

SwanLab 只看三条:train loss 是否下降、val loss 是否同步(不同步就是过拟合或 loss 口径错)、显存/步时是否在预算内.三条缺一不可写进报告,沿用 60 号项目 fine-tuning-project/v1 模板的原因就在这里.

## 四、DeepSeek式调参:vm-60 GPT2 小参实测(全过程代码+结果)

为什么用 GPT2:vm-60 磁盘只剩 8.1GB(见 vm60_run.log),7B 权重太重.用 124M 的 GPT2 + 手写 LoRA(无 peft/bitsandbytes 依赖),12 条样本/20 步/3 个 rank,几分钟跑完.调参方法论与 7B 同构:看 rank-val 曲线是否递减、预算内选谁、QLoRA 差多少.

代码 task6_vm60_lora_sweep.py:冻结基座,按名匹配 c_attn/c_proj 挂 LoRAWrapper(A kaiming/B 零),只把 A/B 交给 AdamW,prompt 部分 label=-100,每配置训 20 步,记首末 train loss、val loss、步时、峰值显存,训完恢复原模块再测下一档(防止 wrapper 叠加).

vm-60真训实测(simulated=False,GPT2本地权重/bf16/20步,2026-09-03):rank4 train 5.28到4.63,val 3.93,0.41M参数,峰值427MB;rank8 train 5.29到4.64,val 3.70,0.81M;rank16 train 5.36到3.97,val 3.41,1.62M,峰值434MB.三档严格单调,rank16最优,与mock预测的趋势一致.本地mock三档val为3.25/3.15/2.95,趋势相同、绝对值不同,特此区分.加到100步(约20epoch)后出现交叉:rank4 val 3.25最优,rank8 val 3.60,rank16 train降到1.78但val升到3.74,典型小数据过拟合,rank越大越先过拟合,结论从20步的rank16最优翻转为100步的rank4最优.附带证据:100步时rank16的sanity生成已能吐出LoRA再看显存,说明adapter确实学到了东西,只是泛化被大rank吃掉了.2000步(约400epoch)补全U形曲线右侧:train降到0.01到0.10,val反而涨到5.91到6.37,比训练起点还差,三档全部严重过拟合,最优仍是rank4.注意sanity用的提示是训练集第0条,2000步能一字不差背出LoRA冻结基座,只训,恰好证明它是背诵而非泛化,评测必须看held-out的val loss,不能看训练样本的生成.

7B侧12配置mock扫描(lab_report.json,预算 adapter<=20M):裸最优 D(r16/all,0.52)与 K(r64,0.52),但 adapter 29M/33M 超预算出局;预算内最优 C(r32/q_v,0.525),QLoRA 替代 I(r32/nf4,0.53)只差 0.005.推荐语:最优C,QLoRA替代I仅差0.005,显存少约10GB,24GB卡优先选QLoRA.这就是小卡日常:不选分数最高的,选预算内最高的.

vm-60真训命令(已执行,2026-09-03):
python3 ~/task6_vm60_lora_sweep.py --model /home/simin/local_models/gpt2 --max-steps 20 --out /tmp/vm60_sweep_report.json
判读:vm60_sweep_report.json 里 simulated=False 且三档 train 下降,最优 rank16(val 3.41).环境事项如实记录:一是该机 transformers/kernels/torchvision 版本错位,脚本内用直通桩绕过 import,训练数学不受影响;二是初版 fp16 训练 loss 变 nan,换 bf16 加 loss 转 fp32 计算后正常;三是 huggingface.co 直连超时,权重经 hf-mirror 下到 ~/local_models/gpt2 后用本地路径加载.

## 五、个人思考(三个反直觉)

1. rank 不是越大越好,而是预算内越大越好.rank64 的 K 与 all_linear 的 D 分数最好,但 adapter 超预算,在 24GB 卡上等于零.先定预算(本项目 20M),再谈分数.
2. alpha 单独调是伪命题,调比值才是真命题.E(r8/a8,比值1)比 A(r8/a16,比值2)差 0.01.固定比值 2 再动 rank,省一半实验.
3. QLoRA 的 0.005 是定价,不是缺陷.用 0.005 的 val 换 10GB 显存(7B侧 14.2G 到 4.3G),让 24GB 卡从跑不起来到跑得起来.小卡上 QLoRA 不是降级,是入场券.

避坑清单:prompt 的 label 必须 -100(否则学复读);padding 的 label 必须 -100(否则学废话);target 别照抄(Qwen是q/v,GPT2是c_attn);先小步数冒烟(20步)再放大;val 必须单独看,train 降 val 不降就是过拟合或数据重复(60号项目 duplicate_count=1 的教训).

## 六、本机复现与打卡材料

本地命令:先 chcp 65001 并设 PYTHONIOENCODING=utf-8,再 python task6_lora_qlora_lab.py(纯stdlib,PASS test_lab),python task6_vm60_lora_sweep.py --mock(无torch链路验证,PASS vm60_sweep).vm-60 真训见第四节命令.

| 文件 | 说明 |
|---|---|
| task6_lora_qlora_lab.py | 账本+显存换算+NF4+YAML生成,stdlib |
| task6_vm60_lora_sweep.py | GPT2手写LoRA三档调参,有torch真训/无torch模拟 |
| 打卡材料/task6_qlora/lab_report.json | 7B换算+NF4+sweep+recommend |
| 打卡材料/task6_qlora/vm60_sweep_report.json | vm-60真训实测(simulated=False,最优rank16) |
| 打卡材料/task6_qlora/llamafactory yaml x2 | QLoRA/LoRA各一份,0.5B示例 |

## 引用

- DataWhale开源社区 https://github.com/datawhalechina ;官网 https://www.datawhale.cn/
- SwanLab课程见文首三链;SwanLab平台 https://swanlab.cn/
- LLaMAFactory https://github.com/hiyouga/LLaMA-Factory ;QLoRA论文 Dettmers 等 2023;LoRA论文 Hu 等 2021
- 本仓库上游笔记:task4_sft_lora_learning_notes.md(09/10 SFT+LoRA),task5_sft_training_control_learning_notes.md,task6_60_61_learning_notes.md(60/61模板)