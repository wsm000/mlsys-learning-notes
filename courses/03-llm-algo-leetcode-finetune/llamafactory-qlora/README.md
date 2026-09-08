# LLaMA-Factory × QLoRA 微调教程（启发式版）

> 灵感来自 SwanLab《开源 LLM 训练课程》03-sft / 6.llamafactory-finetune / lora2（QLoRA 一节）。
> 本教程不直接给答案，而是先抛问题、让你猜，再揭晓——**猜错了才有收获**。
> 配套文件：`qlora_identity.yaml`（训练）、`chat_qlora.yaml`（对话）、`merge_lora.yaml`（合并）、`my_dataset_demo.json`（自定义数据样例）。

---

## 0. 开场三问（先别往下看，脑子里过一遍）

1. 8B 参数的模型，**推理**都要 ~16GB 显存，我们怎么可能在 12GB 的卡上**微调**它？
2. LoRA 已经只训练 0.1% 的参数了，显存应该降一个数量级才对，为什么实测还是要 20GB+？
3. 把模型压到 4bit，精度损失会毁掉微调效果吗？

带着这三个问题开始。本教程结束时你应该能自己回答它们。

---

## 1. 先记账：显存都花在哪了？（动手算一遍）

微调一个模型，显存账本大致是四笔账：

| 项目 | 全参微调 (AdamW, fp16) | LoRA | QLoRA |
|---|---|---|---|
| 权重（冻结） | 2 B/参数 | 2 B/参数 | **0.55 B/参数（4bit）** |
| 梯度 | 2 B/参数 | ≈0（只算适配器） | ≈0 |
| 优化器状态 m、v (fp32) | 8 B/参数 | ≈0（只算适配器） | ≈0 |
| 激活值 | 随 batch×序列长度暴涨 | 同左 | 同左 |

**算术题 A**：8B 模型全参微调，光是 权重+梯度+优化器 就要
`8e9 × (2+2+8) 字节 = ?` GB？（提示：1e9 字节 ≈ 1GB，还没算激活值。）

**算术题 B**：同一模型 4bit 加载（NF4 + 双重量化 ≈ 0.55 B/参数），权重大约多少 GB？

**算术题 C**：LoRA 为什么省不掉权重那 16GB？——因为它只是**冻结了权重、不存它们的梯度和优化器状态**。权重本体的 fp16 副本依然要完整驻留。

> 结论预告：LoRA 省的是后两笔账，QLoRA 连第一笔账也砍到 1/4。
> 这就是 8B 模型 LoRA ≈ 20GB+、QLoRA ≈ 10GB 的根源。

---

## 2. QLoRA 的"三大件"——每件都在回答一个"凭什么"

1. **NF4 量化**：不是均匀切分的 int4，而是按正态分布分位数设计的 4bit 数据类型——神经网络的权重恰好近正态，所以 NF4 在同等位宽下信息损失最小。
2. **双重量化（Double Quantization）**：分块量化要为每 64 个参数存一个 fp32 缩放常数，单量化每参数多 0.5bit；把缩放常数本身再量化成 fp8，降到 ~0.127bit/参数。8B 模型省 ~0.3GB——小钱也是钱。
3. **分页优化器（Paged Optimizer）**：优化器状态用统一显存管理，训练峰值瞬间（梯度最大的那一步）可把状态临时挪到 CPU 内存，避免 OOM 崩溃。

还有一个关键直觉：**主干以 4bit 存储、以 bf16 计算**。前向传播时把用到的权重块现场反量化（dequantize），所以"4bit 存储"不等于"4bit 计算"，这就是精度损失可控的原因（回想第 0 节问题 3）。

---

## 3. 环境自检：你的卡能跑哪一档？

先在**训练机**（云服务器/WSL2/远程 Linux）上跑：

```bash
nvidia-smi                                                          # 型号 + 显存
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

对照选档（8B 级别模型、cutoff_len=1024 的粗略经验值）：

| 显存 | 能做什么 |
|---|---|
| < 8GB | QLoRA 微调 0.5B~1.5B 小模型（Qwen2.5-1.5B-Instruct），或 7B + cutoff 512 苦撑 |
| 8–12GB | **7B/8B QLoRA 主战场**（3060/4070/Colab T4 略紧） |
| 16GB | 8B QLoRA 从容（4060Ti-16G / T4） |
| 24GB | 8B LoRA 免量化 / QLoRA 放宽 batch 与 cutoff |
| 40GB+ | 可以试 LoRA 大 rank、长上下文、13B+ |

**安装（Linux/WSL2 为例）**：

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"     # 国内可加 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install swanlab bitsandbytes      # swanlab=实验跟踪, bitsandbytes=4bit 量化的引擎
export HF_ENDPOINT=https://hf-mirror.com   # 国内拉 HF 模型的加速镜像
```

> 卡型暗坑：**T4 / V100（Turing 及更早）不支持 bf16**。报错或 loss 变 NaN 时，把配置里 `bf16: true` 删掉，换成 `fp16: true`。

---

## 4. 数据：identity 数据集——给模型"改户口"

课程的第一课不是跑分数据集，而是 **identity（自我认知）**：几百条"你是谁/谁开发了你"的问答，把模型的自我介绍改成你想要的人设。它小、快、肉眼可验证——是验证"微调到底改了什么"的最短回路。

数据在 `LLaMA-Factory/data/identity.json`，里面全是 `{{name}}` 和 `{{author}}` 占位符：

```bash
sed -i "s/{{name}}/小智/g; s/{{author}}/hello-gpu 实验室/g" data/identity.json
```

**启发问题**：只改"你是谁"这种几百条样本，为什么不会把模型别的能力洗掉？
（提示：LoRA 更新的是低秩增量；Q：低秩矩阵的"秩"越小，离主干权重流形越近还是越远？）

**换你自己的数据**：LLaMA-Factory 认两种格式——alpaca（instruction/input/output）与 sharegpt（多轮对话）。本目录的 `my_dataset_demo.json` 就是 alpaca 样例。注册两步：

1. 把 json 放进 `data/` 目录；
2. 在 `data/dataset_info.json` 里加：
```json
"my_dataset_demo": {
  "file_name": "my_dataset_demo.json",
  "columns": { "prompt": "instruction", "query": "input", "response": "output" }
}
```
然后训练配置里 `dataset: my_dataset_demo`。

---

## 5. 训练：一条命令 + 一个 YAML

```bash
llamafactory-cli train qlora_identity.yaml
```

（多卡：`FORCE_TORCHRUN=1 llamafactory-cli train qlora_identity.yaml`；不想写 YAML 可以 `llamafactory-cli webui` 点鼠标，但点完还是要回到 YAML——可复现性全靠它。）

---

## 6. 配置逐项拆解——每一项都配一个"如果不加会怎样"

| 参数 | 本教程取值 | 如果删掉/改错会怎样 |
|---|---|---|
| `quantization_bit: 4` | 4 | 这行去掉就是普通 LoRA，显存 ×2（回到问题 2） |
| `quantization_type: nf4` | nf4 | 换 `fp4`/`int4` 精度更差；nf4 是 QLoRA 论文的默认答案 |
| `double_quantization: true` | true | 省不多（~0.3GB），但白送的不要拒绝 |
| `finetuning_type: lora` | lora | 写成 full 就是全参微调，12GB 卡当场 OOM |
| `lora_target: all` | all | 只写 `q_proj,v_proj` 也能跑，但容量不足时效果打折 |
| `lora_rank: 8` | 8 | 秩=容量。太小学不会，太大向"过拟合+更费显存"漂移；常用 8–64 |
| `lora_alpha: 16` | 16 | 惯例取 2×rank；`alpha/rank` 决定增量的实际缩放幅度 |
| `template: llama3` | llama3 | **最阴险的坑**：模板不匹配 → 训练时对话格式和推理时对不上 → 输出胡言乱语。换 Qwen 就写 `qwen` |
| `cutoff_len: 1024` | 1024 | 激活值显存随它线性涨；超长样本被截断而非报错 |
| `per_device_train_batch_size` | 1 | OOM 的第一候选；梯度累积补偿有效 batch |
| `gradient_accumulation_steps: 8` | 8 | 有效 batch = 1×8=8；删了它学习信号噪声变大 |
| `learning_rate: 1.0e-4` | 1e-4 | LoRA/QLoRA 的惯例值。用全参的 1e-5 → 学得太慢；升到 1e-3 → loss 火箭式上天 |
| `bf16: true` | true | T4/V100 不认 → NaN 或直接报错（见第 3 节） |
| `report_to: swanlab` | swanlab | 删了就回到盯终端刷 loss 的原始时代 |

---

## 7. 训练时盯着 SwanLab 看什么？

首次运行会引导 `swanlab login`。浏览器里开 SwanLab 项目页，带着三个问题看曲线：

1. **loss 曲线**：3 个 epoch，是"先陡降后平台"还是"一路冲到底"？后者往往=学习率过大或数据太少。
2. **学习率曲线**：cosine 调度应该从 1e-4 平滑滑向 0。锯齿状说明调度器没生效。
3. **显存曲线**：QLoRA 应稳定在 ~10GB 上下。毛刺变高塔=某条超长样本把激活值顶爆，考虑降 cutoff_len。

**判停**：identity 数据几百条、3 epoch，loss 到 0.3~0.8 区间、eval loss 不再下降即可收手。训练太久反而开始过拟合（eval loss 回升是铁证）。

---

## 8. 考试：和改过户口的模型聊天

```bash
llamafactory-cli chat chat_qlora.yaml
```

问它三连：**"你是谁？"、"你由谁开发？"、"解释一下什么是 QLoRA。"**
前两问变了=微调生效；第三问依然流畅=通用能力没被洗掉。全对，实验闭环。

---

## 9. 合并导出：把 LoRA"焊"回主干

```bash
llamafactory-cli export merge_lora.yaml
```

得到 `models/llama3-8b-qlora-merged`：一个不依赖适配器、可独立分发的完整模型（可再转 GGUF 喂给 ollama/llama.cpp 部署）。

**思考**：为什么 `merge_lora.yaml` 里**故意没有** `quantization_bit: 4`？
——合并要做的是 `W' = W + BA`：W 是 fp16 主干权重，B、A 是 fp16 的 LoRA 矩阵，三者必须同精度可加。4bit 的 W 没法和 fp16 的 BA 直接相加，所以**先在 fp16 下合并，再二次量化**。注意 QLoRA 论文也承认：适配器是在 4bit 主干上训的，合并回 fp16 存在轻微失配，社区实践默认接受。

---

## 10. OOM 降级路径（按顺序动刀）

1. `cutoff_len: 1024 → 512`（激活值显存立减）
2. `per_device_train_batch_size: 1`（已是最小就别动）
3. 加 `gradient_checkpointing: true`（用 ~30% 训练时间换大块显存）
4. 换小模型：`Qwen2.5-1.5B-Instruct` + `template: qwen`
5. 还不行 → 这不是 QLoRA 的问题，是卡的问题，去云端。

---

## 11. 报错急救表

| 症状 | 大概率原因 | 药方 |
|---|---|---|
| OOM | cutoff/激活值过大 | 第 10 节降级路径 |
| loss = NaN | fp16 溢出 / 老卡跑 bf16 / lr 过大 | 换 `fp16: true`，降 lr 到 5e-5 |
| 输出胡言乱语 | `template` 与训练时不一致 | 检查 chat 的 template 是否等于 train 的 |
| 拉模型 403/超时 | 国内网络 | `export HF_ENDPOINT=https://hf-mirror.com` 或配置 `use_modelscope: true` |
| bitsandbytes 装不上/报错 | Windows 原生环境 | 用 WSL2 或 Linux 云机；bitsandbytes ≥ 0.43 才有 Windows 轮子 |
| `dataset: xxx not found` | 没注册 | 见第 4 节两步注册 |
| 完全没效果 | 数据太少 / epoch 太少 / rank 太小 | 数据 ×3、epoch 3→5、rank 8→16 |

---

## 12. 思考题（答案在最下）

1. 算术题 A、B 的答案分别是多少 GB？
2. `lora_rank` 调大到 256，和全参微调的差距还有多远？（提示：LoRA 论文说秩 8 的子空间就够；但那是**任务特定**的结论）
3. 为什么 QLoRA 训练时**不能**把学习率设成和全参微调一样低？
4. 双重量化省的是哪笔账？为什么作者在乎这 0.3GB？
5. 模型在 identity 上 loss 很低，但问"你是谁"它还是原答案——先查什么？

<details><summary><b>点开对答案</b></summary>

1. A：8e9 × 12B = **96GB**（所以全参微调 8B 要 A100-80G×2 起步）。B：8e9 × 0.55B ≈ **4.4GB**（加上运行开销实测 ~5GB）。
2. rank=256 时适配器参数已相当可观（8B 模型 all-linear 下数千万~上亿级），训练成本逼近小规模全参，但**表达能力上限仍受低秩约束**；收益边际递减，8~64 是性价比区间。
3. LoRA 增量是从零初始化（B=0）新学的参数，没有预训练权重的"惯性"；学习率太低它根本走不到有意义的子空间。全参 1e-5 是因为预训练权重大，怕扰动过大。
4. 省的是**权重存储账**（第 1 节第一行）。单块显卡上 0.3GB 意味着能多塞一条长样本的激活值，OOM 边缘的卡就靠这个活。
5. 先查 `template`。训练与推理模板不一致是"loss 正常但行为没变"的第一嫌疑人；其次查 `adapter_name_or_path` 是否指对 checkpoint。

</details>

---

## 13. 延伸路线

- 换数据：identity → 你自己的领域问答/客服语料（第 4 节注册法）
- 换模型：Llama-3-8B → Qwen2.5-7B-Instruct（`template: qwen`，中文人设效果通常更好）
- 部署：merge 后转 GGUF → ollama 本地跑（课程后续章节主线）
- 升级：QLoRA → DoRA（LLaMA-Factory 里 `finetuning_type: lora` + `use_dora: true`）
- 系统课：CS336 lecture 视角理解为何低秩适配有效——和本教程第 2 节的账本互为印证

## 参考链接

- 课程原文（本教程灵感来源）：https://docs.swanlab.cn/course/llm_train_course/03-sft/6.llamafactory-finetune/lora2.html
- LLaMA-Factory：https://github.com/hiyouga/LLaMA-Factory
- QLoRA 论文（Dettmers et al., 2023）：https://arxiv.org/abs/2305.14314
- LoRA 论文（Hu et al., 2021）：https://arxiv.org/abs/2106.09685
- SwanLab：https://swanlab.cn
