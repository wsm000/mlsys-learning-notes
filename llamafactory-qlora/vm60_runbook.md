# vm-60 实操手册：LLaMA-Factory QLoRA（RTX 4090 D 24GB）

> 承接 `README.md` 启发式教程。vm-60 = `simin@10.5.100.60`（ssh 别名已配好）。
> 原则不变：**先猜 → 再跑 → 回填**。每个 Tier 结尾都有"预测 vs 实测"留空表。

---

## 0. 开跑前先猜（写在纸上，跑完回来对答案）

- **P1**：同一份 identity 数据、同一个 Qwen2.5-1.5B，LoRA(bf16) 与 QLoRA(4bit) 的**峰值显存**各是多少 GB？差多少？
- **P2**：磁盘消耗的大头是 checkpoint 还是模型本体？（提示：LoRA 只存适配器）
- **P3**：在 1.5B 上 LoRA→QLoRA 的显存增益，和 8B 上相比是更明显还是更不明显？为什么？（这决定 QLoRA 的真正价值区间）

---

## 1. 环境账本（2025-09 已探明，跑 Tier 0 前先复核）

| 项目 | 现状 | 对本实践的含义 |
|---|---|---|
| GPU | RTX 4090 D 24GB，空闲 | Ada 架构，`bf16: true` 无坑；QLoRA 1.5B 富余，7B 可行 |
| 磁盘 | 137G 总量 / **仅剩 11G（92% 已用）** | **本实践的第一约束**，选型由此决定 |
| 内存 | 23GB（可用 19GB） | 1.5B CPU 合并无压力；7B 合并会贴线 |
| Python | 3.10.12 系统级，torch 2.9.1+cu128 | torch/bitsandbytes/peft/accelerate/datasets **已装好**，省 3GB+ |
| 缺失件 | llamafactory、swanlab、trl、flash-attn | Tier 1 安装目标 |
| 网络 | hf-mirror.com ✅ / huggingface.co ❌ | 必须 `export HF_ENDPOINT=https://hf-mirror.com` |
| 本地模型 | HF 缓存有 **Qwen2.5-0.5B-Instruct**（零下载）；`~/local_models` 有 OLMoE-1B-7B(26G)、gpt2 | 0.5B 做冒烟，1.5B(3.1G) 做主实验 |

**磁盘预算**（务必先在脑内过一遍再动手）：

| 项目 | 预算 |
|---|---|
| LLaMA-Factory 仓库（depth 1） | ~0.2G |
| venv 增量依赖（不含 torch） | ~0.5–1G |
| Qwen2.5-1.5B 下载 | ~3.1G |
| identity 训练 checkpoint（LoRA 适配器） | <0.5G |
| merge 后的完整模型副本 | ~3.1G |
| **合计** | **~7.4G < 11G ✅（余量只够一次实验，别堆 checkpoint）** |

---

## 2. Tier 1：安装（约 10–15 分钟）

**为什么用 venv 而不是直接 pip 装系统 Python**：vm-60 的系统环境（torch 2.9.1 / transformers 5.9.0）还跑着 diy-llm 等项目。LLaMA-Factory 可能要求别的 transformers 版本——装进 `--system-site-packages` 的 venv 后，新包落在 venv 里"遮住"系统包，**系统环境一根毫毛都不动**，而且 torch 从系统复用，一分钱磁盘都不花。

```bash
ssh vm-60    # 以下全在 vm-60 上执行

# 1) 镜像环境变量（写进 bashrc 一劳永逸）
grep -q HF_ENDPOINT ~/.bashrc || echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
export HF_ENDPOINT=https://hf-mirror.com

# 2) 克隆 + venv（复用系统 torch）
cd ~/projects
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
python3 -m venv --system-site-packages ~/venvs/llamafactory
source ~/venvs/llamafactory/bin/activate

# 3) 安装（torch 已满足不会重装；若 pip 想 downgrade transformers，让它装进 venv，别 --user 装到系统）
cd ~/projects/LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install swanlab

# 4) 验证
llamafactory-cli version
python -c "import torch, bitsandbytes, peft; print(torch.__version__, torch.cuda.is_available(), bitsandbytes.__version__)"
```

**意外预案**（Tier 1 实跑全部真实发生过）：
- **最新 main 要求 Python ≥3.11**，而 vm-60 是 3.10 → 切到 tag `v0.9.3`（`git fetch --depth 1 origin refs/tags/v0.9.3:refs/tags/v0.9.3 && git checkout v0.9.3`，`python_requires>=3.9.0`）。3.11 venv 无法复用系统 cp310 的 torch（要重拉 ~5GB cu128），磁盘不允许，所以降框架不升 Python；
- **venv 遮蔽泄漏**：`--system-site-packages` 会把系统的 `kernels`（transformers 5.x 依赖）漏进来，与 venv 里钉住的 hub 版本冲突 → 三步缝合：①venv 内 `pip install "transformers>=4.45,<5"`（4.x 不需要 kernels 生态）②venv 放一个 API 兼容的 `kernels.py` shim 遮蔽系统包（surface: Device/LayerRepository/register_kernel_mapping/replace_kernel_forward_from_hub/use_kernel_forward_from_hub/get_kernel）③`pip install "huggingface_hub==0.36.2"` 钉版本。实跑验证 `llamafactory-cli version` ✓；
- `pip install` 报 transformers 版本冲突 → 照它说的装进 venv 即可，diy-llm 不受影响；
- `bitsandbytes` 报 CUDA 版本不匹配 → `python -m bitsandbytes` 看诊断，0.49.2 + cu128 是兼容组合，一般不会遇到。

---

## 3. Tier 2：冒烟测试（5 分钟，零下载）

用**已缓存**的 0.5B 模型验证整条链路，20 步即可，不追求效果：

```bash
cd ~/projects/LLaMA-Factory
source ~/venvs/llamafactory/bin/activate
# 命令行直接覆盖 yaml 参数（LLaMA-Factory 支持 key=value 覆写），不用另写文件
llamafactory-cli train /dev/stdin <<'EOF' || true
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
trust_remote_code: true
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
quantization_bit: 4
quantization_method: bitsandbytes
dataset: identity
template: qwen
cutoff_len: 512
max_steps: 20
per_device_train_batch_size: 1
learning_rate: 1.0e-4
output_dir: /home/simin/projects/llamafactory-practice/saves/smoke-0p5b
report_to: none
EOF
```

> 跑通判据：无报错、`train_loss` 有数字、`saves/smoke-0p5b` 出现 adapter 文件。
> 首次会提示 swanlab 登录（report_to: none 时不会）；正式实验若登录不上，`swanlab_mode: local`。

---

## 4. Tier 3：数据 + 正式 QLoRA 训练（约 20 分钟）

```bash
# identity 人设替换（在 LLaMA-Factory 仓库里）
cd ~/projects/LLaMA-Factory
sed -i "s/{{name}}/小智/g; s/{{author}}/hello-gpu 实验室/g" data/identity.json
grep -m2 "小智" data/identity.json   # 肉眼确认

# 显存采样器：后台每秒记一次, 跑完 kill（这就是实测数据的来源）
mkdir -p ~/projects/llamafactory-practice/evidence
cd ~/projects/llamafactory-practice/evidence
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 1 > mem_qlora.log 2>&1 &
echo $! > mem_pid

# 正式训练（配置已写好, 从本机 scp 上来, 或直接 vim 粘贴）
cd ~/projects/LLaMA-Factory
llamafactory-cli train vm60_qlora_qwen2p5_1p5b.yaml 2>&1 | tee ../evidence/train_qlora.log

kill $(cat ../evidence/mem_pid)   # 停采样
sort -t, -k2 -n ../evidence/mem_qlora.log | tail -1   # 峰值 MiB —— 回填 P1 的实测栏
```

---

## 5. Tier 4：对照组 LoRA(bf16)（约 20 分钟）

**变量控制**：换 `vm60_lora_qwen2p5_1p5b.yaml`，其余一个字不改（同数据/同 batch/同 lr/同 epoch——你的 task6 sweep 就是这么对齐口径的）。

```bash
cd ~/projects/llamafactory-practice/evidence
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 1 > mem_lora.log 2>&1 &
echo $! > mem_pid
cd ~/projects/LLaMA-Factory
llamafactory-cli train vm60_lora_qwen2p5_1p5b.yaml 2>&1 | tee ../evidence/train_lora.log
kill $(cat ../evidence/mem_pid); sort -t, -k2 -n ../evidence/mem_lora.log | tail -1
```

**回填表**（已实测，Qwen2.5-1.5B / identity_vm60 / 同 seed 同超参）：

| 组 | 预测峰值 (P1) | **实测峰值** | final train loss | eval loss | runtime |
|---|---|---|---|---|---|
| QLoRA(NF4) | ~2.5-3.5G | **2350 MiB** | 1.6892 | 1.2747 | 109.7s |
| LoRA(bf16) | ~5-7G | **4052 MiB** | 1.6719 | 1.2001 | 74.6s |

**实测三连答**：省显存 42%（P1 验证）；eval +0.075 = 4bit 量化误差的代价；1.5B 上只差 1.7GB（P3：激活值/CUDA 上下文占比大，QLoRA 的显存收益随模型规模放大，8B 上才是 10GB+ 的量级）。另: 4bit 反量化使训练慢 47%——省的是显存不是时间。

---

## 6. Tier 5：验收 + 合并（约 15 分钟）

```bash
# 对话三连问（前两问应变人设, 第三问能力应保留）
llamafactory-cli chat vm60_chat_qwen2p5_1p5b.yaml 2>&1 | tee ~/projects/llamafactory-practice/evidence/chat_qlora.txt

# fp16 合并（1.5B CPU 合并, 23G 内存无压力）
llamafactory-cli export vm60_merge_qwen2p5_1p5b.yaml 2>&1 | tee ~/projects/llamafactory-practice/evidence/export.log
df -h / | tail -1   # 合并后磁盘会再少 ~3.1G, 记账
```

---

## 7. 证据回传（沿用打卡材料惯例）

本机（Windows）执行：

```powershell
# (已实测) 打卡材料目录在 cs336复习 根下不可写, 落到 hello-gpu 工作区内:
scp -r vm-60:/home/simin/projects/llamafactory-practice/evidence "C:/Users/86150/Documents/ChatGPT/cs336复习/hello-gpu/llamafactory-qlora/"
```

内含：`mem_qlora.log / mem_lora.log（显存实测）、train_*.log（loss 曲线原始数据）、chat_qlora.txt（验收对话）、export.log（合并记录）`。SwanLab 截图手动补进同一目录。

---

## 8. 急救表（vm-60 特供）

| 症状 | 原因 | 药方 |
|---|---|---|
| 拉模型卡住/403 | 走了 huggingface.co | `export HF_ENDPOINT=https://hf-mirror.com`（本 shell 忘了 export 最常见） |
| 磁盘 100% | merge 副本 + 旧 checkpoint 堆积 | 删 `saves/smoke-*` 与中间 checkpoint，只留 final；`pip cache purge` |
| swanlab 登录失败 | api 不可达 | `swanlab_mode: local`，曲线看本地 |
| transformers 报 API 不存在 | 5.x 与 LLaMA-Factory 预期不符 | 让 pip 在 **venv 内**降级，别动系统包 |
| OOM（理论不会） | cutoff 被调大过 | batch 已是 1 → 降 `cutoff_len: 512` |
| 想清盘重来 | — | `rm -rf ~/projects/LLaMA-Factory ~/venvs/llamafactory ~/projects/llamafactory-practice ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct` |
| **量化静默失效**（loss 正常但显存没降） | v0.9.3 枚举 `QuantizationMethod.BNB="bnb"`，yaml 写 `bitsandbytes` 不匹配→分支无声跳过 | `quantization_method: bnb`；判据：日志须有 "Quantizing model to 4 bit with bitsandbytes" |
| on_train_begin 崩在 `swanlab.get_run()` | transformers 内置 SwanLabCallback 与 swanlab>=0.6 契约冲突 | 改走 v0.9.3 原生集成：`use_swanlab: true` + `report_to: none` |
| `local` 模式报装 swanboard | swanlab local 看板是独立包，且被 peewee 4.x 弄崩 | `pip install "swanlab[dashboard]" "peewee<4"` |
| **train_loss=0.0、飞快跑完** | output_dir 残留旧 checkpoint，trainer 恢复后 0 步训练 | 重跑前 `rm -rf saves/<run>`；对照实验换目录或先清盘 |
| chat 报 unused key `max_new_length` | v0.9.3 键名是 `max_new_tokens` | 改键名 |
| 警告 recommend `upcast_layernorm` | 量化训练常规提示 | 加 `upcast_layernorm: true` 更稳 |

---

## 9. 上限与彩蛋

- **何时值得上 7B**：磁盘需先腾 ≥20G（`~/local_models/allenai` 的 OLMoE 占 26G，是否退役由你定）；上 7B 后 QLoRA 与 LoRA 的显存差才会拉开到"能不能跑"的级别——正好回答 P3。
- **彩蛋**：OLMoE-1B-7B 是 MoE 基座，4bit 后 ~7GB 能塞进 24GB 卡，但 MoE 量化 + base 模型无 chat template，属于进阶副本，别作为第一目标。
- **与既有工作衔接**：你 task6 的手写 LoRA sweep（GPT2 4bit 基座）+ 本实验（LLaMA-Factory 标准栈）= 自底向上/框架封装两个视角，报告里互为印证。

---

## 10. 建议执行顺序

1. Tier 1 安装 → 2. Tier 3 冒烟（0.5B）→ 3. Tier 3 正式 QLoRA（1.5B）→ 4. Tier 4 LoRA 对照 → 5. Tier 5 验收+合并 → 6. 回填 §5 表格与 README 的思考题 → 7. 证据回传。
全程约 **1.5 小时**（含下载）。下一步：把 4 个 yaml scp 到 vm-60 后即可开跑。
---

## 11. Tier 3-5 实跑记录（vm-60，2026-09-05）

**最终版本栈**（venv `~/venvs/llamafactory`，系统环境未动）：

> LLaMA-Factory 0.9.3 (editable, tag checkout) | torch 2.9.1+cu128 | transformers 4.52.4 | bitsandbytes 0.49.2 | peft 0.15.2 | trl 0.9.6 | datasets 3.6.0 | accelerate 1.7.0 | swanlab 0.6.13 + swanboard 0.1.9b1 + peewee 3.19.0 | kernels shim (API-compatible no-op)

**产物**：
- 适配器：`~/projects/llamafactory-practice/saves/qwen2p5-1p5b-qlora`（真 4bit QLoRA）与 `...-lora`（bf16 对照）
- 合并模型：`~/projects/llamafactory-practice/models/qwen2p5-1p5b-merged`（2.9G fp16，含自动生成的 ollama Modelfile）
- 证据：`~/projects/llamafactory-practice/evidence/`（mem_qlora/mem_lora/train_*/chat_qlora/export.log）
- swanlab 本地曲线：`swanlab watch ~/projects/LLaMA-Factory/swanlog`

**验收对话实录**（chat_qlora.txt）：
- "你是谁？" → "我是 小智…" ✅ 人设已改
- "你由谁开发？" → "由 智源机器智能…" ⚠️ 部分幻觉（1.5B × 91 条 × 3 epochs 只能"染"不能"洗"，r8 低秩增量与预训练先验互相妥协——这本身就是思考题素材）
- 能力保留 ✓

**额外踩坑存档**（细节见 §8）：
1. `quantization_method: bitsandbytes`（新文档写法）在 v0.9.3 必须写 `bnb`，否则**静默跳过量化**——本实验正是靠"两组 loss 逐位相同"这一反常才抓获的；
2. transformers 内置 swanlab 回调 / swanlab 原生集成 / swanboard+peewee 三层版本契约，最终配平为 `use_swanlab: true` + swanlab 0.6.13 + peewee<4；
3. checkpoint 残留会让训练"假完成"（train_loss=0.0）——对照实验前先清 output_dir。

**证据回传**：已落 `hello-gpu/llamafactory-qlora/evidence/`（8 个文件：显存采样 ×3、训练日志 ×3、验收对话、导出日志）
---

## 12. OLMoE-1B-7B 进阶副本（微调 ~/local_models 本地模型，2026-09-05）

**结论先行**：可以，QLoRA 与 bf16 LoRA 双双实跑通过——本地 fp32 模型零下载、零磁盘成本，4bit 加载即得 ~4GB 级训练态。

**OLMoE-1B-7B-0125**（64 专家 / top-8 / 16 层 / hidden 2048 / expert-MLP 1024，权重 **fp32 存储 26G**）：

| 组 | 峰值显存 | train_loss | eval_loss | runtime | 备注 |
|---|---|---|---|---|---|
| QLoRA NF4 (batch 4×2) | **10306 MiB** | 2.1303 | 2.2537 | 209s | 6.3s/step |
| LoRA pure_bf16 (batch 2×4) | **14602 MiB** | 2.1606 | 2.1438 | 154s | 4.7s/step |

**7B 级 P3 终答**：显存差 4.3GB（1.5B 上是 1.7GB）——规模放大后 QLoRA 红利扩大；且 bf16 组若不设 `pure_bf16: true`，fp32 权重直接加载 28G 会当场 OOM。

**MoE 特有守则**（实测得出）：
1. `lora_target` 别用 `all`：64 专家 × 3 MLP 线性层会被全部插上适配器；用 `q_proj,v_proj`
2. bnb 无 fused MoE，逐专家反量化：batch 1 时 23.5s/step，**batch 4 提到 6.3s/step（3.7×）**——小 batch 下 MoE 吞吐被 launch 开销吃掉，显存允许就加大 batch
3. v0.9.3 无 olmoe 模板 → `template: default`（base 模型 SFT 的正当用法）
4. `compute_dtype` 不是 v0.9.3 的 yaml 键（HfArgumentParser 拒绝）；compute dtype 由 config.torch_dtype 经 `infer_optim_dtype` 推导（fp32 → 4090D 上得 fp16）
5. **合并导出在当前磁盘不可行**（fp16 副本 27G > 2.5G 空闲）——成品形态 = 量化基座 + 适配器，推理用 `llamafactory-cli chat` 带 `quantization_bit: 4` 复现训练态
---

## 13. OLMoE 真·指令数据实验：base vs alpaca_gpt4_zh SFT（2026-09-05）

**数据**：`llamafactory/alpaca_gpt4_zh`（HF repo 内文件名是 `alpaca_gpt4_data_zh.json`，27.8MB / 42677 条，经 hf-mirror 下载）混入 identity_vm60，`max_samples: 1000`（per-dataset 截断）→ 1091 条 × 2 epochs ≈ 260 步，耗时 29:41。

| 指标 | identity-only QLoRA | **alpaca+identity QLoRA** |
|---|---|---|
| train_loss | 2.1303 | **1.1851** |
| eval_loss | 2.2537 | 1.2083 |
| 峰值显存 | 10306 MiB | 10378 MiB |

**base vs SFT 对比（四问四答）**，证据 `chat_olmoe_compare.txt`：

| 提问 | base（无适配器） | alpaca-SFT |
|---|---|---|
| 1. 标题+三点介绍春联 | 会话跑偏（英文内容、自定义格式，输出里**自己打出 "User:" 标签**——完全不懂对话格式） | 中文回答 ✓ 但**列表格式未学会**（还是段落）且事实错误（"春联是节日"） |
| 2. 一句话解释微调 | 跑偏 | 中文流畅解释 ✓（泛化但结构对） |
| 3. 列三条英语学习建议 | 跑偏 | **编号列表成功** ✓✓（alpaca 格式信号生效） |
| 4. 你是谁 | 无关内容 | "人工智能技术人才…由专家和人才开发"——identity 痕迹被 1000:91 稀释 |

**对"变化不大"预测的裁决**：一半对一半。**知识/事实层变化有限**（春联仍然错）✓ 用户对；但**格式层变化巨大**（base 不懂对话格式 → SFT 学会编号列表，loss 2.13→1.19 的下降几乎全是格式/分布信号）。这就是为何 SFT 被称为"行为克隆/格式教学"而非"知识注入"——与 Task5 板书（SFT = 行为克隆）互相印证。同时：**混数据稀释**（identity 只占 8%），人设被 alpaca 冲刷——混比例是下一个可做实验。
---

## 14. 人设 vs 知识混比扫描（identity_x%:alpaca, OLMoE, 2026-09-05）

**设计**：固定总预算 ≈900 条 / 1 epoch / 同超参；唯一变量 = identity 占比（10% / 30% / 50%，30%+ 档 identity 91 条循环重复填充）。评价 = 四问实测（你是谁 / 谁开发 / 三条建议 / 解释微调）。

**训练指标**：

| 组 | train_loss | eval_loss | peak | 耗时 |
|---|---|---|---|---|
| mix10 (91/800) | 1.2425 | 1.3408 | 10374 MiB | 12:07 |
| mix30 (273/640) | 1.2862 | 1.2578 | 10368 MiB | 12:16 |
| mix50 (455/455) | 1.3145 | 1.2063 | 10374 MiB | 12:21 |

**四问实测评分**（evidence/chat_mix*.txt，完整转录可查）：

| 提问 | mix10 (10%) | mix30 (30%) | mix50 (50%) |
|---|---|---|---|
| 你是谁 | ✗ "我是李晨"（纯幻觉） | ⚠️ "人工智能 assistant"（有影子，无小智） | ✓✓ "我是 **小智**，由 小智 团队开发的智能助手" |
| 谁开发 | ✗ "由百度开发"（但**句式对了**） | ⚠️ "研发团队开发" | ✓✓ "小智 是由 小智 团队开发" |
| 三条学习建议 | ✓ "1. 语言的基础…"（编号在，内容薄） | ✓✓ "1. 学习英语的基本词汇和词组…" **最佳** | ✗ 空输出 |
| 解释微调 | ✓✓ **最佳**（完整通顺） | ✓✓ 完整通顺 | ✗ 空输出 |

**结论**：
1. **trade-off 存在且有一个"甜蜜点"**：identity 30% 时人设影子 + 知识/格式全保留（四问全 ✓）；50% 时小智高保真但**知识任务直接崩**（后两问空输出）——identity 从 30%→50% 的边际收益是负的；
2. **重复样本的代价**：50% 档 identity 每条重复 5 遍 → 模型把人设语料"背死"了（作者栏泛化成"小智团队"），同时挤占 alpaca 到 455 条 → 格式/知识没学够；
3. **eval_loss 陷阱（方法论亮点）**：eval_loss 随 identity 占比**单调下降**（1.34→1.26→1.21），若只看曲线会得出"混比越高越好"——但 val 集是从混比数据切出来的，identity 重复样本占比同步升高，模型背题了。**横比 eval_loss 被数据构成污染，四问实测才是真相**——与"训练集外评估才可信"的教科书原则呼应。
4. **句式迁移**：即使 10% 档答错内容（"由百度开发"），"我是X，由Y开发"的句式已学会——格式/分布学习先于事实学习（§13 结论的再确认）。
