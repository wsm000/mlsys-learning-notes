# DataFlow Task 1 实验记录（1.1 至 1.2.4）

## 完成情况

| 教程部分 | 实际执行与验收 |
|---|---|
| 1.1 PDF→模型 | 2 页示例 PDF→10 个文本块→真实模型生成 20 组候选 QA→原文匹配筛选 7 组→人工审核保留 5 组→4 组训练、1 组留出→48 步 GPU LoRA 微调→保存及重载验证→训练前后提问 |
| 1.2.1 仓库与文档 | 核对 GitHub、官方文档及飞书教程 |
| 1.2.2 安装 | Python 3.10.12，open-dataflow 1.0.11，独立虚拟环境复用系统基础依赖；未修改系统包 |
| 1.2.3 初始化 | `dataflow init` 成功生成工作区和示例数据；`dataflow pdf2model init --qa kbc --model ...` 成功生成配置 |
| 1.2.4 CPU pipeline | 运行官方 `cpu_pipelines/reasoning_pipeline.py`；3 步输出为 [44, 42, 42]，均非空 |

## 与教程默认环境的差异

这是一套实际执行的本地资源适配实验，并非原版一键命令的逐项复现。

- 环境使用 virtualenv，替代 conda；DataFlow 安装在 `pipeline-env`，基础计算库从现有环境读取。
- PDF 使用初始化自带的 `bitter_lesson.pdf`（Richard Sutton, 2019），经检查含可直接提取文本。以 PyMuPDF 替代 MinerU/OCR，未下载 MinerU 模型。
- QA 合成使用真实 DataFlow `FileStorage`、`PromptedGenerator` 与 Ollama `qwen3:0.6b`，无模拟回答；关闭思考模式并限制输出长度。
- 基座为已有 Qwen3.5-2B；以 Transformers + PEFT 执行真实 LoRA，替代模板的 Qwen2.5-7B、多卡 vLLM 与 LlamaFactory 训练后端。
- 原版 `dataflow pdf2model train`、`dataflow chat` 未作为成功执行记录；对应实操分别由 `train_pdf_lora.py`、`chat_pdf_lora.py` 完成。

## 训练与验证结果

- 训练样本：4；留出样本：1；固定随机种子：42。
- LoRA rank=8、alpha=16，目标 q_proj/v_proj；学习率 0.0001；共 48 步。
- 可训练参数：417,792 / 总参数：2,213,659,456。
- 训练集回答 token 平均损失：3.665230 → 1.824853。
- 留出集回答 token 平均损失：4.678292 → 4.097848。
- 独立进程从磁盘重载 adapter 后，重复评估留出损失：4.097848。全部保存张量与实际加载权重逐项相同，最大差值为 0。
- 复核过程峰值 GPU 显存：4.27 GiB；复核耗时（含权重加载和评估）：8.9 秒。
- 原训练进程的“第二命名 adapter”损失一致性断言未通过；训练和保存已完成。随后用独立进程重新加载已保存文件，核对全部权重并重复评估，使用该复核结果作为最终证据。未将失败断言报告为成功。

数据量极小，留出集只有一条；本实验验证 PDF→数据→真实微调→模型验证链路。损失变化和个例回答不能证明泛化能力提升，也不应据此宣称显著性能提升。

## 前后回答

### train 示例

问题：How did the research progress in computer Go compare to computer chess?

微调前：

The research progress in computer Go and computer chess represents two distinct trajectories in artificial intelligence, shaped by different problem spaces, data availability, and the nature of the games themselves. While both fields have seen massive leaps in performance, they have reached different "plateaus" at different times due to these fundamental differences.

### 1. The Nature of the Games and Data
The most significant factor driving the

微调后：

The research progress in computer Go compared to computer chess was much slower.


### heldout 示例

问题：How should the search be conducted?

微调前：

Since your question is very general, the best way to answer is to break it down into the most common scenarios where "search" is used. Here are the standard approaches for different contexts:

### 1. **Digital Search (Browsing the Web)**
If you are looking for information on the internet:
*   **Use a Search Engine**: Start with a major engine like **Google

微调后：

Searches should be conducted systematically, ensuring that all relevant criteria are met and that the results are accurate and reliable.


## 在 vm-60 上复现

工作目录：`~/dataflow-task1`。

```bash
# 1.2.4：官方 CPU 示例
cd ~/dataflow-task1/workspace/cpu_pipelines
../../pipeline-env/bin/python reasoning_pipeline.py

# 1.1：PDF 文本提取与真实 QA 合成
cd ~/dataflow-task1/pdf2model
../pipeline-env/bin/python pdf_to_model_pipeline.py

# 真实 GPU 微调、前后对比及 adapter 重载验证
python3 train_pdf_lora.py

# 与已微调的模型提问
python3 chat_pdf_lora.py "What is the main lesson from the bitter lesson?"

# 完整结果检查
cd ~/dataflow-task1
pipeline-env/bin/python verify_task1.py
```

CPU 输出位于 `workspace/cpu_pipelines/cache_local/`；PDF 数据、训练日志和 adapter 位于 `pdf2model/`。脚本固定使用本机已有模型路径，模型权重不包含在打卡压缩包中。

## 材料索引

- `verification.json`：完整验收结果。
- `cpu-run.log`、`init.log`、`pdf-init.log`：初始化与 CPU 实验日志。
- `pdf2model/preparation_summary.json`、`qa_evidence.json`、`manual_review.json`：数据来源和审核记录。
- `pdf2model/train.json`、`eval.json`：互不重叠的数据划分。
- `pdf2model/training.log`、`training_summary.json`、`training_losses.json`、`comparison.json`：训练与前后对比证据。
- `pdf2model/adapter/`：已保存且通过重载验证的 LoRA 权重。

教程：[飞书](https://wcny4qa9krto.feishu.cn/wiki/I9tbw2qnBi0lEakmmAGclTysnFd)；[仓库](https://github.com/OpenDCAI/DataFlow)；[官方安装文档](https://opendcai.github.io/DataFlow-Doc/zh/guide/install/)。
