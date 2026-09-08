# DIY-LLM Task 3：语言模型架构与超参数学习笔记

## 1. 学习主线

本笔记对应 Stanford CS336 第 3 讲《Architectures, Hyperparameters》与 Datawhale《DIY-LLM》第 4 章《语言模型架构和训练的技术细节》。这一章回答一个从零写 LLM 绕不开的问题：架构每个零件长什么样、为什么这么设计、超参数到底怎么定。

参考资料：

- 参考笔记：https://github.com/zizhengwang2026/0-1LLM-Stanford-CS336-Language-Modeling-from-Scratch-/blob/main/notes/03-architecture-and-hyperparameters.md
- CS336 课程主页：https://stanford-cs336.github.io/
- Datawhale DIY-LLM：https://github.com/datawhalechina/diy-llm

一条统一的学习链：

    2017 标准 Transformer 五大件
      -> 现代模型的逐项改造（每处改动都有理由）
      -> 注意力变体（一条“压 KV Cache”的主线）
      -> 超参数经验法则（真正需要拍板的只有几个数）
      -> 训练稳定性三板斧（按住两个爱炸的 softmax）

与已有笔记的分工：RoPE 数学细节、MHA/GQA/KV Cache 的计算链路见 `task2_rope_attention_learning_notes.md`；LLaMA Block 组装代码、MoE Router 见 `学习笔记_模型结构_05-08_LLaMA_Block与MoE.md`。本篇是总纲视角：把零件放回整机，补齐“为什么是它们”。

## 2. 标准 Transformer：五大件基线（2017 原始论文）

自注意力是 Transformer 唯一的核心创新：抛弃 RNN/CNN，序列可完全并行，且任意两 token 一步直连，天然捕捉长距离依赖。围绕它有五个必备零件。

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

| 零件 | 2017 年原始设计 | 解决什么问题 |
|---|---|---|
| 自注意力 | scaled dot-product，多头 | 序列内任意位置交互，可并行 |
| 位置编码 | 正余弦绝对位置编码，加到词向量上 | 自注意力本身排列不变 |
| 归一化 | LayerNorm，放在残差之后（Post-Norm） | 稳定激活分布，加速收敛 |
| 残差连接 | Output = Input + Sublayer(Input) | 深层梯度高速公路 |
| FFN | 两层 MLP，ReLU，d_ff = 4 x d_model | 逐位置非线性变换 |

三个必须能自己讲出来的“为什么”：

1. 为什么除以 sqrt(d_k)？点积 Q·K 的方差随维度 d_k 线性放大，不缩放则 logits 过大，softmax 进入梯度极小的饱和区。除一下把方差拉回稳定范围。
2. 为什么要多头？每个头在 d_model/h 的低维子空间里关注不同模式，最后拼接。每头维度变小，总 FLOPs 和单头相近，但表达能力更高——本质是低秩分解换多样性，不是靠堆算力。
3. 为什么必须有位置编码？“我爱你”和“你爱我”在自注意力眼里是同一个集合（排列不变），必须显式注入位置信息。

## 3. 现代主流改法：每一刀都有理由

### 3.1 Post-Norm 改 Pre-Norm

原始顺序是 子层 -> 残差 -> LayerNorm；现代改成 LayerNorm -> 子层 -> 残差。

核心收益是保持残差流干净：恒等通路从底层直达顶层不被归一化打断，梯度无损穿行。换来的是训练更稳、不依赖精细的 warmup、能训百层以上深网络。GPT-3、PaLM 之后 Pre-Norm 已是默认。

### 3.2 LayerNorm 改 RMSNorm

    LayerNorm: y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
    RMSNorm:  y = x / sqrt(mean(x^2) + eps) * gamma

RMSNorm 砍掉减均值这一步和 beta 参数，只保留除以均方根。表达能力几乎无损，但少一次统计量计算、少读一份参数，速度更快。LLaMA、PaLM、Chinchilla、T5 后期版本都用它。

### 3.3 去掉所有 bias

现代实现几乎删光线性层偏置。原因不只是省显存：bias 相关运算算术强度低、内存搬运贵，而且实证上去掉后训练反而更稳。它已经从“省显存的优化”升级成“默认稳定性保障”。

### 3.4 ReLU 改 GeLU，再改 SwiGLU

演化路线：ReLU（原点不可微）-> GeLU（更平滑，GPT 系用过）-> GLU 门控家族（2023 后绝对主流）。

SwiGLU 把 FFN 的升维矩阵拆成两路：内容通道 up 和门控通道 gate，门控用 SiLU 生成逐元素开关：

    SwiGLU(x) = down( silu(gate(x)) * up(x) )

直觉是一扇智能百叶窗：每个神经元通过多少由输入动态决定，而不是像 ReLU 那样一刀切。

参数量核算（呼应 Task 2）：普通 FFN 是 2 个 d x d_ff 矩阵；SwiGLU 是 3 个（gate/up/down）。若仍取 d_ff = 4d，参数会膨胀约 1.33 倍，所以主流把 d_ff 降到约 8/3 d（即 2.66 d，常向上取整到 64/128 的倍数），使两种激活下 FFN 参数量都约为 8 d^2，可以公平比较。

### 3.5 绝对位置编码改 RoPE

RoPE 给每个位置的 Q/K 乘一个旋转矩阵，旋转角度随位置变化；做 Q·K^T 时利用旋转矩阵的性质，点积结果只依赖相对位移 m-n，于是注意力自然获得相对位置信息。零额外参数、显式相对位置、可外推。数学推导与实现见已有的 RoPE 笔记，这里只需记住定位：RoPE 契合注意力的本质——关心关系而非绝对坐标，因此一统江湖。

## 4. 注意力变体：一条“压 KV Cache”的主线

自回归推理每生成一个 token 都要读取全部历史的 K/V，算术强度极低，GPU 在等数据——这是典型的 memory-bound 场景（Task 2 的 Roofline 分析直接适用）。所以变体的共同目标是：把要读的东西变小。

| 方案 | 做法 | 代价/收益 |
|---|---|---|
| MHA | 每个 Q 头配独立 K/V 头 | 基线 |
| MQA | 所有 Q 头共享一份 K/V | 显存最省，表达力下降明显 |
| GQA | Q 头分组，每组共享一份 K/V | 折中；LLaMA 2/3、Qwen 采用 |
| MLA | K/V 低秩联合压缩进潜在空间 | KV Cache 减 90%+，DeepSeek 系列 |
| 滑窗/稀疏 | 只看局部窗口 + 少量全局层 | 控制长文本成本；全局+局部交替混合是 2024-2025 流行范式 |

关键认知：GQA 的价值不只是省显存。推理时瓶颈在读 KV Cache，减少 K/V 头数直接减少了要搬运的数据量，吞吐量大幅提升。MQA 到 GQA 到 MLA 是在同一条“压缩 KV、保住表达力”的坐标轴上找不同的工作点。

## 5. 超参数经验法则

从零训模型时真正需要拍板的只有少数几个数，业界已有共识：

| 超参数 | 经验法则 | 备注 |
|---|---|---|
| FFN 中间维度 d_ff | ReLU 用 4 x d_model；GLU 类用约 2.66 x d_model | 实验表明 1~10 倍区间都接近最优，不必死守 |
| 头数与头维 | NumHeads x HeadDim 约等于 d_model | 主流比例约 1；T5 是著名例外 |
| 宽深比 width/depth | 约 100 时损失最小 | 但受系统工程约束修正 |
| 词表大小 | 早期 3 万~5 万；现代多语言生产模型 10 万~20 万 | 大词表让低资源语言更省 token |
| Dropout | 预训练不用（单轮训练几乎不过拟合） | 微调小数据场景才回归 |
| 权重衰减 | 仍普遍使用 | 作用被重新理解，见下 |

宽深比的工程约束值得单独记：越深越依赖流水线并行（有气泡、调度复杂，工程师不爱）；越宽越适合张量并行（切大矩阵、通信规整好实现）。所以实际模型大多适度控深、保留宽度，是表达能力、并行效率、工程复杂度三方的妥协，而不是纯精度最优。

权重衰减的真实作用：预训练只有一轮、数据巨大，几乎不存在过拟合，所以它的经典解释失效。实际观测到的作用是训练末期的优化加速器——当学习率趋零时，权重衰减与优化动态相互作用产生隐性加速，帮模型收敛到更优解。

## 6. 训练稳定性：按住两个 softmax

Transformer 里最容易数值爆炸的位置是两个 softmax：输出层的和注意力里的。它们都包含指数和除法，logits 一大就溢出或进入饱和区。模型越大、训练越久，越需要专门手段。

| 技巧 | 机制 | 使用者 |
|---|---|---|
| z-loss | 损失中加 lambda x log(Z)^2，把 softmax 配分函数 Z 压在合理区间 | PaLM 提出；Baichuan2、DCLM、OLMo 2 跟进 |
| QK 归一化 | softmax 前对 Q、K 各做一次 RMSNorm/LayerNorm，直接控住数值范围 | Gemma 2、DCLM、OLMo 2 |
| 软截断 soft-capping | 内积后用 cap x tanh(logits / cap) 把 logits 软夹到 (-cap, +cap) | Gemma 2、OLMo 2 |

三者思路一致：不让进入指数的数失控。区别只在作用位置——z-loss 管输出层 softmax，QK 归一化管注意力 softmax（顺带整体稳了深层网络），soft-capping 是更局部的紧急刹车但尚未成为主流默认。

## 7. 组装验证：参数量近似公式

把本章结论拼成一个可核算的整体（延续 Task 2 的资源账）：

    每层参数量（不含嵌入）
      = 注意力投影 4 d^2（Q/K/V/O 四个 d x d）
      + FFN 参数 8 d^2（ReLU: 2 x d x 4d；SwiGLU: 3 x d x (8/3)d）
      大约 equals 12 d^2

    非嵌入总参数 approximately equals 12 x d_model^2 x L

验算 LLaMA-2-7B：d_model = 4096，L = 32，

    12 x 4096^2 x 32 = 6.44e9

与官方约 6.5B 非嵌入参数吻合。这说明两点：其一，主流选择确实收敛到了 12 d^2 这条线上；其二，看到任何新架构公告，先用这个公式估个量级再读细节。

## 8. 自测清单（合上笔记回答）

1. 除以 sqrt(d_k) 在修什么数值问题？
2. 多头注意力为什么“几乎不加算力却更有表达力”？
3. Pre-Norm 比 Post-Norm 稳的根本原因是什么？（提示：残差流）
4. 写出 RMSNorm 公式，说明比 LayerNorm 少了什么、省在哪。
5. SwiGLU 有三个投影矩阵，为什么 d_ff 要降到约 8/3 d？
6. RoPE 让注意力分数只依赖什么的数学性质？
7. 从算术强度角度解释为什么 GQA 能提升推理吞吐而不仅是省显存。
8. 宽深比约 100 最优，为什么实际模型不无限加深？
9. 预训练不用 dropout，为什么还用权重衰减？它的真实作用是什么？
10. 两个爱炸的 softmax 分别对应哪些稳定技巧？

关键校正（自测中容易答错的点）：

- “多头计算量大很多”——错。每头 d/h 维，总 FLOPs 与单头相近。
- “去掉 bias 只是省显存”——不完整。实证上去掉后训练更稳，是稳定性保障。
- “GQA 只是为了省显存”——不准确。推理是 memory-bound，少读 KV 才是吞吐提升的主因。
- “权重衰减防过拟合”——预训练语境下不成立，真实作用接近训练末期的优化加速器。
- “d_ff 用 SwiGLU 也设 4 倍”——参数量会比同预算 ReLU 版多约 1.33 倍，失去可比性。

## 9. 一页速查表

    Attention        softmax(Q K^T / sqrt(d_head)) V
    Pre-Norm         x = x + Norm(Sublayer(x))
    RMSNorm          x / sqrt(mean(x^2) + eps) * gamma
    SwiGLU           down(silu(gate(x)) * up(x))
    z-loss           loss += lambda * log(Z)^2
    soft-cap         cap * tanh(logits / cap)
    层参数量         approximately 12 d_model^2（注意力 4d^2 + FFN 8d^2）
    非嵌入总量       approximately 12 d_model^2 L

经验法则速查：

    d_ff             4x（ReLU）/ 约 2.66x（GLU 族）
    头数             NumHeads x HeadDim 约等于 d_model
    宽深比           约 100，深度受流水线并行制约
    词表             现代多语言模型 10 万~20 万
    正则             预训练无 dropout，权重衰减照常

## 10. 一句话总结

现代 LLM 架构 = 2017 五大件骨架 + 四刀改造（Pre-Norm/RMSNorm 保残差流、去 bias、SwiGLU 门控、RoPE 相对位置），注意力变体沿“压 KV Cache 提吞吐”一条线演进，超参数按 d_ff、头数、宽深比、词表的共识经验法则拍板，再用 z-loss/QK 归一化/软截断按住两个 softmax——每一步都能说出“为什么”，才是真的懂架构。
