# Task 3 · KV Cache 与模型属性 —— 显存与规模的权衡(打卡笔记)

> 用途:复习速查 + 直接作为 issue #133 的打卡底稿(填好昵称/日期/实验输出后即可贴)。
> 单位约定:1 KiB=1024 B,1 MiB=1024 KiB,1 GiB=1024 MiB(1 GiB≈1.07 GB,显卡标称是十进制 GB,注意口径)。

## 1. 三条核心公式

1) **KV Cache 显存(字节)**
```
KV = 2(K&V) × L(层数) × b(batch) × s(上下文长度) × h_kv(KV头数) × d_head(头维) × p(每元素字节)
```
- MHA 时 h_kv·d_head = d_model,可简写成 `2 × L × b × s × d_model × p`
- p:fp32=4,fp16/bf16=2,fp8/int8=1,int4=0.5
- MQA:h_kv=1;GQA:h_kv < h;省的倍数 = h / h_kv

2) **权重显存 = 总参数量 × p**,参数量(解码器 + SwiGLU + 不共享 embedding):
```
P ≈ L × [ 2·d² + 2·d·(h_kv·d_head) + 3·d·d_ff ] + 2·V·d
```
- 粗估口诀(标准 GPT 架构):`P ≈ 12·L·d² + V·d`(tie 词表)或 `+ 2·V·d`(不 tie)
- tie 时词嵌入与 lm_head 共享,只算一份 V·d

3) **推理总显存 ≈ (权重 + KV Cache) × 1.15**(激活 + 框架开销)
- decode 每步速度上限 ≈ 显存带宽 / (权重字节 + 每步要读的 KV 字节)

## 2. 常见模型速查表(fp16)

| 模型 | L | d | h | h_kv | d_head | d_ff | V | 参数量(估算) | KV/token/序列 | 权重显存 |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-2 | 12 | 768 | 12 | 12 | 64 | 3072 | 50257 | ≈1.24 亿(tie) | 36 KiB | 0.23 GiB |
| LLaMA-2-7B | 32 | 4096 | 32 | 32(MHA) | 128 | 11008 | 32000 | 6.74B | 512 KiB | 12.55 GiB |
| LLaMA-2-13B | 40 | 5120 | 40 | 40(MHA) | 128 | 13824 | 32000 | 13.0B | 800 KiB | 24.24 GiB |
| LLaMA-2-70B | 80 | 8192 | 64 | 8(GQA) | 128 | 28672 | 32000 | 69.0B | 320 KiB | 128.5 GiB |
| Mistral-7B | 32 | 4096 | 32 | 8(GQA) | 128 | 14336 | 32000 | 7.24B | 128 KiB | 13.5 GiB |
| Qwen2-7B | 28 | 3584 | 28 | 4(GQA) | 128 | 18944 | 151936 | 7.6B | 56 KiB | 14.2 GiB |

(LLaMA-3-8B 与 Mistral 同参数量级:32 层、h_kv=8 → KV/token 同为 128 KiB)

## 3. 三道例题(带解)

1. LLaMA-2-7B fp16,b=1,s=2048:KV = 2×32×4096×2×2048 B = **1.0 GiB**;权重 12.55 GiB;总 ≈ (12.55+1)×1.15 ≈ **15.6 GiB** → 16GB 卡勉强,建议 24GB。
2. LLaMA-2-70B 的 GQA:KV/token 从 MHA 等价的 2.5 MiB → 0.31 MiB(÷8);4K 上下文单序列 KV:1.25 GiB → 0.16 GiB。
3. A100-80G 跑 70B int8(权重 64.3 GiB):KV 预算 ≈ 80−64.3−4 ≈ 11.7 GiB → 11.7 GiB / 320 KiB ≈ 3.85 万个 token 槽;平均 4K/请求 → **约 9 路并发**。

## 4. 为什么需要 KV Cache(3 句话)
1. 自回归逐 token 生成;因果注意力下,历史 token 的 K/V 每步都不变。
2. 不缓存 → 每步重算全部历史 K/V,总计算量 O(T²);缓存 → 每步只算新 token 的 q/k/v,总量 O(T)。
3. 本质是**用显存换时间**:KV ∝ L·b·s,长上下文与大并发的显存大头。

## 5. 权衡(本任务主题)
- 权重显存是**常数**(与 b、s 无关);KV 随 b·s **线性增长** → 模型规模、精度、上下文、并发四者在显存里此消彼长。
- batch↑:权重读取被摊薄(算术强度↑)吞吐↑;但 KV 同步涨,占满显存 → 容量封顶。
- decode 是**访存受限**:A100(2.0 TB/s)跑 fp16 7B 单流理论上限 ≈ 2039/13.48 ≈ 151 tok/s。
- 训练 vs 推理:全参训练每参数 ≈ 16 B(fp16 权重+梯度+Adam 两状态+fp32 master),推理只要 2 B + KV。

## 6. 优化技术图谱
| 技术 | 省什么 | 效果 | 代表 |
|---|---|---|---|
| 权重量化 int8/int4 | 权重 | ÷2 / ÷4 | GPTQ、AWQ、llama.cpp |
| KV 量化 | KV | ÷2~÷4 | fp8 KV、KIVI(K 8bit/V 低比特) |
| MQA(KV 头=1) | KV | ÷h(略掉点) | PaLM、Falcon |
| GQA(KV 头=h_kv) | KV | ÷(h/h_kv),质量≈MHA | LLaMA-2-70B、LLaMA-3、Mistral、Qwen2 |
| MLA(低秩压缩) | KV | vs MHA ≈ ÷17 | DeepSeek-V2/V3 |
| PagedAttention | 碎片/预留浪费 | 利用率↑,支持前缀共享 | vLLM |
| 滑动窗口 | KV 上限 | 恒定 W×L | Mistral、Longformer |
| 跨层共享 KV | KV | 约省一大部分 | YOCO |
| FlashAttention | 激活显存 | prefill O(s²)→O(s) | 通用(不省 KV) |
| Prefix/Context Cache | 重复前缀 | 复用 KV | vLLM、各家 context caching |

## 7. 面试快问快答
1. KV Cache 存什么?为什么存 K/V 不存 Q?——K/V 属于历史 token 且被反复使用;Q 只来自当前 token,用完即弃。
2. 7B fp16 推理要多少显存?——权重 12.5 GiB + KV(2K 约 1 GiB)×1.15 ≈ 15~16 GiB。
3. prefill vs decode?——prefill 计算密集(TTFT),decode 访存密集(TPOT)。
4. MQA/GQA/MLA 区别?——MQA 全头共享 1 组 KV(最省、略掉点);GQA 分组共享(质量≈MHA,主流);MLA 低秩潜压缩(最激进,DeepSeek)。
5. PagedAttention 解决什么?——传统按最大长度连续预留导致碎片/浪费(内部碎片+外部碎片),分页管理后按需分配、可共享前缀,思路同 OS 虚存。
6. batch 加大为何吞吐先升后饱和?——先摊薄权重读取;后受 KV 容量与每步读 KV 的带宽限制。
7. KV 量化为什么 K 比 V 敏感?——attention 分布由 QK 点积决定,K 精度直接影响 softmax;V 是加权和的内容,更宽容(KIVI:K 分通道 8bit、V 低比特)。
8. 128K 上下文显存怎么办?——KV 线性于 s:GQA/MLA、KV 量化、滑窗/稀疏注意力、PagedAttention + 卸载。

## 8. 实验脚本 verify_memory.py
```python
# pip install transformers  (本脚本不需要下载权重)
def params(d, L, h, h_kv, d_head, d_ff, V, tie=False):
    attn = 2*d*d + 2*d*(h_kv*d_head)      # Q,O + K,V(GQA 时 K/V 矩阵更瘦)
    mlp  = 3*d*d_ff                       # SwiGLU: gate/up/down
    emb  = (1 if tie else 2)*V*d          # embedding(+lm_head)
    return L*(attn+mlp) + emb

def kv_bytes(L, h_kv, d_head, b, s, p=2):
    return 2 * L * b * s * h_kv * d_head * p

MODELS = {
    "LLaMA-2-7B":  dict(d=4096, L=32, h=32, h_kv=32, d_head=128, d_ff=11008, V=32000),
    "LLaMA-2-13B": dict(d=5120, L=40, h=40, h_kv=40, d_head=128, d_ff=13824, V=32000),
    "LLaMA-2-70B": dict(d=8192, L=80, h=64, h_kv=8,  d_head=128, d_ff=28672, V=32000),
    "Mistral-7B":  dict(d=4096, L=32, h=32, h_kv=8,  d_head=128, d_ff=14336, V=32000),
    "Qwen2-7B":    dict(d=3584, L=28, h=28, h_kv=4,  d_head=128, d_ff=18944, V=151936),
}
if __name__ == "__main__":
    GIB, MIB = 2**30, 2**20
    for name, c in MODELS.items():
        P = params(**c)
        print(f"{name:12s} params≈{P/1e9:6.2f}B  weights={P*2/GIB:7.2f}GiB  "
              f"KV/token={kv_bytes(c['L'], c['h_kv'], c['d_head'], 1, 1)/MIB:7.3f}MiB")
    print("LLaMA-2-7B KV(b=1,s=2048) =", kv_bytes(32, 32, 128, 1, 2048)/GIB, "GiB")
    print("Qwen2-7B  KV(b=1,s=131072)=", kv_bytes(28, 4, 128, 1, 131072)/GIB, "GiB")
```
预期输出:params≈6.74/13.02/68.98/7.24/7.61 B;KV/token = 0.5/0.78/0.31/0.125/0.055 MiB;两行 KV = 1.0 GiB / 7.0 GiB。
进阶:用 `AutoConfig.from_pretrained("Qwen/Qwen2-7B")` 核对 num_hidden_layers/num_key_value_heads/hidden_size,再与 HF 模型卡公布的总参数量对齐(差值是 norm/bias)。

## 9. 打卡模板(复制到 issue #133 评论)
```markdown
## Task3 打卡:KV Cache 与模型属性——显存与规模的权衡
**日期/昵称**:

### 一、核心公式
- KV Cache = 2 × L × b × s × h_kv × d_head × p;MHA 可写 2 × L × b × s × d × p
- 权重显存 = 参数量 × p;P ≈ L·[2d² + 2d·(h_kv·d_head) + 3d·d_ff] + 2V·d
- 推理总显存 ≈ (权重 + KV) × 1.15

### 二、手算练习
1. LLaMA-2-7B fp16, b=1, s=2048:KV = 2×32×4096×2×2048 B = 1.0 GiB;权重 12.55 GiB;总 ≈ 15.6 GiB
2. LLaMA-2-70B 用 GQA(h_kv=8):KV/token 由 2.5 MiB → 0.31 MiB,省 8 倍
3. Qwen2-7B fp16 单流 128K:KV ≈ 56 KiB × 131072 ≈ 7.0 GiB(可跑在 24GB 卡上)

### 三、学习要点(自己的话)
1. KV cache 用显存换计算,把生成的注意力计算从 O(T²) 降到 O(T)
2. 推理显存 = 静态权重 + 动态 KV;长上下文和大 batch 的瓶颈都是 KV
3. decode 访存受限 → 量化权重/增大 batch/投机解码都在围绕带宽优化
4. GQA 几乎无损地把 KV 显存 ÷(h/h_kv),已是当代标配;PagedAttention 管碎片,KV 量化管总量,互补

### 四、实验输出
(贴 verify_memory.py 运行结果截图/文本)

### 五、自测题(附答案见讲义)
T1: LLaMA-2-13B fp16, b=4, s=4096 的 KV 是多少?
T2: 为什么 GQA 几乎不掉点?
T3: 24GB 卡上 Qwen2-7B、8K 上下文,理论最多并发几路?
T4: MQA 为什么把 KV 显存省为 1/h?
T5: b=1 的 decode 为什么慢?怎么提速?

### 参考
- GQA: Training Generalized Multi-Query Transformer (Google, 2023)
- Fast Transformer Decoding: One Write-Head Is All You Need (MQA, Shazeer 2019)
- Efficient Memory Management for LLM Serving with PagedAttention (SOSP'23, vLLM)
- FlashAttention (2022); KIVI (KV 量化); DeepSeek-V2 (MLA)
```
