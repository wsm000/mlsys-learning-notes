#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""加分挑战: 预分配 KV buffer + 原地写入 (vLLM 静态 KV cache 的最小可读版)

与 molecular_gpt_KVCache.py 的区别:
  cat 版:  每步 torch.cat([past_k, k], dim=2) —— 把全部历史 KV 重新拷贝一遍, 每步拷贝量 O(T)
  buf 版:  每层预分配 k_buf/v_buf (B, nh, max_len, hs), 每步只写入新 token 的 1 个位置,
           注意力用切片视图 k_buf[:, :, :total] 读, 零拷贝。每步写量 O(1)。
正确性: buffer 中 offset 之后是上一次生成的残留, 但每步"先写后读 [:total]",
        读到的永远是本次生成自己写过的位置, 残留不可见。
"""
import math
import random
import numpy as np
import torch
import torch.nn as nn

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

SPECIAL_TOKENS = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


def decode_tokens(indices, idx2char):
    return ''.join(idx2char.get(i, '') for i in indices
                   if i not in (PAD_IDX, BOS_IDX, EOS_IDX))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, offset=0):
        return x + self.pe[:, offset:offset + x.size(1), :]


class CausalSelfAttention(nn.Module):
    """KV buffer 版注意力: 预分配 + 原地写入 + 切片读取"""

    def __init__(self, d_model, n_heads, max_len, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_len = max_len
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # 惰性分配: 构造时不知道 batch 大小
        self.k_buf = None
        self.v_buf = None

    def _ensure_buf(self, B, device, dtype):
        if self.k_buf is None or self.k_buf.size(0) < B:
            self.k_buf = torch.zeros(B, self.n_heads, self.max_len, self.head_dim,
                                     device=device, dtype=dtype)
            self.v_buf = torch.zeros_like(self.k_buf)

    def forward(self, x, kv_offset):
        """x: (B, T, C); kv_offset: 本段写入 buffer 的起点 = 已生成 token 数"""
        B, T, C = x.shape
        self._ensure_buf(B, x.device, x.dtype)
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 原地写入: 只写本段 T 个位置 (decode 时 T=1)
        self.k_buf[:, :, kv_offset:kv_offset + T] = k
        self.v_buf[:, :, kv_offset:kv_offset + T] = v
        total = kv_offset + T

        k_full = self.k_buf[:, :, :total]   # 切片视图, 无拷贝
        v_full = self.v_buf[:, :, :total]

        att = (q @ k_full.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if T > 1:  # prefill 需要因果掩码; decode T=1 天然只看历史
            mask = torch.tril(torch.ones(T, total, device=x.device, dtype=torch.bool))
            att = att.masked_fill(~mask, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        out = (att @ v_full).transpose(1, 2).contiguous().reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, max_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, kv_offset):
        x = x + self.attn(self.ln1(x), kv_offset)
        x = x + self.ff(self.ln2(x))
        return x


class MolecularGPT(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, ff_dim=2048,
                 max_len=256, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, max_len, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.token_embed.weight = self.head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, kv_offset=0):
        """x: (B, T) token ids; kv_offset: 已生成 token 数 (写入位置 + 位置编码偏移)"""
        x = self.token_embed(x)
        x = self.pos_enc(x, offset=kv_offset)
        for block in self.blocks:
            x = block(x, kv_offset)
        x = self.ln_f(x)
        return self.head(x)


def _sample(logits_last, temperature):
    if temperature != 1.0:
        logits_last = logits_last / temperature
    probs = torch.softmax(logits_last, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def generate(model, start_fragment, char2idx, idx2char, device,
             max_new_tokens=100, temperature=1.0):
    """两阶段生成, KV 全程驻留 buffer, 零拷贝。"""
    model.eval()
    tokens = [BOS_IDX] + [char2idx.get(c, UNK_IDX) for c in start_fragment]
    ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    generated = tokens[:]

    with torch.no_grad():
        # prefill: 整段起始片段一次写入 buffer
        logits = model(ids, kv_offset=0)
        offset = ids.size(1)
        next_token = _sample(logits[:, -1, :], temperature)
        generated.append(next_token)

        for _ in range(max_new_tokens - 1):
            if next_token == EOS_IDX:
                break
            last = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
            logits = model(last, kv_offset=offset)   # 写入 1 个位置, 读 total 个
            offset += 1
            next_token = _sample(logits[:, -1, :], temperature)
            generated.append(next_token)

    return decode_tokens(generated[1:], idx2char)


if __name__ == '__main__':
    # 独立演示: 随机权重直接跑通形状与长度
    model = MolecularGPT(vocab_size=40, d_model=128, n_heads=4,
                         n_layers=2, ff_dim=512, max_len=512, dropout=0.1).eval()
    char2idx = {c: i for i, c in enumerate('abcdefg')}
    smi = generate(model, 'c1c', char2idx, {v: k for k, v in char2idx.items()},
                   torch.device('cpu'), max_new_tokens=50)
    print('demo generated len:', len(smi))
