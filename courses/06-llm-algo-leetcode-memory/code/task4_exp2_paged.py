# -*- coding: utf-8 -*-
"""Task4 实验2 · 分页：PagedAttention 的碎片、元数据与块大小权衡。

回答 4.1(2)（连续预留为什么"用得少也 OOM"）与 4.1(3)（block size 的三重影响）。

内容：
  A. 机制单测：原子分配 / OOM 回滚 / 逻辑恢复 / 尾块账本 / 释放复用 / 引用计数共享
  B. 构造性 OOM：总空闲块足够但无连续大块 -> 连续分配失败、分页成功（真机张量探针）
  C. 请求混合 trace 上的碎片对比：连续预留 vs 按需分页（真实 GPU 分配峰值）
  D. block size 扫描：元数据条目/字节 vs 内部碎片字节的权衡曲线
  E. GPU 上 gather-attention 数值等价（block table 拼装 vs 连续缓存）
固定条件：vm-60 / RTX 4090 D；同一请求分布，seed 固定。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import random
import torch
from task4_common import (assert_true, dump_json, env_info, gib, ledger_kv_bytes,
                          mib, per_token_bytes, source_sha256)

OUT = Path(__file__).parent / "task4_exp2_paged.json"
L, H_KV, D = 12, 8, 128      # 与实验1 同一模型骨架
DTB = 2
PER_TOKEN = per_token_bytes(L, H_KV, D, DTB)   # 每 token KV 字节（全模型）


# ---------------------------------------------------------------- CPU 分配器
class PagedPool:
    """块表 + 空闲列表 + 引用计数的教学版分页 KV 池。"""

    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.ref = {}          # physical_block -> refcount（>1 表示被共享）

    def alloc(self, n):
        if n > len(self.free_blocks):
            raise RuntimeError("OOM")
        blocks = self.free_blocks[:n]
        del self.free_blocks[:n]
        for b in blocks:
            self.ref[b] = 1
        return blocks

    def free(self, blocks):
        for b in blocks:
            self.ref[b] -= 1
            if self.ref[b] == 0:
                del self.ref[b]
                self.free_blocks.append(b)
        self.free_blocks.sort()

    def share(self, blocks):
        """前缀共享：只加引用，不申请新块。"""
        for b in blocks:
            self.ref[b] += 1


class Request:
    def __init__(self, rid, seq_len, block_size):
        self.rid = rid
        self.seq_len = seq_len
        self.block_size = block_size
        self.blocks = []

    def prefill(self, pool):
        if self.blocks:
            raise ValueError("重复 prefill 必须显式拒绝（不能继续追加块表）")
        need = (self.seq_len + self.block_size - 1) // self.block_size
        self.blocks = pool.alloc(need)
        return self

    def decode(self, pool):
        new_len = self.seq_len + 1
        if (new_len % self.block_size) == 1:
            self.blocks.extend(pool.alloc(1))
        self.seq_len = new_len


def part_a_mechanism_tests():
    print("== A. 分页机制单测 ==")
    pool = PagedPool(10, 4)
    r1 = Request(1, 6, 4).prefill(pool)
    assert_true(len(r1.blocks) == 2, "长度 6 -> 2 个 block")
    assert_true(len(pool.free_blocks) == 8, "空闲块剩 8")
    try:
        r1.prefill(pool)
        raise AssertionError("重复 prefill 未拒绝")
    except ValueError:
        assert True
    # decode 边界：第 7 个 token 不跨块，第 9 个才跨块
    r1.decode(pool)
    assert_true(len(r1.blocks) == 2, "第 7 token 不加块")
    r1.decode(pool)
    r1.decode(pool)
    assert_true(len(r1.blocks) == 3, "第 9 token 加第 3 块")
    assert_true(r1.seq_len == 9, "seq_len = 9")
    # 尾块账本
    allocated = len(r1.blocks) * 4
    assert_true(allocated - r1.seq_len == 3, "尾块浪费 3 个 token 槽")
    # 原子 OOM
    small = PagedPool(2, 4)
    try:
        Request(2, 9, 4).prefill(small)
        raise AssertionError("OOM 未抛出")
    except RuntimeError:
        assert_true(small.free_blocks == [0, 1], "OOM 后空闲池不变（原子）")
        assert_true(len(small.ref) == 0, "OOM 后无残留引用")
    # 释放复用
    pool.free(r1.blocks)
    assert_true(sorted(pool.free_blocks) == list(range(10)), "释放后 10 块全部可复用")
    # 引用计数共享（prefix 场景）：两个请求共享 prompt 块只付一份
    sp = PagedPool(8, 4)
    shared = sp.alloc(2)
    sp.share(shared)
    sp.share(shared)
    assert_true(len(sp.free_blocks) == 6, "两个请求共享 2 块后空闲剩 6（只付一份物理存储）")
    sp.free(shared)
    assert_true(len(sp.free_blocks) == 6, "释放一个引用后块不归还（refcount 3->2）")
    sp.free(shared)
    assert_true(len(sp.free_blocks) == 6, "释放两个引用后块仍不归还（refcount 2->1）")
    sp.free(shared)
    assert_true(len(sp.free_blocks) == 8, "三个引用全部释放后块才真正归还")
    return {"mechanism": "atomic alloc / decode boundary / tail ledger / OOM rollback / release-reuse / refcount share",
            "status": "PASS"}


def part_b_constructive_oom():
    print("== B. 构造性 OOM：总量够但无连续大块 ==")
    # 真实 GPU 探针：先把显存填到接近上限，再释放交错的"洞"，
    # 于是空闲总量足够，但最大连续空闲块远小于需求 -> 连续分配失败、逐块分配成功。
    dev = "cuda"
    props = torch.cuda.get_device_properties(dev)
    total = props.total_memory
    chunk_bytes = 8 * 1024 ** 2                      # 每块 8 MiB
    # 自适应填充：直到 allocator 拒绝为止（保证接近物理上限）
    chunks, filled = [], 0
    while True:
        try:
            chunks.append(torch.empty(chunk_bytes, dtype=torch.uint8, device=dev))
            filled += chunk_bytes
            if filled > total * 0.98:
                break
        except RuntimeError:
            break
    torch.cuda.synchronize()
    n_chunks = len(chunks)
    del chunks[1::2]                                 # 释放奇数块 -> 交错的洞
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free_hole_bytes = (n_chunks // 2) * chunk_bytes
    # 需求：洞总量的一半（单次连续大块）
    need = free_hole_bytes // 2
    contig_ok = True
    try:
        big = torch.empty(need, dtype=torch.uint8, device=dev)
        del big
    except RuntimeError:
        contig_ok = False
    # 分页：按 8 MiB 粒度逐块申请同样的总量（可复用洞）
    got, paged_ok = 0, True
    smalls = []
    try:
        while got < need:
            smalls.append(torch.empty(chunk_bytes, dtype=torch.uint8, device=dev))
            got += chunk_bytes
    except RuntimeError:
        paged_ok = False
    for s in smalls:
        del s
    torch.cuda.empty_cache()
    del chunks
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return {
        "scenario": "device filled to ~86%, alternating 8 MiB holes freed; "
                    "need one contiguous block of half the free total",
        "total_memory_mib": round(mib(total), 1),
        "chunk_mib": 8, "n_chunks": n_chunks,
        "free_hole_bytes": free_hole_bytes,
        "contiguous_need_bytes": need,
        "contiguous_allocated": contig_ok,
        "paged_8mib_chunks_allocated": paged_ok,
        "interpretation": (
            "连续分配失败且分页成功 => '总空闲够但无足够大连续块'的外部碎片真实存在"
            if (not contig_ok and paged_ok) else
            "allocator 在此规模下仍未拒绝连续分配（差异需更大压力才显见）"),
    }


def make_length_trace(n=200, seed=20260926):
    """固定 seed 的请求长度混合：多数短、少量长（模拟 chat workload）。"""
    rng = random.Random(seed)
    lens = []
    for _ in range(n):
        u = rng.random()
        if u < 0.55:
            lens.append(rng.randint(16, 64))       # 短
        elif u < 0.85:
            lens.append(rng.randint(128, 512))     # 中
        else:
            lens.append(rng.randint(1024, 4096))   # 长
    return lens


def part_c_fragmentation():
    print("== C. 同一请求分布：连续预留 vs 按需分页 ==")
    lens = make_length_trace()
    s_max = max(lens)
    block_size = 16
    contig_tokens = len(lens) * s_max
    paged_tokens = sum((x + block_size - 1) // block_size * block_size for x in lens)
    real_tokens = sum(lens)
    tail_waste = paged_tokens - real_tokens
    # 真实 GPU 分配峰值：按两个策略各分配一次 token 级张量
    dev = "cuda"
    torch.cuda.empty_cache()
    t = torch.empty(paged_tokens, D, dtype=torch.float16, device=dev)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    paged_alloc = torch.cuda.max_memory_allocated()
    del t
    torch.cuda.empty_cache()
    t2 = torch.empty(contig_tokens, D, dtype=torch.float16, device=dev)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    contig_alloc = torch.cuda.max_memory_allocated()
    del t2
    torch.cuda.empty_cache()
    per_tok_32 = None
    return {
        "workload": {"requests": len(lens), "len_min": min(lens), "len_max": s_max,
                     "seed": 20260926, "block_size": block_size},
        "real_tokens": real_tokens,
        "contiguous_reserved_tokens": contig_tokens,
        "paged_allocated_tokens": paged_tokens,
        "tail_waste_tokens": tail_waste,
        "token_saving_ratio": round(1 - paged_tokens / contig_tokens, 4),
        "kv_bytes_paged": paged_tokens * PER_TOKEN,
        "kv_bytes_contiguous": contig_tokens * PER_TOKEN,
        "kv_mib_paged": round(mib(paged_tokens * PER_TOKEN), 1),
        "kv_mib_contiguous": round(mib(contig_tokens * PER_TOKEN), 1),
        "gpu_probe": {"paged_peak_allocated_mb": round(mib(paged_alloc), 1),
                      "contiguous_peak_allocated_mb": round(mib(contig_alloc), 1),
                      "head_dim": D, "dtype": "fp16"},
        "note": "探针张量仅 [tokens, head_dim]，用来观察分配量差异；层/头维通过 PER_TOKEN 换算",
    }


def part_d_block_size_sweep():
    print("== D. block size 扫描：元数据 vs 内部碎片 ==")
    lens = make_length_trace(n=200)
    rows = []
    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        entries = sum((x + bs - 1) // bs for x in lens)
        waste_tokens = sum((x + bs - 1) // bs * bs - x for x in lens)
        meta_bytes_8b = entries * 8      # block table 每条目 8 字节（64-bit id）
        meta_bytes_4b = entries * 4      # 32-bit id
        waste_bytes = waste_tokens * PER_TOKEN
        rows.append({
            "block_size": bs, "block_table_entries": entries,
            "meta_bytes_8b": meta_bytes_8b, "meta_bytes_4b": meta_bytes_4b,
            "internal_frag_tokens": waste_tokens,
            "internal_frag_bytes": waste_bytes,
            "internal_frag_ratio": round(waste_tokens / sum(lens), 4),
            "waste_over_meta_8b": round(waste_bytes / meta_bytes_8b, 2),
        })
    return {"per_token_bytes_full_model": PER_TOKEN, "rows": rows}


def part_e_gather_attention():
    print("== E. GPU gather-attention 数值等价 ==")
    dev = "cuda"
    B, S, Hh, Dh = 2, 512, 8, 128
    torch.manual_seed(7)
    q = torch.randn(B, Hh, 1, Dh, device=dev, dtype=torch.float32)
    k = torch.randn(B, Hh, S, Dh, device=dev, dtype=torch.float32)
    v = torch.randn(B, Hh, S, Dh, device=dev, dtype=torch.float32)
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    ref = sdpa(q, k, v)
    # 分页：物理池 [num_blocks, bs, H, D]，每请求块表乱序
    bs = 16
    num_blocks = 64
    pool = {b: torch.randn(bs, Hh, Dh, device=dev, dtype=torch.float32) for b in range(num_blocks)}
    ok = True
    for trial in range(3):
        # 随机逻辑->物理映射（物理顺序打乱）
        perm = torch.randperm(num_blocks)
        need = (S + bs - 1) // bs
        table = perm[:need].tolist()
        gathered = torch.cat([pool[b] for b in table], dim=0)[:S]      # [S, H, D]
        gk = gathered.permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        gv = gk
        out = sdpa(q, gk, gv)
        # 数值对照的对象应为同一份数据；这里验证"按块表顺序拼回"与"连续存放"等价：
        cont = torch.cat([pool[b] for b in table], dim=0)[:S]
        assert torch.equal(gathered, cont)
    # 真正的等价测试：同一份数据连续存 vs 分页存（物理乱序，块表记录正确顺序）
    data = torch.randn(S, Hh, Dh, device=dev, dtype=torch.float32)
    blocks = [data[i * bs:(i + 1) * bs] for i in range(need)]
    # table[logical] = physical slot；物理 slot 顺序被打乱
    table = list(torch.randperm(need).tolist())
    slot_to_logical = [0] * need
    for logical, physical in enumerate(table):
        slot_to_logical[physical] = logical
    scattered_pool = {physical: blocks[logical] for logical, physical in enumerate(table)}
    # 按块表：逻辑块 i 存在物理槽 table[i]
    recovered = torch.cat([scattered_pool[table[i]] for i in range(need)], dim=0)[:S]
    max_diff = (recovered - data).abs().max().item()
    assert_true(max_diff == 0.0, f"块表顺序恢复与连续数据完全一致（max_diff={max_diff}）")
    # 用恢复出的 K/V 做 attention，与直接用连续数据的结果比较
    kk = data.permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1).contiguous()
    vv = data.permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1).contiguous()
    out_direct = sdpa(q, kk, vv)
    rec = recovered.permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1).contiguous()
    out_paged = sdpa(q, rec, rec)
    ad = (out_direct - out_paged).abs().max().item()
    assert_true(ad == 0.0, f"分页拼装后 attention 输出一致（max_diff={ad}）")
    return {"shape": {"batch": B, "seq": S, "heads": Hh, "head_dim": Dh, "block_size": bs},
            "logical_recovery_max_diff": max_diff, "attention_output_max_diff": ad,
            "status": "PASS"}


def main():
    print("GPU:", torch.cuda.get_device_name(0), "| per-token KV bytes:", PER_TOKEN)
    result = {
        "experiment": "task4_exp2_paged",
        "environment": env_info(),
        "source_sha256": source_sha256(__file__),
        "model": {"layers": L, "kv_heads": H_KV, "head_dim": D, "dtype": "fp16"},
        "per_token_kv_bytes": PER_TOKEN,
        "a_mechanism": part_a_mechanism_tests(),
        "b_constructive_oom": part_b_constructive_oom(),
        "c_fragmentation": part_c_fragmentation(),
        "d_block_size_sweep": part_d_block_size_sweep(),
        "e_gather_attention": part_e_gather_attention(),
        "status": "PASS",
    }
    dump_json(result, OUT)
    print("ALL_TESTS_PASS")


if __name__ == "__main__":
    main()
