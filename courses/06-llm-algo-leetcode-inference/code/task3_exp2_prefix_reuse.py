# -*- coding: utf-8 -*-
"""Task3 实验 2：RadixAttention 前缀复用——共享 system prompt workload 的命中账本。

问题：RadixAttention 如何通过前缀树组织多个请求的 KV Cache？最长前缀匹配起什么作用？
- workload：N 个请求共享同一段 system prompt，各自带不同后缀；其中一批请求是
  「多轮对话」——后一轮在前一轮基础上追加，前缀逐轮变长。
- 指标：hit_len / reused_tokens / prefill_work_reduction（用 notebook 24/34 的
  同款实现，保证与题目区口径一致）。
- 容量治理：用 LRU 缓存（notebook 69 同款 block 模拟）扫描容量->命中率曲线，
  观察驱逐如何把复用收益还回去。
"""
import json
import random

from importlib import import_module

# 直接引用三个解答版 notebook 提取出的实现，保证与题目区口径一致
import sys
sys.path.insert(0, ".")
radix_mod = import_module("nb_radix") if False else None

OUT = "task3_exp2_prefix_reuse.json"

SYSTEM_PROMPT = list(range(1000, 1000 + 256))   # 256 token 的共享 system prompt


def build_workload(n_requests=64, seed=7):
    """生成共享前缀 workload：一半独立请求，一半多轮对话（前缀逐轮增长）。"""
    rng = random.Random(seed)
    workload = []
    # 独立请求：system prompt + 各自独特的 suffix
    for i in range(n_requests // 2):
        suffix = [rng.randint(20000, 30000) for _ in range(rng.randint(20, 80))]
        workload.append({"type": "independent", "tokens": SYSTEM_PROMPT + suffix})
    # 多轮对话：turn k 的输入 = system + 历史 + 新问题，前缀逐轮变长
    for i in range(n_requests // 2, n_requests):
        turns = rng.randint(2, 5)
        history = []
        for t in range(turns):
            qa = [rng.randint(20000, 30000) for _ in range(rng.randint(10, 40))]
            history = history + qa
            workload.append({"type": "multiturn",
                             "tokens": SYSTEM_PROMPT + history})
    return workload


# ---- Radix Tree（与 notebook 24 解答版一致的最小实现） ----
class TreeNode:
    def __init__(self, key_tokens, terminal=False):
        self.key_tokens = list(key_tokens)
        self.children = []
        self.terminal = terminal


class SimpleRadixCache:
    def __init__(self):
        self.root = TreeNode([])

    def _find_child(self, node, first_token):
        for child in node.children:
            if child.key_tokens and child.key_tokens[0] == first_token:
                return child
        return None

    def _lcp_len(self, a, b):
        n = 0
        while n < len(a) and n < len(b) and a[n] == b[n]:
            n += 1
        return n

    def insert(self, tokens):
        tokens = list(tokens)
        if not tokens:
            raise ValueError("tokens 不能为空")
        node, offset = self.root, 0
        while offset < len(tokens):
            child = self._find_child(node, tokens[offset])
            common = self._lcp_len(child.key_tokens, tokens[offset:]) if child else 0
            if child is None:
                node.children.append(TreeNode(tokens[offset:], terminal=True))
                return
            if common == len(child.key_tokens):
                node, offset = child, offset + common
                continue
            shared = TreeNode(child.key_tokens[:common])
            old_suffix = TreeNode(child.key_tokens[common:], terminal=child.terminal)
            old_suffix.children = child.children
            shared.children.append(old_suffix)
            node.children[node.children.index(child)] = shared
            if offset + common == len(tokens):
                shared.terminal = True
            else:
                shared.children.append(TreeNode(tokens[offset + common:], terminal=True))
            return

    def match_prefix(self, prompt_tokens):
        best, node, offset = 0, self.root, 0
        while offset < len(prompt_tokens):
            child = self._find_child(node, prompt_tokens[offset])
            common = self._lcp_len(child.key_tokens, prompt_tokens[offset:]) if child else 0
            if child is None or common < len(child.key_tokens):
                break
            offset += common
            node = child
            if node.terminal:
                best = offset
        return best


def simulate_lru(token_sequences, block_size=16, capacity_blocks=8):
    """notebook 69 同款 LRU block 缓存模拟。"""
    cache, clock = {}, 0
    reused = total = hit_req = evict = 0
    for tokens in token_sequences:
        if not tokens:
            continue
        total += len(tokens)
        blocks = [tuple(tokens[s:s + block_size])
                  for s in range(0, len(tokens) - block_size + 1, block_size)]
        hit_blocks = 0
        for i, b in enumerate(blocks):
            if (i, b) not in cache:
                break
            hit_blocks += 1
            clock += 1
            cache[(i, b)] = clock
        if hit_blocks:
            hit_req += 1
        reused += hit_blocks * block_size
        for i, b in enumerate(blocks):
            clock += 1
            cache[(i, b)] = clock
            while len(cache) > capacity_blocks:
                del cache[min(cache, key=cache.get)]
                evict += 1
    n = len([t for t in token_sequences if t])
    return {"hit_rate": hit_req / n if n else 0.0,
            "token_reuse_rate": reused / total if total else 0.0,
            "eviction_count": evict}


def main():
    workload = build_workload()
    sequences = [item["tokens"] for item in workload]
    print("workload requests:", len(sequences), flush=True)

    # 1) Radix Tree 最长前缀匹配：按请求顺序插入+匹配
    cache = SimpleRadixCache()
    rows = []
    total_tokens = hit_tokens = 0
    for idx, tokens in enumerate(sequences):
        hit_len = cache.match_prefix(tokens)
        cache.insert(tokens)
        total_tokens += len(tokens)
        hit_tokens += hit_len
        rows.append({"idx": idx, "prompt_len": len(tokens), "hit_len": hit_len,
                     "type": workload[idx]["type"]})
    report = {
        "workload": {"requests": len(sequences), "system_prompt_len": len(SYSTEM_PROMPT)},
        "radix": {
            "avg_hit_len": round(hit_tokens / len(sequences), 2),
            "avg_prompt_len": round(total_tokens / len(sequences), 2),
            "prefill_work_reduction": round(hit_tokens / total_tokens, 4),
            "requests_with_hit": sum(1 for r in rows if r["hit_len"] > 0),
            "max_hit_len": max(r["hit_len"] for r in rows),
        },
        "rows_head": rows[:8],
    }
    print("radix:", json.dumps(report["radix"], ensure_ascii=False), flush=True)

    # 2) LRU 容量扫描：容量不足时驱逐把复用收益还回去
    sweep = []
    for cap in (8, 16, 32, 64, 128, 256):
        row = simulate_lru(sequences, block_size=16, capacity_blocks=cap)
        row["capacity_blocks"] = cap
        row["capacity_tokens"] = cap * 16
        sweep.append(row)
        print("lru cap", cap, json.dumps(row, ensure_ascii=False), flush=True)
    report["lru_sweep"] = sweep

    # 3) 无缓存 baseline：全部重算
    report["no_cache"] = {"prefill_work_reduction": 0.0,
                          "avg_hit_len": 0.0}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
