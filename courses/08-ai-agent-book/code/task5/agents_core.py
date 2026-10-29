# -*- coding: utf-8 -*-
"""
agents_core.py -- 第10章 §10.3 实验10-1 / §10.2 表10-1 的共享内核。

只依赖 Python 标准库。提供：
  1) ResponsesClient : 面向本机 8789 端口 OpenAI 兼容代理(/v1/responses)的客户端，
     带重试+指数退避、逐次失败记录、token/延迟记账。
  2) 同一套本地工具实现（read_file / write_file / list_dir / calc / local_csv_query /
     web_fetch / run_python / load_skill），两条路径 A2/A3 用的是同一份代码，只是可见集不同。
  3) 一份"已固化的本地数据集"（中国 2021-2023 新能源汽车销量），带 provenance。
     注意：这是**固化数据**，不是实时检索；数据来源写在 DATASET_PROVENANCE 里。

用法：python agents_core.py --selftest   跑离线自检（不花钱、不发网络请求）
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import ssl
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------------------
# 0. 路径 & 常量
# --------------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
BOOK_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))          # courses/08-ai-agent-book
EVIDENCE_ROOT = os.path.join(BOOK_ROOT, "evidence", "task5")
DATA_ROOT = os.path.join(HERE, "data")

API_URL = os.environ.get("TASK5_API_URL", "http://127.0.0.1:8789/v1/responses")
MODEL = os.environ.get("TASK5_MODEL", "gpt-5.6-sol")

# 代理偶发 502 / 超时：指数退避重试
MAX_ATTEMPTS = int(os.environ.get("TASK5_MAX_ATTEMPTS", "5"))
BASE_BACKOFF = float(os.environ.get("TASK5_BASE_BACKOFF", "2.0"))
REQUEST_TIMEOUT = float(os.environ.get("TASK5_TIMEOUT", "300"))


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


# --------------------------------------------------------------------------------------
# 1. 本地固化数据集（PROVENANCE 必写清楚，笔记里也必须明示"非实时检索"）
# --------------------------------------------------------------------------------------
# 中国新能源汽车销量（万辆）。口径：中国汽车工业协会(CAAM) 发布的"新能源汽车"销量口径
# （纯电+插混+燃料电池，不含出口口径差异），单位万辆。此处为**人工核对后固化在仓库里的快照**，
# 采集方式：作者按 CAAM 年度产销快报/统计年报逐条抄录并取到 0.1 万辆精度，于 2026-10 固化。
# 数值用于本实验的"数据管道"演示；若要发表结论请回到原始发布重新核对。
DATASET_PROVENANCE = {
    "file": "data/cn_nev_sales_2021_2023.csv",
    "description": "中国新能源汽车销量（万辆），2021-2023 分年度与分季度",
    "source_org": "中国汽车工业协会 (CAAM)",
    "source_kind": "frozen_snapshot",          # <-- 非实时检索
    "collected_by": "manual transcription from CAAM annual press releases, rounded to 0.1 万辆",
    "frozen_at": "2026-10-26",
    "caveat": "这是仓库内固化的快照，不是实时联网检索结果；数字以 CAAM 公布口径为准，"
              "本实验用它保证可重复运行。",
    "annual_nev_sales_wan": {"2021": 352.1, "2022": 688.7, "2023": 949.5},
    # 分季度（万辆），加总与年度值一致（四舍五入到 0.1）
    "quarterly_nev_sales_wan": {
        "2021": {"Q1": 51.5, "Q2": 73.4, "Q3": 94.4, "Q4": 132.8},
        "2022": {"Q1": 125.7, "Q2": 134.0, "Q3": 196.7, "Q4": 232.3},
        "2023": {"Q1": 158.6, "Q2": 217.9, "Q3": 271.8, "Q4": 301.2},
    },
    # 同期汽车总销量（万辆），用于算渗透率
    "annual_total_auto_sales_wan": {"2021": 2627.5, "2022": 2686.4, "2023": 3009.4},
}

CSV_TEXT = """metric,period,value_wan,unit,source
nev_sales,Y2021,352.1,万辆,CAAM
nev_sales,Y2022,688.7,万辆,CAAM
nev_sales,Y2023,949.5,万辆,CAAM
total_auto_sales,Y2021,2627.5,万辆,CAAM
total_auto_sales,Y2022,2686.4,万辆,CAAM
total_auto_sales,Y2023,3009.4,万辆,CAAM
"""


def dataset_path() -> str:
    ensure_dir(DATA_ROOT)
    p = os.path.join(DATA_ROOT, "cn_nev_sales_2021_2023.csv")
    provenance = os.path.join(DATA_ROOT, "cn_nev_sales_2021_2023.provenance.json")
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            f.write(CSV_TEXT)
    if not os.path.exists(provenance):
        with open(provenance, "w", encoding="utf-8") as f:
            json.dump(DATASET_PROVENANCE, f, ensure_ascii=False, indent=2)
    return p


# --------------------------------------------------------------------------------------
# 2. Responses API 客户端：重试 + 退避 + 记账
# --------------------------------------------------------------------------------------
class ApiFailure(RuntimeError):
    pass


@dataclass
class CallRecord:
    """一次 HTTP 往返的完整记账（含失败尝试）。"""
    ok: bool
    wall_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    attempts: int = 0
    attempt_errors: List[Dict[str, Any]] = field(default_factory=list)
    status: Optional[str] = None
    response_id: Optional[str] = None
    error: Optional[str] = None
    started_at: str = ""
    finished_at: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "ok": self.ok, "wall_s": round(self.wall_s, 4),
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens, "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "attempts": self.attempts, "attempt_errors": self.attempt_errors,
            "status": self.status, "response_id": self.response_id, "error": self.error,
            "started_at": self.started_at, "finished_at": self.finished_at,
        }


class ResponsesClient:
    """POST /v1/responses 的薄封装。只做四件事：发、重试、记账、抽文本。"""

    def __init__(self, url: str = API_URL, model: str = MODEL,
                 max_attempts: int = MAX_ATTEMPTS, base_backoff: float = BASE_BACKOFF,
                 timeout: float = REQUEST_TIMEOUT, verbose: bool = True):
        self.url, self.model = url, model
        self.max_attempts, self.base_backoff, self.timeout = max_attempts, base_backoff, timeout
        self.verbose = verbose
        self.n_calls = 0
        self.n_requests = 0
        self.n_failed_attempts = 0
        self.n_failed_calls = 0
        self.total_wall_s = 0.0

    def _post_once(self, body: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    @staticmethod
    def sanitize_history(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """多轮时把 reasoning / item_reference 之类"服务端对象"剔除。

        实测：把上一轮返回的 reasoning item 原样回传会 400
        （"The requested item was created under a different Azure OpenAI resource"）。
        我们只保留可移植的 message / function_call / function_call_output 三种条目，
        function_call 的 id 换成局部稳定 id。
        """
        out = []
        for it in items:
            t = it.get("type")
            if t == "message" or (t is None and "role" in it):
                content = it.get("content")
                if isinstance(content, list):
                    txt = "".join(c.get("text", "") for c in content if c.get("type") in ("output_text", "input_text"))
                else:
                    txt = content or ""
                role = it.get("role") or "assistant"
                out.append({"role": role, "content": txt})
            elif t == "function_call":
                out.append({"type": "function_call", "name": it.get("name"),
                            "arguments": it.get("arguments"), "call_id": it.get("call_id")})
            elif t == "function_call_output":
                out.append({"type": "function_call_output", "call_id": it.get("call_id"),
                            "output": it.get("output")})
        return out

    def call(self, input_items: List[Dict[str, Any]], instructions: Optional[str] = None,
             tools: Optional[List[Dict[str, Any]]] = None, max_output_tokens: int = 1024,
             extra: Optional[Dict[str, Any]] = None) -> Tuple[CallRecord, Optional[Dict[str, Any]]]:
        body: Dict[str, Any] = {
            "model": self.model,
            "input": self.sanitize_history(input_items),
            "max_output_tokens": max_output_tokens,
        }
        if instructions is not None:
            body["instructions"] = instructions
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if extra:
            body.update(extra)

        rec = CallRecord(ok=False, wall_s=0.0, started_at=now_iso())
        t0 = time.time()
        last_exc: Optional[str] = None
        for attempt in range(1, self.max_attempts + 1):
            rec.attempts = attempt
            self.n_requests += 1
            try:
                payload = self._post_once(body)
                rec.status = payload.get("status")
                if payload.get("status") == "failed":
                    raise ApiFailure("status=failed: %s" % json.dumps(payload.get("error"), ensure_ascii=False)[:400])
                usage = payload.get("usage") or {}
                det = usage.get("input_tokens_details") or {}
                odet = usage.get("output_tokens_details") or {}
                rec.ok = True
                rec.input_tokens = int(usage.get("input_tokens") or 0)
                rec.output_tokens = int(usage.get("output_tokens") or 0)
                rec.reasoning_tokens = int(odet.get("reasoning_tokens") or 0)
                rec.cached_tokens = int(det.get("cached_tokens") or 0)
                rec.cache_write_tokens = int(det.get("cache_write_tokens") or 0)
                rec.response_id = payload.get("id")
                rec.incomplete_details = payload.get("incomplete_details")  # type: ignore[attr-defined]
                rec.wall_s = time.time() - t0
                rec.finished_at = now_iso()
                self.n_calls += 1
                self.total_wall_s += rec.wall_s
                return rec, payload
            except urllib.error.HTTPError as e:
                try:
                    body_txt = e.read().decode("utf-8", "replace")[:600]
                except Exception:
                    body_txt = "<unreadable>"
                last_exc = "HTTP %s: %s" % (e.code, body_txt)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError, ApiFailure) as e:
                last_exc = "%s: %s" % (type(e).__name__, e)
            self.n_failed_attempts += 1
            rec.attempt_errors.append({"attempt": attempt, "at": now_iso(), "error": last_exc})
            if self.verbose:
                print("    [retry] attempt %d/%d failed: %s" % (attempt, self.max_attempts, (last_exc or "")[:160]),
                      flush=True)
            if attempt < self.max_attempts:
                time.sleep(min(self.base_backoff * (2 ** (attempt - 1)), 60.0))

        rec.ok = False
        rec.error = last_exc
        rec.wall_s = time.time() - t0
        rec.finished_at = now_iso()
        self.n_failed_calls += 1
        self.total_wall_s += rec.wall_s
        return rec, None

    @staticmethod
    def extract_text(payload: Dict[str, Any]) -> str:
        parts = []
        for item in payload.get("output") or []:
            if item.get("type") == "message":
                for c in item.get("content") or []:
                    if c.get("type") == "output_text":
                        parts.append(c.get("text") or "")
        return "".join(parts).strip()

    @staticmethod
    def extract_function_calls(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [i for i in (payload.get("output") or []) if i.get("type") == "function_call"]

    def stats(self) -> Dict[str, Any]:
        return {
            "calls": self.n_calls, "requests": self.n_requests,
            "failed_attempts": self.n_failed_attempts, "failed_calls": self.n_failed_calls,
            "total_wall_s": round(self.total_wall_s, 3),
        }


# --------------------------------------------------------------------------------------
# 3. 工具实现（A2 / A3 共用同一份代码，只是"可见集"不同）
# --------------------------------------------------------------------------------------
ALLOWED_ROOTS = [DATA_ROOT, os.path.join(HERE, "skills"), HERE]


def _resolve_readable(path: str) -> str:
    cands = [path] if os.path.isabs(path) else [os.path.join(HERE, path), os.path.join(DATA_ROOT, path)]
    for c in cands:
        ap = os.path.abspath(c)
        if os.path.exists(ap) and any(ap.startswith(os.path.abspath(r)) for r in ALLOWED_ROOTS):
            return ap
    raise PermissionError("path not readable by harness: %r" % path)


def tool_read_file(path: str) -> str:
    ap = _resolve_readable(path)
    with open(ap, "r", encoding="utf-8") as f:
        txt = f.read(20000)
    return txt


def tool_list_dir(path: str = ".") -> str:
    ap = _resolve_readable(path)
    if not os.path.isdir(ap):
        return "not a directory: %s" % path
    return "\n".join(sorted(os.listdir(ap)))


def tool_write_file(path: str, content: str) -> str:
    ap = os.path.join(EVIDENCE_ROOT, "_scratch")
    ensure_dir(ap)
    fp = os.path.abspath(os.path.join(ap, os.path.basename(path)))
    with open(fp, "w", encoding="utf-8") as f:
        f.write(content)
    return "written %d bytes to %s" % (len(content), fp)


_BINOPS = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b, ast.Mod: lambda a, b: a % b,
}
_UNARY = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}
_SAFE_FUNCS = {"sqrt": math.sqrt, "log": math.log, "log10": math.log10,
               "exp": math.exp, "abs": abs, "round": round, "min": min, "max": max}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCS:
        return _SAFE_FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id == "pi":
        return math.pi
    raise ValueError("unsupported expression node: %s" % type(node).__name__)


def tool_calc(expression: str) -> str:
    """安全的四则/幂/取根计算器（不用 eval）。CAGR 由模型自己拼表达式。"""
    try:
        tree = ast.parse(expression, mode="eval")
        v = _eval_node(tree)
        return "= %.6f" % v if isinstance(v, float) else "= %s" % v
    except Exception as e:  # noqa: BLE001
        return "ERROR: %s: %s" % (type(e).__name__, e)


def tool_local_csv_query(metric: str = "nev_sales", period: str = "all") -> str:
    """查询固化数据集。返回 JSON 字符串（含 provenance 说明）。"""
    p = dataset_path()
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            if not line.strip():
                continue
            rows.append(dict(zip(header, line.strip().split(","))))
    sel = [r for r in rows if r["metric"] == metric and (period == "all" or r["period"] == period)]
    return json.dumps({
        "rows": sel,
        "provenance": {
            "kind": DATASET_PROVENANCE["source_kind"],
            "source_org": DATASET_PROVENANCE["source_org"],
            "note": DATASET_PROVENANCE["caveat"],
        },
    }, ensure_ascii=False)


def tool_web_fetch(url: str, max_chars: int = 4000) -> str:
    """真实联网抓取（urllib）。允许失败，失败信息原样返回给模型。"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE           # 部分站点证书链有问题（已知），这里放宽只为可读性
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (task5-experiment)"})
    try:
        with urllib.request.urlopen(req, timeout=25, context=ctx) as r:
            raw = r.read(400000)
            enc = r.headers.get_content_charset() or "utf-8"
        txt = raw.decode(enc, "replace")
        txt = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", txt)
        txt = re.sub(r"(?s)<[^>]+>", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt[:max_chars]
    except Exception as e:  # noqa: BLE001
        return "FETCH_ERROR: %s: %s" % (type(e).__name__, e)


def tool_run_python(code: str) -> str:
    p = dataset_path()
    return "ERROR: disabled in context-mode experiment (use calc/local_csv_query). Dataset at %s" % p


SKILL_DIR = os.path.join(HERE, "skills")


def list_skills() -> List[str]:
    if not os.path.isdir(SKILL_DIR):
        return []
    return sorted(d for d in os.listdir(SKILL_DIR) if os.path.isfile(os.path.join(SKILL_DIR, d, "SKILL.md")))


def tool_load_skill(name: str) -> str:
    """Skill 式路径：把 SKILL.md 全量追加成 tool result（= 轨迹末尾），不动静态前缀。"""
    safe = re.sub(r"[^A-Za-z0-9_-]", "", name or "")
    p = os.path.join(SKILL_DIR, safe, "SKILL.md")
    if not os.path.isfile(p):
        return "SKILL_NOT_FOUND: %r. available=%s" % (name, list_skills())
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


TOOL_IMPLS: Dict[str, Callable[..., str]] = {
    "load_skill": tool_load_skill,
    "read_file": tool_read_file,
    "list_dir": tool_list_dir,
    "write_file": tool_write_file,
    "calc": tool_calc,
    "local_csv_query": tool_local_csv_query,
    "web_fetch": tool_web_fetch,
    "run_python": tool_run_python,
}


def tool_schema(name: str, description: str, props: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {"type": "function", "name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}


def visible_tool_schemas(names: List[str]) -> List[Dict[str, Any]]:
    specs = {
        "read_file": tool_schema("read_file", "读取工作区内的文本文件（数据文件在 data/ 下）。",
                                 {"path": {"type": "string"}}, ["path"]),
        "list_dir": tool_schema("list_dir", "列出目录内容。", {"path": {"type": "string"}}, []),
        "write_file": tool_schema("write_file", "把草稿写入 scratch 目录。",
                                  {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
        "calc": tool_schema("calc", "安全计算器，支持 + - * / ** % 与 sqrt/log/exp/round/abs/min/max/pi。"
                                    "输入一个 Python 表达式字符串，例如 '(949.5/352.1)**(1/2)-1'。",
                            {"expression": {"type": "string"}}, ["expression"]),
        "local_csv_query": tool_schema("local_csv_query",
                                       "查询本地固化数据集（metric: nev_sales 或 total_auto_sales）。"
                                       "注意：这是仓库内固化快照，不是实时联网检索。",
                                       {"metric": {"type": "string"}, "period": {"type": "string"}},
                                       ["metric"]),
        "web_fetch": tool_schema("web_fetch", "抓取一个 URL 的纯文本内容（真实联网，可能失败）。",
                                 {"url": {"type": "string"}}, ["url"]),
        "run_python": tool_schema("run_python", "执行 Python 片段并返回 stdout。",
                                  {"code": {"type": "string"}}, ["code"]),
        "load_skill": tool_schema("load_skill", "按名字加载一份 SKILL.md（行为规程），内容作为工具结果进入轨迹。",
                                  {"name": {"type": "string"}}, ["name"]),
    }
    return [specs[n] for n in names]


# --------------------------------------------------------------------------------------
# 4. 轨迹 / 记账工具函数
# --------------------------------------------------------------------------------------
def count_chinese_chars(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text or ""))


def cjk_len(text: str) -> int:
    """中文字数口径：每个汉字算 1；连续的 ASCII 单词/数字算 1 个词。用于演示"字数"歧义。"""
    t = text or ""
    cjk = len(re.findall(r"[\u4e00-\u9fff]", t))
    words = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9.%\-]*", t))
    return cjk + words


def write_json(path: str, obj: Any) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# 5. 离线自检
# --------------------------------------------------------------------------------------
def _selftest() -> int:
    ok = True
    p = dataset_path()
    print("[selftest] dataset:", p)
    print("[selftest] csv head:", open(p, encoding="utf-8").read().splitlines()[:2])
    annual = DATASET_PROVENANCE["annual_nev_sales_wan"]
    q = DATASET_PROVENANCE["quarterly_nev_sales_wan"]
    for y in ("2021", "2022", "2023"):
        s = round(sum(q[y].values()), 1)
        if s != annual[y]:
            ok = False
            print("[selftest] FAIL quarterly sum mismatch", y, s, annual[y])
    cagr = (annual["2023"] / annual["2021"]) ** (1 / 2) - 1
    print("[selftest] CAGR 2021->2023 = %.6f  (calc tool: %s)" % (cagr, tool_calc("(949.5/352.1)**(1/2)-1")))
    print("[selftest] calc guard:", tool_calc("__import__('os').system('echo pwn')"))
    print("[selftest] csv_query:", tool_local_csv_query("nev_sales")[:120])
    print("[selftest] cjk_len('新能源车 CAGR 31% 吧') =", cjk_len("新能源车 CAGR 31% 吧"))
    print("[selftest] schema count:", len(visible_tool_schemas(["calc", "read_file"])))
    print("[selftest] RESULT:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
