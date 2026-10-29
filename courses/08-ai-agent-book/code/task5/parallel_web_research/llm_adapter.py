"""
LLM 适配器：真实模型 + 证据约束抽取 + 确定性门禁
================================================

移植自上游 llm.py，但把 openai SDK 换成本机 http://127.0.0.1:8789 的
**OpenAI Responses API**（chat/completions 在这个网关上不可用）：

    POST /v1/responses
    body: {"model": "gpt-5.6-sol", "input": <str 或 messages>, "max_output_tokens": 2048}
    resp: output[].content[] 中 type == "output_text" 的 text
           usage.input_tokens / usage.output_tokens

三个关键设计（对应上游的"证据约束抽取 + 名称门禁"）：

1. **确定性门禁（gate）**：目标的姓名/别名不在可见文本里，直接 not_found，
   不花一次 LLM 调用。它挡住"模型凭参数记忆编造履历"。
2. **逐字证据校验（verbatim evidence）**：模型必须回一段证据原文，且这段原文
   必须在页面可见文本里逐字存在（空白归一后），否则判为 unsupported。
   它挡住"模型凭参数记忆编造证据"。
3. **语义口径二次抽取（criterion）**：目标不是一个人名而是一个"角色的现任者"
   （例如 'Provost, Stanford University'），光靠字面匹配会误命中——真实页面
   profiles.stanford.edu/andrew-ng 里就有 'Vice Provost and Dean of Research'
   这一行。所以先做字面命中统计，再由模型判定"页面上陈述的现任者是谁"。

失败隔离：网络 5xx / 超时 / 返回为空 / JSON 解析失败都重试（指数退避 + 抖动），
重试耗尽后抛 LLMError，由 worker 当作"该站点失败"记录——**不重试整个站点**，
一个站点坏掉不能拖垮整批。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

DEFAULT_ENDPOINT = "http://127.0.0.1:8789/v1/responses"
DEFAULT_MODEL = "gpt-5.6-sol"

ReceiptSink = Optional[Callable[[Dict[str, Any]], None]]


class LLMError(RuntimeError):
    """重试耗尽后的 LLM 调用失败（分类后计入 failure_summary）。"""

    def __init__(self, kind: str, detail: str, attempts: int):
        super().__init__(f"{kind}: {detail} (attempts={attempts})")
        self.kind = kind
        self.detail = detail
        self.attempts = attempts


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def name_gate(target: str, text: str, aliases: Sequence[str] = ()) -> Dict[str, Any]:
    """确定性名称门禁：目标姓名/别名在可见文本里出现过吗？出现几次？"""
    haystack = _normalize(text).casefold()
    needles = [target] + list(aliases)
    hits = {}
    for needle in needles:
        needle_norm = _normalize(needle).casefold()
        if not needle_norm:
            continue
        hits[needle] = haystack.count(needle_norm)
    return {
        "target": target,
        "present": any(v > 0 for v in hits.values()),
        "literal_hits": hits,
        "total_hits": sum(hits.values()),
    }


def gate_terms(site_name: str) -> List[str]:
    """从站点名里取出关键词，作为"页面上必须出现某个词"的廉价门禁。

    站点名形如 stanford-provost-office，拆出 {stanford, provost, office}；
    只要任一词出现在可见文本里就允许调用模型（真正的语义判定仍由模型 + 证据校验负责）。
    """
    parts = [p for p in re.split(r"[^a-z0-9]+", site_name.casefold()) if len(p) >= 4]
    stop = {"stanford", "university", "college", "school", "people", "faculty", "office",
            "profiles", "directory", "departments", "department"}
    # 保留 'stanford' 之外的通用词会被当成主题词，造成几乎必然触发；
    # 站点名里的学科词（cs / provost / law / education ...）才是有效信号。
    return [p for p in parts if p not in stop]


@dataclass
class LLMResult:
    text: str
    usage: Dict[str, Any]
    response_id: Optional[str]
    latency_seconds: float
    attempts: int


class LLMClient:
    """最小 Responses API 客户端：urllib + asyncio.to_thread，纯标准库。"""

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        max_output_tokens: int = 2048,
        timeout: float = 180.0,
        max_attempts: int = 4,
        backoff_base: float = 1.2,
        receipt_sink: ReceiptSink = None,
    ):
        self.endpoint = endpoint
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.receipt_sink = receipt_sink
        self.calls = 0
        self.attempts = 0
        self.failures: List[Dict[str, Any]] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.in_flight = 0
        self.max_in_flight = 0

    # —— 单次 HTTP ——
    def _post_once(self, body: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _extract_text(payload: Dict[str, Any]) -> str:
        parts = []
        for item in payload.get("output") or []:
            for chunk in item.get("content") or []:
                if chunk.get("type") == "output_text":
                    parts.append(chunk.get("text") or "")
        return "".join(parts)

    async def complete(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> LLMResult:
        """调一次模型；失败按指数退避重试。重试耗尽抛 LLMError。"""
        context = dict(context or {})
        body = {"model": self.model, "input": prompt, "max_output_tokens": self.max_output_tokens}
        started = time.monotonic()
        last_kind, last_detail = "unknown", ""
        for attempt in range(1, self.max_attempts + 1):
            self.attempts += 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            t0 = time.monotonic()
            try:
                payload = await asyncio.to_thread(self._post_once, body)
                text = self._extract_text(payload)
                if not text.strip():
                    raise ValueError("empty model response")
                usage = payload.get("usage") or {}
                self.calls += 1
                self.total_input_tokens += int(usage.get("input_tokens") or 0)
                self.total_output_tokens += int(usage.get("output_tokens") or 0)
                result = LLMResult(
                    text=text,
                    usage=usage,
                    response_id=payload.get("id"),
                    latency_seconds=round(time.monotonic() - t0, 3),
                    attempts=attempt,
                )
                if self.receipt_sink:
                    self.receipt_sink({
                        "kind": "llm_response",
                        "context": context,
                        "model": self.model,
                        "response_id": result.response_id,
                        "usage": usage,
                        "latency_seconds": result.latency_seconds,
                        "attempts": attempt,
                        "prompt_sha256_len": len(prompt),
                    })
                return result
            except Exception as exc:  # noqa: BLE001 —— 网络层异常种类太多，统一分类
                last_kind = type(exc).__name__
                last_detail = str(exc)[:300]
                if isinstance(exc, urllib.error.HTTPError):
                    last_kind = f"HTTP{exc.code}"
                elif isinstance(exc, urllib.error.URLError):
                    last_kind = "URLError"
                if self.receipt_sink:
                    self.receipt_sink({
                        "kind": "llm_error",
                        "context": context,
                        "error_type": last_kind,
                        "error_message": last_detail,
                        "attempt": attempt,
                        "latency_seconds": round(time.monotonic() - t0, 3),
                    })
                if attempt < self.max_attempts:
                    # 指数退避 + 抖动：避免 N 个 worker 在同一时刻一起重试
                    delay = self.backoff_base * (2 ** (attempt - 1))
                    await asyncio.sleep(delay + 0.15 * attempt)
            finally:
                self.in_flight -= 1
        self.failures.append({
            "error_type": last_kind,
            "error_message": last_detail,
            "context": context,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        raise LLMError(last_kind, last_detail, self.max_attempts)


# —— 提示词构造 ——

def _clip(text: str, limit: int = 12000) -> str:
    return text[:limit]


def build_literal_prompt(query: str, site_name: str, url: str, text: str, gate: Dict[str, Any]) -> str:
    return f"""You are an evidence-constrained web-extraction worker inside a parallel research system.
WORKER SITE: {site_name}
WORKER URL: {url}
TARGET: {query}
DETERMINISTIC PRE-CHECK: the target string occurs {gate["total_hits"]} time(s) in the page.

Decide whether the page contains a FACULTY PROFILE / PEOPLE DIRECTORY ENTRY for the target.
Rules:
- Use ONLY the page text below. Do not use memorized knowledge about any person.
- "evidence" must be a SHORT VERBATIM excerpt copied character-for-character from the page
  (30-200 chars). If you cannot copy such an excerpt, the answer is not_found.
- If the name appears only as part of a different string or as navigation/menu text, set
  found=false and explain in "reason".
- If the page only lists the name and the requested field is absent, set found=true and leave
  that field empty. Never invent a value.

Return ONLY compact JSON with keys:
found (bool), name (string|null), college (string|null), position (string|null),
research (string|null), evidence (string|null), reason (string|null)

PAGE TEXT:
{_clip(text)}"""


def build_criterion_prompt(criterion: str, site_name: str, url: str, text: str, gate: Dict[str, Any]) -> str:
    literal = gate["total_hits"]
    if literal:
        precheck = (f'DETERMINISTIC PRE-CHECK: the literal string "{gate["target"]}" does occur '
                    f'{literal} time(s) in the page - but a literal occurrence is NOT proof that the '
                    f'page names the role holder. Many pages mention a role title without naming who holds it.')
    else:
        precheck = (f'DETERMINISTIC PRE-CHECK: the literal string "{gate["target"]}" does NOT appear on '
                    f'this page at all. A page can still describe the office and its responsibilities; '
                    f'decide from what the page actually states.')
    return f"""You are an evidence-constrained web-extraction worker inside a parallel research system.
WORKER SITE: {site_name}
WORKER URL: {url}
ROLE BEING RESOLVED: {criterion}
{precheck}

Task: identify the person who CURRENTLY HOLDS the role described by ROLE BEING RESOLVED,
but ONLY if this specific page states that person and their role.
Rules:
- Use ONLY the page text below. Do not use memorized knowledge; do not pick a name because you
  recognize it from the wider world.
- Other people who appear near the phrase (deans, vice provosts, former holders, staff) are NOT
  the answer unless the page explicitly assigns them the role.
- If the page merely mentions the role, or lists unrelated people, set found=false.
- "evidence" must be a SHORT VERBATIM excerpt (30-200 chars) copied character-for-character from
  the page that assigns the role to the person.
- Never invent a field value; leave it empty when the page does not state it.

Return ONLY compact JSON with keys:
found (bool), name (string|null), role (string|null), institution (string|null),
evidence (string|null), reason (string|null)

PAGE TEXT:
{_clip(text)}"""


def parse_json_object(raw: str) -> Dict[str, Any]:
    """模型偶尔会包一层 ```json 或加解释文字，这里做宽容解析。"""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError(f"model did not return JSON: {raw[:160]!r}")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("model JSON is not an object")
    return value


def verify_evidence(evidence: Any, text: str) -> Dict[str, Any]:
    """逐字证据校验：证据必须是页面可见文本的子串（空白归一后）。"""
    if not isinstance(evidence, str) or not evidence.strip():
        return {"verbatim": False, "reason": "missing evidence"}
    ev = _normalize(evidence).casefold()
    page = _normalize(text).casefold()
    if ev in page:
        return {"verbatim": True, "reason": "evidence found verbatim"}
    # 退一步：证据被模型截断成多行时，检查最长的 60 字符片段
    probe = ev[: max(30, min(60, len(ev)))]
    if probe in page:
        return {"verbatim": True, "reason": "evidence prefix found verbatim"}
    return {"verbatim": False, "reason": "evidence not found in page text"}


async def extract_profile(
    client: LLMClient,
    query: str,
    site_name: str,
    url: str,
    text: str,
    criterion: Optional[str] = None,
    aliases: Sequence[str] = (),
    context: Optional[Dict[str, Any]] = None,
    gate_terms_override: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """带门禁与证据校验的抽取。返回 dict，永不抛异常（失败信息放在 status 里）。

    门禁分两级：
      1) 字面门禁：query 串本身出现在页面上 -> 一定调用模型（precision 检查）
      2) 主题门禁：query 串不在页面上，但站点主题词出现 -> 仍调用模型，
         由模型的语义判定 + 逐字证据校验决定（recall 检查，避免"字面匹配漏召回"）
      两者都不满足 -> 直接 not_found，不花一次 LLM 调用。
    """
    started = time.monotonic()
    result: Dict[str, Any] = {
        "query": query,
        "site": site_name,
        "url": url,
        "found": False,
        "status": "error",
        "reason": "",
        "profile": None,
        "gate": None,
        "evidence_check": None,
        "llm_latency_seconds": 0.0,
        "llm_calls": 0,
        "seconds": 0.0,
    }
    gate = name_gate(query, text, aliases)
    terms = list(gate_terms_override) if gate_terms_override is not None else gate_terms(site_name)
    for word in gate_terms(query):
        if word not in terms:
            terms.append(word)
    haystack = _normalize(text).casefold()
    topic_hits = {t: haystack.count(t) for t in terms}
    gate["topic_terms"] = topic_hits
    gate["topic_present"] = any(v > 0 for v in topic_hits.values())
    result["gate"] = gate
    if not gate["present"] and not gate["topic_present"]:
        # 两级门禁都没过：页面既没有目标串也没有主题词 -> 直接 not_found，省掉一次 LLM 调用
        result.update(status="not_found",
                      reason="target string and site topic both absent from rendered page (deterministic gate)")
        result["seconds"] = round(time.monotonic() - started, 3)
        return result

    # 判据（criterion）与字面目标不同时，说明要的是"角色现任者"而不是字面串
    used_criterion = bool(criterion) and (criterion or "").casefold() != query.casefold()
    result["mode"] = "criterion" if used_criterion else "literal"
    prompt = (
        build_criterion_prompt(criterion, site_name, url, text, gate)
        if used_criterion
        else build_literal_prompt(query, site_name, url, text, gate)
    )
    try:
        llm = await client.complete(prompt, context=context)
    except LLMError as exc:
        result.update(status="error", reason=f"LLMError[{exc.kind}]: {exc.detail}")
        result["seconds"] = round(time.monotonic() - started, 3)
        return result
    result["llm_calls"] += 1
    result["llm_latency_seconds"] = round(result["llm_latency_seconds"] + llm.latency_seconds, 3)
    result["llm_response_id"] = llm.response_id
    try:
        data = parse_json_object(llm.text)
    except ValueError as exc:
        result.update(status="error", reason=f"ModelOutput: {exc}")
        result["seconds"] = round(time.monotonic() - started, 3)
        return result

    found = bool(data.get("found"))
    check = verify_evidence(data.get("evidence"), text)
    result["evidence_check"] = check
    if not found:
        result.update(status="not_found", reason=str(data.get("reason") or "model: not found"))
        result["seconds"] = round(time.monotonic() - started, 3)
        return result
    if not check["verbatim"]:
        # 证据不能逐字对上 -> 视为幻觉，判 unsupported
        result.update(status="unsupported", reason="evidence not verbatim in page text")
        result["profile"] = data
        result["seconds"] = round(time.monotonic() - started, 3)
        return result

    name = _normalize(str(data.get("name") or ""))
    if name and not name_gate(name, text)["present"]:
        result.update(status="unsupported", reason=f"claimed name {name!r} absent from page")
        result["profile"] = data
        result["seconds"] = round(time.monotonic() - started, 3)
        return result

    result.update(
        found=True,
        status="found",
        reason="evidence-verified hit",
        profile={
            "name": name or None,
            "college": data.get("college") or data.get("institution"),
            "position": data.get("position") or data.get("role"),
            "research": data.get("research"),
            "evidence": _normalize(str(data.get("evidence"))),
            "criterion": criterion if used_criterion else None,
        },
    )
    result["seconds"] = round(time.monotonic() - started, 3)
    return result
