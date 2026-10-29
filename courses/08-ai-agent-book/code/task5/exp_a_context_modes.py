# -*- coding: utf-8 -*-
"""
exp_a_context_modes.py -- §10.3 实验10-1（简化版）

任务：查找中国 2021-2023 年新能源汽车销量 -> 算 CAGR -> 写不超过 120 字的投资人摘要。

三条路径（同一模型、同一任务、同一套工具实现、同一份角色规程）：
  A1 单系统提示词基线：一份"什么都管"的长 system prompt，工具全集可见，多轮工具循环。
  A2 角色切换（transfer_to_agent）：triage -> research -> coding -> data_analysis -> writing，
     每阶段替换 system prompt（= instructions）与工具可见集，完整轨迹一路继承。
  A3 Skill 式：system prompt 与工具全集全程固定，模型按需 load_skill(name) 把 SKILL.md
     追加进轨迹（作为 tool result）。

记录：每阶段 input/output token、耗时、attempts、失败尝试；最终 CAGR、摘要字数、术语一致性。
产出：evidence/task5/exp10-1/raw/<run_id>.json、runs.jsonl、summary.json

用法：
  python exp_a_context_modes.py --repeats 3
  python exp_a_context_modes.py --repeats 1 --modes A1      # 冒烟
  python exp_a_context_modes.py --summarize-only            # 只用已有 runs.jsonl 重算 summary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agents_core as core  # noqa: E402

EXP_DIR = os.path.join(core.EVIDENCE_ROOT, "exp10-1")
RAW_DIR = os.path.join(EXP_DIR, "raw")
RUNS_PATH = os.path.join(EXP_DIR, "runs.jsonl")
MAX_STEPS = 14          # 单次运行的模型调用上限（预算意识：别让它无限打转）
MAX_OUT = 1200

TASK_TEXT = (
    "任务：查清中国 2021—2023 年新能源汽车（NEV）的年度销量，计算 2021→2023 的复合年均增长率 CAGR，"
    "并写一段不超过 120 字的投资人摘要。\n"
    "硬性要求：(1) 销量数字必须来自工具返回的真实数据，不许凭记忆编；(2) CAGR 必须用 calc 工具实算；"
    "(3) 摘要不超过 120 字；(4) 摘要里要标注数据口径。\n"
    "完成后请在最后一条消息里给出 JSON：{\"summary\": \"...\", \"cagr\": <数值或null>, \"v0\": <数值或null>, "
    "\"v1\": <数值或null>, \"years\": <整数或null>, \"provenance_noted\": true/false}"
)

# ----------------------------------------------------------------------------- 角色规程
BASE_RULES = (
    "你在一个多阶段 agent 系统里工作。工具返回的内容才是事实，你脑中的记忆不是事实。\n"
    "不要臆造数据。每次需要数字时优先用工具拿。"
)

ROLE_PROMPTS: Dict[str, str] = {
    "triage": BASE_RULES + "\n\n【角色：triage / 需求收集】\n"
        "你的职责是把用户需求收敛成一份可执行的交接单：目标、必须产出的交付物、硬约束（长度/口径/精度）。"
        "你只有 transfer_to_agent 和 read_file。收敛完成后立刻移交 research。交接单写在你的消息正文里，"
        "后面的角色会看到完整轨迹，所以不要丢关键约束。",
    "research": BASE_RULES + "\n\n【角色：research / 信息检索】\n"
        "你的职责是拿到**真实数据**：优先 local_csv_query（本地固化数据集，注意在结论里标注它是快照），"
        "web_fetch 只做交叉验证，失败也没关系，如实记录。拿到 2021/2022/2023 三个年度值后移交 coding。",
    "coding": BASE_RULES + "\n\n【角色：coding / 编程】\n"
        "你的职责是用 calc 工具把 CAGR 算出来。n 是间隔年数。算完把 V0、Vn、n、CAGR 写清楚，移交 data_analysis。",
    "data_analysis": BASE_RULES + "\n\n【角色：data_analysis / 数据分析】\n"
        "你的职责是复核前面阶段的数字：口径是否一致、单位是否一致、CAGR 公式是否正确、结论是否被数据支持。"
        "发现矛盾就明确写出矛盾，然后移交给 writing。你不再调用检索工具。",
    "writing": BASE_RULES + "\n\n【角色：writing / 写作】\n"
        "你的职责是写最终投资人摘要：不超过 120 字、先结论后数字、术语统一（新能源汽车（NEV）首次写全称）、"
        "标注数据口径、不写情绪化形容词。写完后按用户要求输出 JSON。",
}

ROLE_TOOLS: Dict[str, List[str]] = {
    "triage": ["read_file"],
    "research": ["read_file", "local_csv_query", "web_fetch", "list_dir"],
    "coding": ["calc", "read_file"],
    "data_analysis": ["calc", "read_file", "list_dir"],
    "writing": ["read_file", "write_file"],
}
ROLE_ORDER = ["triage", "research", "coding", "data_analysis", "writing"]

TRANSFER_SCHEMA = core.tool_schema(
    "transfer_to_agent",
    "把控制权移交给下一个角色。history 会完整保留，新角色只换 system prompt 和工具集。"
    "target 必须是 triage/research/coding/data_analysis/writing 之一。",
    {"target": {"type": "string"}, "handoff": {"type": "string"}},
    ["target", "handoff"])

# A1：一份静态的"什么都管"的长 prompt（把所有阶段指导压在一起 = 书中说的另一种选择）
A1_SYSTEM = (
    BASE_RULES + "\n\n你是一个全能的单 agent：既要理解需求、又要检索数据、又要写代码算指标、又要审查、又要写作。\n"
    "工作流程建议：(1) 先看清任务和硬约束；(2) 用 local_csv_query 拿真实数据；(3) 用 calc 算 CAGR（n 是间隔年数）；"
    "(4) 自己复核一遍数字与口径；(5) 写不超过 120 字的投资人摘要。\n"
    "摘要要求：先结论后数字、术语统一（新能源汽车（NEV）首次写全称）、标注数据口径、不写情绪化形容词。\n"
    "第一次系统调用的 system prompt 会把所有阶段指导一次给你。"
)
A1_TOOLS = ["read_file", "list_dir", "local_csv_query", "web_fetch", "calc", "write_file"]

# A3：静态前缀 + 技能目录；工具全集固定
A3_SYSTEM = (
    BASE_RULES + "\n\n你是一个配备了技能库（Skill）的 agent。system prompt 与工具集在整个会话中固定不变，"
    "角色规程不进 system prompt，而是按需用 load_skill(name) 加载，加载内容作为工具结果进入轨迹。\n"
    "可用技能目录：\n"
    "  - dataset-provenance : 本地固化数据集的口径与引用规范\n"
    "  - cagr-computation   : CAGR 的定义、年数口径与必须用 calc 实算的规则\n"
    "  - investment-memo    : 投资人摘要的长度/结构/术语一致性规程\n"
    "工作方式：先用 local_csv_query 拿数据，然后按需要 load_skill 加载相应规程，再用 calc 计算，最后按规程写作。\n"
    "在最后一条消息里输出 JSON（字段见用户消息）。"
)
A3_TOOLS = ["read_file", "list_dir", "local_csv_query", "web_fetch", "calc", "write_file", "load_skill"]


# ----------------------------------------------------------------------------- 运行记录
class Run:
    def __init__(self, run_id: str, mode: str, repeat: int, model: str):
        self.run_id, self.mode, self.repeat, self.model = run_id, mode, repeat, model
        self.started_at = core.now_iso()
        self.steps: List[Dict[str, Any]] = []
        self.tool_events: List[Dict[str, Any]] = []
        self.final_text: str = ""
        self.error: Optional[str] = None

    def step(self, role: str, instructions: str, tools: List[str], rec: core.CallRecord,
             payload: Optional[Dict[str, Any]], note: str = "") -> Dict[str, Any]:
        text = core.ResponsesClient.extract_text(payload) if payload else ""
        calls = core.ResponsesClient.extract_function_calls(payload) if payload else []
        s = {
            "step": len(self.steps) + 1,
            "role": role,
            "note": note,
            "instructions_chars": len(instructions or ""),
            "visible_tools": tools,
            "static_prefix_fingerprint": prefix_fingerprint(instructions or "", tools),
            **rec.to_json(),
            "assistant_text": text,
            "function_calls": [{"name": c.get("name"), "arguments": c.get("arguments"),
                                "call_id": c.get("call_id")} for c in calls],
        }
        self.steps.append(s)
        return s

    def to_json(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id, "mode": self.mode, "repeat": self.repeat, "model": self.model,
            "started_at": self.started_at, "finished_at": core.now_iso(),
            "task": TASK_TEXT, "max_steps": MAX_STEPS, "n_steps": len(self.steps),
            "steps": self.steps, "tool_events": self.tool_events,
            "final_text": self.final_text, "error": self.error,
        }


def prefix_fingerprint(instructions: str, tools: List[str]) -> str:
    h = hashlib.sha256()
    h.update((instructions or "").encode("utf-8"))
    h.update(("|".join(sorted(tools))).encode("utf-8"))
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------- 轨迹操作
def user_msg(text: str) -> Dict[str, Any]:
    return {"role": "user", "content": text}


def exec_tool(name: str, args: Dict[str, Any], run: Run, allowed: List[str], role: str) -> Tuple[str, bool]:
    """harness 侧的工具执行 + 硬权限闸门（Skill 路径的硬权限靠这里，不靠模型自觉）。"""
    t0 = time.time()
    ok = True
    if name not in allowed:
        out = "TOOL_NOT_PERMITTED: %s 不在当前可见工具集 %s 内" % (name, allowed)
        ok = False
    elif name == "transfer_to_agent":
        out = "transfer received"
    elif name in core.TOOL_IMPLS:
        try:
            out = core.TOOL_IMPLS[name](**args)
        except TypeError as e:
            out, ok = "BAD_ARGS: %s" % e, False
        except Exception as e:  # noqa: BLE001
            out, ok = "%s: %s" % (type(e).__name__, e), False
    else:
        out, ok = "UNKNOWN_TOOL: %s" % name, False
    dt = time.time() - t0
    ev = {"role": role, "tool": name, "arguments": args, "ok": ok,
          "result_preview": (out or "")[:1200], "result_chars": len(out or ""),
          "latency_s": round(dt, 4), "at": core.now_iso()}
    if name == "load_skill":
        ev["skill_chars"] = len(out or "")
    run.tool_events.append(ev)
    return out, ok


def append_output(history: List[Dict[str, Any]], payload: Dict[str, Any]) -> None:
    """把模型输出的 items（含 reasoning）原样塞回轨迹，保持多轮一致性。"""
    for item in payload.get("output") or []:
        history.append(item)


def parse_final_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = list(re.finditer(r"\{", text))
    for mm in reversed(m):
        sub = text[mm.start():]
        depth = 0
        for i, ch in enumerate(sub):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(sub[:i + 1])
                    except Exception:  # noqa: BLE001
                        break
    return None


# ----------------------------------------------------------------------------- 三个路径
def run_a1(client: core.ResponsesClient, repeat: int) -> Run:
    rid = "A1_r%02d" % repeat
    run = Run(rid, "A1", repeat, client.model)
    tools = core.visible_tool_schemas(A1_TOOLS)
    history: List[Dict[str, Any]] = [user_msg(TASK_TEXT)]
    for i in range(MAX_STEPS):
        rec, payload = client.call(history, instructions=A1_SYSTEM, tools=tools, max_output_tokens=MAX_OUT)
        st = run.step("monolithic", A1_SYSTEM, A1_TOOLS, rec, payload)
        if not rec.ok:
            run.error = rec.error
            break
        append_output(history, payload)
        calls = core.ResponsesClient.extract_function_calls(payload)
        if not calls:
            run.final_text = core.ResponsesClient.extract_text(payload)
            break
        for c in calls:
            try:
                args = json.loads(c.get("arguments") or "{}")
            except Exception:  # noqa: BLE001
                args = {}
            out, _ = exec_tool(c.get("name"), args, run, A1_TOOLS, "monolithic")
            history.append({"type": "function_call_output", "call_id": c.get("call_id"), "output": out})
    else:
        run.final_text = st.get("assistant_text", "")
        run.error = "MAX_STEPS_REACHED"
    run.rec = rec
    return run


def run_a2(client: core.ResponsesClient, repeat: int) -> Run:
    rid = "A2_r%02d" % repeat
    run = Run(rid, "A2", repeat, client.model)
    role = "triage"
    history: List[Dict[str, Any]] = [user_msg(TASK_TEXT)]
    transitions: List[Dict[str, Any]] = []
    for i in range(MAX_STEPS):
        names = ROLE_TOOLS[role]
        tools = core.visible_tool_schemas(names) + [TRANSFER_SCHEMA]
        rec, payload = client.call(history, instructions=ROLE_PROMPTS[role], tools=tools, max_output_tokens=MAX_OUT)
        st = run.step(role, ROLE_PROMPTS[role], names + ["transfer_to_agent"], rec, payload)
        if not rec.ok:
            run.error = rec.error
            break
        append_output(history, payload)
        calls = core.ResponsesClient.extract_function_calls(payload)
        if not calls:
            run.final_text = core.ResponsesClient.extract_text(payload)
            break
        transfer_to = None
        for c in calls:
            try:
                args = json.loads(c.get("arguments") or "{}")
            except Exception:  # noqa: BLE001
                args = {}
            out, _ = exec_tool(c.get("name"), args, run, names + ["transfer_to_agent"], role)
            history.append({"type": "function_call_output", "call_id": c.get("call_id"), "output": out})
            if c.get("name") == "transfer_to_agent":
                transfer_to = str(args.get("target") or "")
        if transfer_to:
            transitions.append({"step": st["step"], "from": role, "to": transfer_to,
                                "prefix_before": st["static_prefix_fingerprint"],
                                "at": core.now_iso()})
            if transfer_to not in ROLE_PROMPTS:
                run.error = "BAD_TRANSFER_TARGET: %s" % transfer_to
                break
            role = transfer_to
            # 交接语作为 user 消息补一条，保证轨迹连续
            history.append(user_msg("[harness] 控制权已移交给你（角色=%s）。前序轨迹完整保留。" % role))
    else:
        run.final_text = st.get("assistant_text", "")
        run.error = run.error or "MAX_STEPS_REACHED"
    run.transitions = transitions
    run.final_role = role
    return run


def run_a3(client: core.ResponsesClient, repeat: int) -> Run:
    rid = "A3_r%02d" % repeat
    run = Run(rid, "A3", repeat, client.model)
    tools = core.visible_tool_schemas(A3_TOOLS)
    history: List[Dict[str, Any]] = [user_msg(TASK_TEXT)]
    for i in range(MAX_STEPS):
        rec, payload = client.call(history, instructions=A3_SYSTEM, tools=tools, max_output_tokens=MAX_OUT)
        st = run.step("skill-mode", A3_SYSTEM, A3_TOOLS, rec, payload)
        if not rec.ok:
            run.error = rec.error
            break
        append_output(history, payload)
        calls = core.ResponsesClient.extract_function_calls(payload)
        if not calls:
            run.final_text = core.ResponsesClient.extract_text(payload)
            break
        for c in calls:
            try:
                args = json.loads(c.get("arguments") or "{}")
            except Exception:  # noqa: BLE001
                args = {}
            out, _ = exec_tool(c.get("name"), args, run, A3_TOOLS, "skill-mode")
            history.append({"type": "function_call_output", "call_id": c.get("call_id"), "output": out})
    else:
        run.final_text = st.get("assistant_text", "")
        run.error = "MAX_STEPS_REACHED"
    return run


# ----------------------------------------------------------------------------- 质量评估（确定性规则，可复算）
def evaluate(run: Run) -> Dict[str, Any]:
    text = run.final_text or ""
    parsed = parse_final_json(text) or {}
    # 摘要：优先 JSON.summary，否则取最后一段非 JSON 文本
    summary = parsed.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        cand = re.sub(r"\{.*\}", "", text, flags=re.S).strip()
        lines = [l.strip() for l in cand.splitlines() if l.strip()]
        summary = lines[-1] if lines else ""
    summary = summary.strip()

    rows = [e for e in run.tool_events if e["tool"] == "local_csv_query"]
    has_real_data = any('"nev_sales"' in (e.get("result_preview") or "") and e["ok"] for e in rows)
    used_calc = any(e["tool"] == "calc" and e["ok"] for e in run.tool_events)
    calc_ok = any(e["tool"] == "calc" and e["ok"] and "0.642" in (e.get("result_preview") or "")
                  for e in run.tool_events)
    # 数字：用固定口径从数据集重算（证据可复算）
    ann = core.DATASET_PROVENANCE["annual_nev_sales_wan"]
    v0, v1, n = ann["2021"], ann["2023"], 2
    cagr_true = (v1 / v0) ** (1 / n) - 1
    cagr_model = parsed.get("cagr")
    try:
        cagr_model_f = float(cagr_model) if cagr_model is not None else None
    except Exception:  # noqa: BLE001
        cagr_model_f = None
    cagr_ok = cagr_model_f is not None and abs(cagr_model_f - cagr_true) <= 0.002

    cjk = core.count_chinese_chars(summary)
    ck = core.cjk_len(summary)
    length_ok = 0 < ck <= 120

    # 术语一致性
    full_name = "新能源汽车（NEV）" in summary or "新能源汽车(NEV)" in summary
    uses_nev = "NEV" in summary
    bad_terms = [t for t in ("新能源车", "电动车", "EV车") if t in summary]
    term_ok = (uses_nev and not bad_terms)

    provenance_ok = bool(parsed.get("provenance_noted")) or ("CAAM" in summary and "万辆" in summary)
    # 幻觉数字：摘要里出现 3 位以上、且不属于 {352.1,688.7,949.5,64.2,64.3,2627.5,2686.4,3009.4,120,2021,2022,2023}
    allowed = {"352.1", "688.7", "949.5", "64.2", "64.3", "0.64", "2627.5", "2686.4", "3009.4",
               "2021", "2022", "2023", "120", "1.7", "1.7x", "269", "175"}
    nums = re.findall(r"\d+(?:\.\d+)?", summary)
    dubious = [x for x in nums if x not in allowed and len(x.replace(".", "")) >= 3]

    score = sum([1 if has_real_data else 0, 1 if calc_ok else 0, 1 if cagr_ok else 0,
                 1 if length_ok else 0, 1 if term_ok else 0, 1 if provenance_ok else 0,
                 1 if not dubious else 0])
    core_ok = has_real_data and cagr_ok and length_ok and term_ok and provenance_ok and not dubious

    return {
        "summary": summary,
        "summary_chars_total": len(summary),
        "summary_cjk_chars": cjk,
        "summary_cjk_len": ck,
        "length_ok": length_ok,
        "cagr_true": round(cagr_true, 6),
        "cagr_model": cagr_model_f,
        "cagr_ok": cagr_ok,
        "used_calc_tool": used_calc,
        "calc_returned_true_cagr": calc_ok,
        "used_real_data_tool": has_real_data,
        "term_full_name": full_name,
        "term_uses_nev": uses_nev,
        "term_bad": bad_terms,
        "term_ok": term_ok,
        "provenance_ok": provenance_ok,
        "dubious_numbers": dubious,
        "parsed_final_json": parsed,
        "rule_score_0to7": score,
        "all_core_checks_pass": core_ok,
    }


# ----------------------------------------------------------------------------- 从原始轨迹重算评估
def reevaluate(raw_dir: str, out_path: str) -> Dict[str, Any]:
    """只读 evidence/task5/exp10-1/raw/*.json，用确定性规则重算每题质量分。

    与在线运行完全解耦：任何人重跑本函数都应得到同一个 summary。
    """
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".json"))
    runs = [core.load_json(os.path.join(raw_dir, f)) for f in files]
    for r in runs:
        r["evaluation"] = evaluate_run(r)
    s = summarize(runs)
    s["reevaluated_at"] = core.now_iso()
    s["reevaluated_from"] = [os.path.join("raw", f) for f in files]
    s["evaluator_note"] = EVALUATOR_NOTE
    core.write_json(out_path, s)
    return s


EVALUATOR_NOTE = (
    "质量判定口径（确定性、可复算）："
    "used_real_data_tool = 该 run 里至少一次 local_csv_query 返回了真实的 nev_sales 行（按结果内容判定，"
    "不按调用时传入的 period 大小写判定；数据工具的 period 参数对大小写敏感，这是工具设计缺陷，已在笔记中说明）；"
    "calc_returned_true_cagr = 至少一次 calc 返回含 0.642 的结果；cagr_ok = 最终 JSON 的 cagr 与 (949.5/352.1)**0.5-1 相差 <= 0.002；"
    "length_ok = 摘要按 cjk_len 口径 <=120；term_ok = 用了 NEV 且没有出现 新能源车/电动车/EV车；"
    "dubious_numbers = 摘要里出现了不在白名单内的 3 位以上数字；rule_score_0to7 = 七项各 1 分的和；"
    "all_core_checks_pass = 数据工具 + CAGR + 长度 + 术语 + 口径标注 + 无可疑数字 全部通过。"
)


def evaluate_run(r: Dict[str, Any]) -> Dict[str, Any]:
    """对一条已保存的 run 记录重算评估（纯函数，不碰网络）。"""
    text = r.get("final_text") or ""
    parsed = parse_final_json(text) or {}
    summary = parsed.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        cand = re.sub(r"\{.*\}", "", text, flags=re.S).strip()
        lines = [l.strip() for l in cand.splitlines() if l.strip()]
        summary = lines[-1] if lines else ""
    summary = summary.strip()

    rows = [e for e in (r.get("tool_events") or []) if e.get("tool") == "local_csv_query"]
    has_real_data = any('"nev_sales"' in (e.get("result_preview") or "") and e.get("ok") for e in rows)
    used_calc = any(e.get("tool") == "calc" and e.get("ok") for e in (r.get("tool_events") or []))
    calc_ok = any(e.get("tool") == "calc" and e.get("ok") and "0.642" in (e.get("result_preview") or "")
                  for e in (r.get("tool_events") or []))

    ann = core.DATASET_PROVENANCE["annual_nev_sales_wan"]
    v0, v1, n = ann["2021"], ann["2023"], 2
    cagr_true = (v1 / v0) ** (1 / n) - 1
    cagr_model = parsed.get("cagr")
    try:
        cagr_model_f = float(cagr_model) if cagr_model is not None else None
    except Exception:  # noqa: BLE001
        cagr_model_f = None
    cagr_ok = cagr_model_f is not None and abs(cagr_model_f - cagr_true) <= 0.002

    cjk = core.count_chinese_chars(summary)
    ck = core.cjk_len(summary)
    length_ok = 0 < ck <= 120

    full_name = "新能源汽车（NEV）" in summary or "新能源汽车(NEV)" in summary
    uses_nev = "NEV" in summary
    bad_terms = [t for t in ("新能源车", "电动车", "EV车") if t in summary]
    term_ok = (uses_nev and not bad_terms)

    provenance_ok = bool(parsed.get("provenance_noted")) or ("CAAM" in summary and "万辆" in summary)
    allowed = {"352.1", "688.7", "949.5", "64.2", "64.22", "64.3", "0.64", "2627.5", "2686.4", "3009.4",
               "2021", "2022", "2023", "120", "1.7", "2021-2023", "2021—2023"}
    nums = re.findall(r"\d+(?:\.\d+)?", summary)
    dubious = [x for x in nums if x not in allowed and len(x.replace(".", "")) >= 3]

    score = sum([has_real_data, calc_ok, cagr_ok, length_ok, term_ok, provenance_ok, not dubious])
    return {
        "summary": summary,
        "summary_chars_total": len(summary),
        "summary_cjk_chars": cjk,
        "summary_cjk_len": ck,
        "length_ok": length_ok,
        "cagr_true": round(cagr_true, 6),
        "cagr_model": cagr_model_f,
        "cagr_ok": cagr_ok,
        "used_calc_tool": used_calc,
        "calc_returned_true_cagr": calc_ok,
        "used_real_data_tool": has_real_data,
        "term_full_name": full_name,
        "term_uses_nev": uses_nev,
        "term_bad": bad_terms,
        "term_ok": term_ok,
        "provenance_ok": provenance_ok,
        "dubious_numbers": dubious,
        "parsed_final_json": parsed,
        "rule_score_0to7": score,
        "all_core_checks_pass": bool(has_real_data and calc_ok and cagr_ok and length_ok
                                     and term_ok and provenance_ok and not dubious),
    }


# ----------------------------------------------------------------------------- 汇总
def summarize(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    modes = ["A1", "A2", "A3"]
    out: Dict[str, Any] = {"generated_at": core.now_iso(), "model": core.MODEL,
                           "dataset_provenance": core.DATASET_PROVENANCE,
                           "n_runs": len(runs), "modes": {}}
    ann = core.DATASET_PROVENANCE["annual_nev_sales_wan"]
    out["ground_truth"] = {
        "v0_2021_wan": ann["2021"], "v1_2023_wan": ann["2023"], "n_years": 2,
        "cagr": round((ann["2023"] / ann["2021"]) ** 0.5 - 1, 6),
        "cagr_pct_1dp": round(((ann["2023"] / ann["2021"]) ** 0.5 - 1) * 100, 1),
    }
    for m in modes:
        rs = [r for r in runs if r["mode"] == m]
        if not rs:
            continue
        ok = [r for r in rs if not r.get("error")]
        def sum_step(k):
            return sum(s.get(k, 0) or 0 for r in rs for s in r["steps"])
        def mean(xs):
            return round(sum(xs) / len(xs), 4) if xs else None
        stage_tokens = []
        for r in rs:
            for s in r["steps"]:
                stage_tokens.append({"run_id": r["run_id"], "step": s["step"], "role": s["role"],
                                     "input_tokens": s["input_tokens"], "output_tokens": s["output_tokens"],
                                     "wall_s": s["wall_s"], "attempts": s["attempts"],
                                     "instructions_chars": s["instructions_chars"],
                                     "visible_tools": s["visible_tools"],
                                     "prefix_fp": s["static_prefix_fingerprint"]})
        prefixes = []
        for r in rs:
            for s in r["steps"]:
                if s["static_prefix_fingerprint"] not in prefixes:
                    prefixes.append(s["static_prefix_fingerprint"])
        ev = [evaluate_dict(r.get("evaluation") or {}) for r in rs]
        out["modes"][m] = {
            "n_runs": len(rs),
            "n_runs_without_error": len(ok),
            "errors": [{"run_id": r["run_id"], "error": r.get("error")} for r in rs if r.get("error")],
            "total_input_tokens": sum_step("input_tokens"),
            "total_output_tokens": sum_step("output_tokens"),
            "total_tokens": sum_step("input_tokens") + sum_step("output_tokens"),
            "mean_input_tokens_per_run": mean([sum(s["input_tokens"] for s in r["steps"]) for r in rs]),
            "mean_output_tokens_per_run": mean([sum(s["output_tokens"] for s in r["steps"]) for r in rs]),
            "mean_tokens_per_run": mean([sum(s["input_tokens"] + s["output_tokens"] for s in r["steps"]) for r in rs]),
            "mean_wall_s_per_run": mean([sum(s["wall_s"] for s in r["steps"]) for r in rs]),
            "mean_model_calls_per_run": mean([len(r["steps"]) for r in rs]),
            "failed_attempts_total": sum((s.get("attempts", 1) - 1) for r in rs for s in r["steps"]),
            "failed_calls_total": sum(0 if s["ok"] else 1 for r in rs for s in r["steps"]),
            "n_distinct_static_prefixes": len(prefixes),
            "static_prefix_fingerprints": prefixes,
            "cache_read_tokens_total": sum_step("cached_tokens"),
            "cache_write_tokens_total": sum_step("cache_write_tokens"),
            "mean_instructions_chars": mean([len(set()) or 0 for _ in [0]]) if False else
                mean([s["instructions_chars"] for r in rs for s in r["steps"]]),
            "stage_tokens": stage_tokens,
            "quality": {
                "n_evaluated": len(ev),
                "all_core_checks_pass": sum(1 for e in ev if e.get("all_core_checks_pass")),
                "mean_rule_score_0to7": mean([e.get("rule_score_0to7", 0) for e in ev]),
                "length_ok": sum(1 for e in ev if e.get("length_ok")),
                "cagr_ok": sum(1 for e in ev if e.get("cagr_ok")),
                "term_ok": sum(1 for e in ev if e.get("term_ok")),
                "provenance_ok": sum(1 for e in ev if e.get("provenance_ok")),
                "used_real_data_tool": sum(1 for e in ev if e.get("used_real_data_tool")),
                "calc_returned_true_cagr": sum(1 for e in ev if e.get("calc_returned_true_cagr")),
                "dubious_number_rate": mean([1 if e.get("dubious_numbers") else 0 for e in ev]),
                "summary_cjk_len_mean": mean([e.get("summary_cjk_len", 0) for e in ev]),
            },
            "per_run": [{"run_id": r["run_id"], "error": r.get("error"),
                         "tokens": sum(s["input_tokens"] + s["output_tokens"] for s in r["steps"]),
                         "wall_s": round(sum(s["wall_s"] for s in r["steps"]), 3),
                         "steps": len(r["steps"]),
                         "evaluation": r.get("evaluation")} for r in rs],
        }
        if m == "A2":
            out["modes"][m]["n_role_transitions_mean"] = mean([len(r.get("transitions") or []) for r in rs])
            out["modes"][m]["final_roles"] = [r.get("final_role") for r in rs]
        if m == "A3":
            sk = []
            for r in rs:
                sk.append(sorted({e["arguments"].get("name") for e in r["tool_events"]
                                  if e["tool"] == "load_skill" and isinstance(e.get("arguments"), dict)}))
            out["modes"][m]["skills_loaded_per_run"] = sk
            out["modes"][m]["skill_load_chars_total"] = sum(
                e.get("skill_chars", 0) for r in rs for e in r["tool_events"] if e["tool"] == "load_skill")
    # 对比
    if all(m in out["modes"] for m in modes):
        base = out["modes"]["A1"]["mean_tokens_per_run"] or 1
        out["comparison"] = {
            "token_ratio_vs_A1": {m: round((out["modes"][m]["mean_tokens_per_run"] or 0) / base, 3) for m in modes},
            "latency_ratio_vs_A1": {m: round((out["modes"][m]["mean_wall_s_per_run"] or 0) /
                                             (out["modes"]["A1"]["mean_wall_s_per_run"] or 1), 3) for m in modes},
            "quality_pass_rate": {m: round(out["modes"][m]["quality"]["all_core_checks_pass"] /
                                           max(1, out["modes"][m]["quality"]["n_evaluated"]), 3) for m in modes},
            "cost_per_passing_run_tokens": {
                m: (None if out["modes"][m]["quality"]["all_core_checks_pass"] == 0 else
                    round((out["modes"][m]["total_tokens"]) / out["modes"][m]["quality"]["all_core_checks_pass"], 1))
                for m in modes},
            "n_distinct_static_prefixes": {m: out["modes"][m]["n_distinct_static_prefixes"] for m in modes},
        }
    return out


def evaluate_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return d or {}


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--modes", default="A1,A2,A3")
    ap.add_argument("--model", default=core.MODEL)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--append", action="store_true", help="保留已有 runs.jsonl 并追加")
    ap.add_argument("--reevaluate", action="store_true",
                    help="只读 raw/*.json 用确定性规则重算 summary.json（不发起任何模型调用）")
    args = ap.parse_args()

    core.ensure_dir(RAW_DIR)
    if args.reevaluate:
        s = reevaluate(RAW_DIR, os.path.join(EXP_DIR, "summary.json"))
        print("reevaluated from %d raw files" % len(s["reevaluated_from"]))
        for m, d in s["modes"].items():
            print("%s: runs=%d tokens/run=%s wall/run=%ss prefixes=%d quality_pass=%d/%d score=%s" % (
                m, d["n_runs"], d["mean_tokens_per_run"], d["mean_wall_s_per_run"],
                d["n_distinct_static_prefixes"], d["quality"]["all_core_checks_pass"],
                d["quality"]["n_evaluated"], d["quality"]["mean_rule_score_0to7"]))
        print("comparison:", json.dumps(s.get("comparison"), ensure_ascii=False))
        return 0
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    existing: List[Dict[str, Any]] = []
    if args.append and os.path.exists(RUNS_PATH):
        with open(RUNS_PATH, "r", encoding="utf-8") as f:
            existing = [json.loads(l) for l in f if l.strip()]

    if not args.summarize_only:
        client = core.ResponsesClient(model=args.model)
        done: List[Dict[str, Any]] = list(existing)
        for repeat in range(1, args.repeats + 1):
            for m in modes:
                # 幂等：已有同名 run 就跳过
                rid = "%s_r%02d" % (m, repeat)
                if any(r["run_id"] == rid for r in done):
                    print("[skip] %s already recorded" % rid, flush=True)
                    continue
                print("=== run %s ===" % rid, flush=True)
                t0 = time.time()
                if m == "A1":
                    run = run_a1(client, repeat)
                elif m == "A2":
                    run = run_a2(client, repeat)
                else:
                    run = run_a3(client, repeat)
                run.evaluation = evaluate(run)  # type: ignore[attr-defined]
                obj = run.to_json()
                obj["evaluation"] = run.evaluation          # type: ignore[attr-defined]
                obj["transitions"] = getattr(run, "transitions", [])
                obj["final_role"] = getattr(run, "final_role", None)
                obj["elapsed_s"] = round(time.time() - t0, 2)
                core.write_json(os.path.join(RAW_DIR, "%s.json" % rid), obj)
                done.append(obj)
                print("    tokens=%d wall=%.1fs steps=%d err=%s score=%d/%d" % (
                    sum(s["input_tokens"] + s["output_tokens"] for s in obj["steps"]),
                    obj["elapsed_s"], obj["n_steps"], obj["error"],
                    obj["evaluation"]["rule_score_0to7"], 7), flush=True)
                with open(RUNS_PATH, "w", encoding="utf-8") as f:
                    for r in done:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("[client stats]", json.dumps(client.stats(), ensure_ascii=False))

    runs = []
    if os.path.exists(RUNS_PATH):
        with open(RUNS_PATH, "r", encoding="utf-8") as f:
            runs = [json.loads(l) for l in f if l.strip()]
    if not runs:
        print("no runs recorded", file=sys.stderr)
        return 1
    s = summarize(runs)
    s["proxy_stats"] = runs and None
    core.write_json(os.path.join(EXP_DIR, "summary.json"), s)
    print("\n=== summary ===")
    for m, d in s["modes"].items():
        print("%s: runs=%d tokens/run=%s wall/run=%ss calls/run=%s prefixes=%d quality_pass=%d/%d" % (
            m, d["n_runs"], d["mean_tokens_per_run"], d["mean_wall_s_per_run"],
            d["mean_model_calls_per_run"], d["n_distinct_static_prefixes"],
            d["quality"]["all_core_checks_pass"], d["quality"]["n_evaluated"]))
    print("written:", os.path.join(EXP_DIR, "summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())