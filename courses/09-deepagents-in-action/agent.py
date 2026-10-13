"""Real model tool calling through Deep Agents; configure credentials locally."""
import argparse
import json
import os
import getpass
from datetime import datetime, timezone
from pathlib import Path

from tools import analyze_price_volume


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default="分析 DEMO 最近5个交易日上涨是否伴随成交量放大，列出依据和局限。")
    args = parser.parse_args()
    if not args.question.strip() or len(args.question) > 4000:
        parser.error("question must contain 1–4000 characters")
    from deepagents import create_deep_agent
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("QUANTY_API_KEY") or os.environ.get("AMD_API_KEY")
    if not api_key:
        api_key = getpass.getpass("AMD API Key (hidden, not saved): ").strip()
    if not api_key:
        parser.error("AMD API Key is required")
    model_name = os.environ.get("QUANTY_MODEL", "MiMo-V2.6-Flash")
    model = ChatOpenAI(model=model_name,
                       api_key=api_key,
                       base_url=os.environ.get("QUANTY_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1"),
                       temperature=0, timeout=90, max_retries=0)
    agent = create_deep_agent(
        model=model, tools=[analyze_price_volume],
        system_prompt=("你是量价研究助理。量价比较调用 analyze_price_volume，以工具返回为依据。"
                       "例外：用户要求的window超出1–5时，不得调用较小window代替，不得宣称完成。"
                       "应说明当前仅10条数据，比较最近N条与此前N条需要2N条，工具目前仅支持1–5。"
                       "询问用户是否接受5条演示，获得明确同意前不调用替代方案，不修改工具代码。"
                       "只有 DEMO 模拟数据，不可冒充真实行情。不支持的标的应明确告知。"
                       "给出区间、涨幅、前后日均成交量、比值及局限。比较结果不能证明因果或预测。"
                       "简单查询无需规划、委派或文件操作。"))
    result = agent.invoke({"messages": [{"role": "user", "content": args.question}]})
    output = Path(__file__).parent / "outputs"
    output.mkdir(exist_ok=True)
    trace = output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    messages = [m.model_dump(mode="json") for m in result["messages"]]
    requests = {call["id"]: call for m in result["messages"]
                for call in getattr(m, "tool_calls", [])
                if call["name"] == "analyze_price_volume"}
    returns = [m for m in result["messages"]
               if getattr(m, "type", None) == "tool"
               and getattr(m, "tool_call_id", None) in requests]
    successful = [m for m in returns if getattr(m, "status", "success") == "success"]
    verification = {"business_tool_requests": len(requests),
                    "matching_returns": len(returns),
                    "successful_returns": len(successful),
                    "tool_chain_observed": bool(successful)}
    trace.write_text(json.dumps({"model": model_name,
                                 "kind": "message trajectory, not full execution spans",
                                 "verification": verification,
                                 "messages": messages}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result["messages"][-1].content)
    print(f"Message trajectory: {trace}")
    print(f"Tool-chain check: {verification}")
    if not successful:
        print("No successful tool chain observed. This can be correct for clarification/refusal; evaluate the task policy separately.")


if __name__ == "__main__":
    main()
