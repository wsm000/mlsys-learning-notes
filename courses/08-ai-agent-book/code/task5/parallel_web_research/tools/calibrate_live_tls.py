"""
L1 标定：真实 HTTPS 抓一条会额外花掉多少时间（TLS/代理惩罚）
==============================================================

这一项**不是**协调机制实验的结果，而是"本机用本地镜像回放页面"这件事的代价标定：
  * verify_on ：默认证书校验（本机必然失败，记录失败耗时）
  * verify_off：关闭校验（能拿到真实页面，但有 TLS 重连开销）

结论只用来解释笔记里的一句话：L0（本地回放）把真实链路的传输与 TLS 成本归零，
因此并行/串行对比里的加速比只刻画**代理+抽取**这一段并行收益；
如果站点本身是秒级慢站，加速比的分母会变大（真实世界通常更接近 2-4 倍而不是 20 倍）。

用法：python tools/calibrate_live_tls.py --output ../../../evidence/task5/exp10-4/live_tls_calibration.json
"""

from __future__ import annotations

import argparse
import json
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36"
URLS = [
    "https://profiles.stanford.edu/andrew-ng",
    "https://provost.stanford.edu/",
    "https://www.cs.stanford.edu/people/faculty",
    "https://www.gsb.stanford.edu/faculty-research/faculty",
]


def attempt(url: str, verify: bool, timeout: float) -> dict:
    ctx = ssl.create_default_context()
    if not verify:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            n = len(resp.read())
        return {"ok": True, "status": 200, "bytes": n, "seconds": round(time.monotonic() - t0, 3)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                "seconds": round(time.monotonic() - t0, 3)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()
    rows = []
    for url in URLS:
        on = attempt(url, verify=True, timeout=args.timeout)
        off = attempt(url, verify=False, timeout=args.timeout)
        row = {"url": url, "verify_on": on, "verify_off": off}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "live_tls_calibration",
        "note": "真实 HTTPS 抓取的传输/TLS 成本；不用于协调机制结论",
        "rows": rows,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("written", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
