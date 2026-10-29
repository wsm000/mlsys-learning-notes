"""
语料抓取脚本（一次性的真实抓取，结果已落在 corpus/）
====================================================

为什么默认关闭 TLS 校验：本机的出站请求都会经过一个会重写证书链的中转，
`ssl.create_default_context()` 对 https://profiles.stanford.edu 会直接报
`CERTIFICATE_VERIFY_FAILED: self signed certificate in certificate chain`。
关掉校验后可以拿到**真实页面**，但这条链路的抓取本身不可信，所以：
  - corpus/ 里的正文只用于"协调机制"实验（L0：本地 http.server 回放）；
  - 真实联网抓取的手感（TLS 重连耗时）单独在 L1 标定里测量，不当成实验结果。

用法：
    python tools/fetch_corpus.py            # 重新抓一遍（需要能出网）
    python tools/fetch_corpus.py --dry-run  # 只打印计划
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from html_text import html_to_visible_text  # noqa: E402
from sites import REAL_PAGES  # noqa: E402

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36"


def fetch(url: str, timeout: float) -> tuple[int, bytes]:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return resp.status, resp.read()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    corpus = ROOT / "corpus"
    corpus.mkdir(exist_ok=True)
    summary = {}
    for name, meta in REAL_PAGES.items():
        url = str(meta["url"])
        if args.dry_run:
            print("would fetch", name, url)
            continue
        started = time.monotonic()
        try:
            status, raw = fetch(url, args.timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            summary[name] = {"url": url, "error": f"{type(exc).__name__}: {exc}"}
            continue
        text = html_to_visible_text(raw.decode("utf-8", "replace"))
        html = ("<!DOCTYPE html><html><head><meta charset='utf-8'><title>{}"
                "</title></head><body><pre id='rendered-text'>{}</pre></body></html>").format(name, text)
        (corpus / f"{name}.html").write_text(html, encoding="utf-8")
        summary[name] = {
            "url": url,
            "http_status": status,
            "html_bytes": len(raw),
            "text_chars": len(text),
            "fetch_seconds": round(time.monotonic() - started, 3),
        }
        print(f"OK   {name}: {len(raw)}B -> {len(text)} chars in {summary[name]['fetch_seconds']}s")
    if not args.dry_run:
        (corpus / "fetch_meta.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
