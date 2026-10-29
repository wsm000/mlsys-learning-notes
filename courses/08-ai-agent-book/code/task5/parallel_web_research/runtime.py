"""
本地站点运行时（替代 Playwright 浏览器）
======================================

上游每个 worker 拥有**独立 Playwright Chromium context**，用真实浏览器访问真实站点。
本机没有 Playwright，因此把"浏览器上下文"换成一个**可替换的抓取运行时**，语义逐一对应：

| 上游（浏览器）                       | 本机（PageRuntime）                                  |
|-------------------------------------|------------------------------------------------------|
| BrowserPool.new_context()           | PageRuntime.new_context() -> FetchContext（计数+1）  |
| page.goto(url) 不可取消原生导航      | ctx.fetch(site) 在线程中跑，可被协程放弃             |
| page.locator("body").inner_text()   | Site.fetcher() -> visible_text（html_to_visible_text）|
| context.close()                     | ctx.close()（幂等，计数+1）                          |

**关键语义：worker 在安全点取消抓取时，后台线程仍在跑完那次 fetch。**
这正是"取消是协作式的、资源必须显式关闭"的真实演示：没有 `async with`，
线程与 socket 就是泄漏的；有了它，审计里的"创建数 = 关闭数"才有意义。

抓取等级（隔离"抓取能力"与"协调机制"，见笔记）：

- L0 local_mirror  : 本地 http.server 提供**真实抓取下来的页面**，TLS 惩罚 0
- L1 live_unverified: 真实 HTTPS（本机证书链过不了，只能关校验），TLS 实测惩罚
- L2 simulated_network: 本地站点 + 本地服务端按参数注入的延迟（网络慢的模拟）
- L3 offline_fixture: 不起服务，直接读 corpus 文件 + 固定延迟（纯协调实验）
"""

from __future__ import annotations

import asyncio
import http.server
import json
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from html_text import html_to_visible_text

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


# "永久挂起"的站点用它阻塞抓取线程（进程退出时才释放）
_HANG_EVENT = threading.Event()


class FetchError(RuntimeError):
    """抓取失败：超时 / 连接失败 / HTTP 错误 / TLS 错误。"""

    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


@dataclass
class Site:
    name: str
    college: str
    url: str
    level: str = "local_mirror"
    expected: str = "not_found"          # found | not_found | error —— 半合成真值
    latency_seconds: float = 0.0         # L2：服务端注入延迟
    fail: Optional[str] = None           # L2/L3：simulated 500 / hang / malformed

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "college": self.college,
            "url": self.url,
            "level": self.level,
            "expected": self.expected,
            "latency_seconds": self.latency_seconds,
            "fail": self.fail,
        }


# —— 抓取实现 ——

def _http_get(url: str, timeout: float, verify_tls: bool) -> Tuple[int, str, int]:
    ctx = None
    if url.startswith("https://"):
        ctx = ssl.create_default_context()
        if not verify_tls:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        raw = resp.read()
        return resp.status, raw.decode("utf-8", "replace"), len(raw)


def fetch_site(site: Site, timeout: float, corpus_dir: Optional[Path] = None) -> Dict[str, object]:
    """同步抓取（在线程里跑）。返回观测记录。"""
    started = time.monotonic()
    if site.fail == "hang":
        # 真实挂起：线程卡在一个永远不会被设置的 Event 上，直到进程退出。
        # 这里刻意**不用 sleep**：sleep 到点会自己醒来，看不出"取消/超时是协作式的"；
        # 卡死才逼出超时路径，也正好演示"被放弃的抓取线程不会自己结束"。
        _HANG_EVENT.wait()
        raise FetchError("Timeout", f"site did not respond within {timeout}s")
    if site.level == "offline_fixture":
        if site.fail == "malformed":
            raise FetchError("MalformedPage", "fixture page missing rendered-text container")
        path = Path(corpus_dir or Path("corpus")) / f"{site.name}.html"
        raw = path.read_text(encoding="utf-8")
        if site.latency_seconds:
            time.sleep(site.latency_seconds)
        return {
            "url": site.url,
            "final_url": site.url,
            "http_status": 200,
            "bytes": len(raw.encode("utf-8")),
            "text": html_to_visible_text(raw),
            "seconds": round(time.monotonic() - started, 3),
            "level": site.level,
        }
    try:
        status, raw, nbytes = _http_get(site.url, timeout, verify_tls=(site.level != "live_unverified"))
        if site.fail == "malformed":
            raise FetchError("MalformedPage", "page has no rendered-text container")
        text = html_to_visible_text(raw)
        return {
            "url": site.url,
            "final_url": site.url,
            "http_status": status,
            "bytes": nbytes,
            "text": text,
            "seconds": round(time.monotonic() - started, 3),
            "level": site.level,
        }
    except FetchError:
        raise
    except socket.timeout as exc:
        raise FetchError("Timeout", str(exc)) from exc
    except urllib.error.HTTPError as exc:
        raise FetchError(f"HTTP{exc.code}", str(exc)) from exc
    except ssl.SSLError as exc:
        raise FetchError("TLSError", str(exc)) from exc
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, socket.timeout):
            raise FetchError("Timeout", str(reason)) from exc
        if isinstance(reason, ssl.SSLError):
            raise FetchError("TLSError", str(reason)) from exc
        raise FetchError("ConnectionError", str(reason)) from exc
    except OSError as exc:
        raise FetchError("ConnectionError", str(exc)) from exc


# —— 可取消的 "page" ——

class FetchContext:
    """一个 worker 的独立抓取上下文，对应上游一个 Chromium browser context。"""

    def __init__(self, runtime: "PageRuntime", owner: str):
        self.runtime = runtime
        self.owner = owner
        self.created_at = time.monotonic()
        self.closed = False
        self.fetch_abandoned = False

    async def fetch(self, site: Site, timeout: float) -> Dict[str, object]:
        loop = asyncio.get_running_loop()

        def run() -> Dict[str, object]:
            return fetch_site(site, timeout, corpus_dir=self.runtime.corpus_dir)

        future = loop.run_in_executor(self.runtime.executor, run)
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        except asyncio.TimeoutError as exc:
            # wait_for 超时：放弃等待，但线程仍会跑完（真实世界的取消即如此）
            self.fetch_abandoned = True
            raise FetchError("Timeout", f"exceeded {timeout}s") from exc
        except asyncio.CancelledError:
            self.fetch_abandoned = True
            raise

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.runtime.mark_closed(self)

    def to_dict(self) -> Dict[str, object]:
        return {
            "owner": self.owner,
            "closed": self.closed,
            "fetch_abandoned": self.fetch_abandoned,
            "age_seconds": round(time.monotonic() - self.created_at, 3),
        }


class PageRuntime:
    """浏览器池的替身：只负责创建/关闭上下文并记账。"""

    def __init__(self, corpus_dir: Optional[Path] = None, max_workers: int = 32, label: str = "local"):
        from concurrent.futures import ThreadPoolExecutor

        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"fetch-{label}")
        self.contexts_created = 0
        self.contexts_closed = 0
        self.corpus_dir = corpus_dir
        self.label = label
        self._open: List[FetchContext] = []

    # 子类可以替换成别的上下文实现（测试里用"注入延迟的假上下文"）
    context_class = FetchContext

    def new_context(self, owner: str) -> FetchContext:
        ctx = type(self).context_class(self, owner)
        self.contexts_created += 1
        self._open.append(ctx)
        return ctx

    def mark_closed(self, ctx: FetchContext) -> None:
        self.contexts_closed += 1
        if ctx in self._open:
            self._open.remove(ctx)

    @property
    def open_contexts(self) -> int:
        return len(self._open)

    def audit(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "contexts_created": self.contexts_created,
            "contexts_closed": self.contexts_closed,
            "contexts_leaked": self.contexts_created - self.contexts_closed,
            "audit_ok": self.contexts_created == self.contexts_closed,
            "open_now": self.open_contexts,
        }

    async def shutdown(self, close_leaked: bool = True) -> Dict[str, object]:
        leaked = list(self._open)
        if close_leaked:
            for ctx in leaked:
                await ctx.close()
        await asyncio.to_thread(self.executor.shutdown, True)
        return self.audit()


# —— 本地镜像站点服务器 ——

class _MirrorHandler(http.server.BaseHTTPRequestHandler):
    corpus_dir: Path
    routes: Dict[str, Dict[str, object]] = {}

    def log_message(self, fmt, *args):  # 静默
        pass

    def do_GET(self):  # noqa: N802
        name = self.path.lstrip("/").split("?")[0]
        if name.endswith(".html"):
            name = name[:-5]
        route = self.routes.get(name)
        if route is None:
            self.send_error(404, "unknown mirror path")
            return
        delay = float(route.get("delay") or 0.0)
        if delay:
            time.sleep(delay)
        if route.get("fail") == "http500":
            self.send_error(500, "injected upstream failure")
            return
        path = self.corpus_dir / f"{route['file']}.html"
        if not path.exists():
            self.send_error(404, "mirror file missing")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class MirrorServer:
    """在 127.0.0.1 上把真实抓下来的页面目录当站点提供。

    routes: {站点名: {"file": corpus 文件名, "delay": 秒, "fail": "http500"|None}}
    """

    def __init__(self, corpus_dir: Path, routes: Optional[Dict[str, Dict[str, object]]] = None):
        self.corpus_dir = Path(corpus_dir)
        handler = type("Handler", (_MirrorHandler,), {
            "corpus_dir": self.corpus_dir,
            "routes": dict(routes or {}),
        })
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="mirror-http", daemon=True)

    def __enter__(self) -> "MirrorServer":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def url_for(self, name: str) -> str:
        return f"http://127.0.0.1:{self.port}/{name}.html"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
