"""
HTML -> 可见文本抽取（纯标准库）
================================

上游用 Playwright 的 `page.locator("body").inner_text()` 拿到"渲染后的可见文本"。
本机没有 Playwright（且不联网安装），所以用 html.parser 做一个**近似**的可见文本抽取：

- 丢掉 script / style / noscript；
- 块级标签处断行；
- 归一化空白、把常见的智能标点降级为 ASCII（保证不同控制台/终端下稳定）。

**边界（必须写进笔记）**：这不是浏览器渲染引擎。JS 动态注入的内容拿不到，
这正是 HAI 的 /people 页面只抽出 12 个字符的原因——那台机器上的页面内容靠 JS 生成，
所以它的抽取结果是"看起来成功、其实空"，属于**结构/渲染差异**而非网络失败。
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import List

_BLOCK_TAGS = {
    "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr", "section",
    "article", "header", "footer", "ul", "ol", "table", "td", "th", "form", "nav",
    "address", "blockquote", "pre", "figure", "figcaption", "dt", "dd", "hr",
}
_SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}

_PUNCT_MAP = {
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2014": "-", "\u2013": "-", "\u2026": "...", "\u00a0": " ",
    "\u00a9": "(c)", "\u00ae": "(R)", "\u2122": "(TM)",
}


class _TextCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.chunks: List[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self.skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag in _BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_data(self, data):
        if self.skip_depth == 0:
            self.chunks.append(data)


def html_to_visible_text(html: str) -> str:
    parser = _TextCollector()
    parser.feed(html)
    parser.close()
    text = "".join(parser.chunks)
    for bad, good in _PUNCT_MAP.items():
        text = text.replace(bad, good)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    # 只保留可打印 ASCII + 换行：真实站点的中文/emoji 在 GBK 控制台上会炸，
    # 而目标抽取只依赖拉丁姓名，因此统一降级为 '?'。
    text = re.sub(r"[^\x20-\x7e\n]", "?", text)
    return text.strip()
