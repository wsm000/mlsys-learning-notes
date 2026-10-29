"""
实验 10-4 的站点/语料配置
=========================

语料来源（全部是一次**真实抓取**后落地的文件，corpus/*.html 里保存的是抓下来的可见文本）：

- Stanford 各院系页面：用 urllib + 关闭 TLS 校验抓取（本机在代理后面，证书链自签名，
  校验必然失败，见 corpus/fetch_meta.json 与 tools/fetch_corpus.py）
- 真实"同名不同人"对照页 profiles.stanford.edu/andrew-ng：Andrew Ng 在 Stanford 的职业
  办公室目录里是 "Senior Director of Finance, Vice Provost and Dean of Research"，
  与世人所知的 AI 教授 Andrew Ng **不是同一职位**，也不是同一个人。

页面标签（隔离"抓取能力"与"协调机制"）：

- real_gateway     真实网关页（在别人的站点上问同一个目标）——记录"跨站检索失败有多真实"
- real_literal     真实负例（页面上压根没有目标串）
- real_counterexample 真实同名不同人陷阱页
- synthetic_slow   半合成：真实页正文 + 服务端注入延迟，让"晚到命中"可复现
- synthetic_fail   半合成：真实页正文 + 注入 500 / hang / 结构损坏，验证错误隔离

目标（provost_status）的语义口径是"Stanford University 的现任教务长"。由于没有任何作业页面
会直接陈述这一点（这正是"同名/同角色对象一致性"问题本身），本实验给出一个**真实网关页**：
一个只描述教务长办公室、不含任何具体姓名的页面（网关 A）。命中它之后协调者立刻结算，
同时把"还需要去权威来源确认"写进结论——**本实验不宣称找到了教务长是谁**。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from runtime import Site

HERE = Path(__file__).resolve().parent
CORPUS_DIR = HERE / "corpus"

REAL_PAGES: Dict[str, Dict[str, object]] = {
    "stanford-profiles-andrew-ng": {
        "url": "https://profiles.stanford.edu/andrew-ng",
        "college": "Stanford Profiles / Office of the Vice Provost and Dean of Research",
        "role": "real_counterexample",
        "note": "Andrew Ng 在此页面上的职位是 Senior Director of Finance, Vice Provost and Dean of Research",
    },
    "stanford-provost-office": {
        "url": "https://provost.stanford.edu/",
        "college": "Office of the Provost, Stanford University",
        "role": "real_gateway_candidate",
        "note": "教务长办公室首页：只讲办公室与职责，不含现任者姓名",
    },
    "stanford-cs-faculty": {
        "url": "https://www.cs.stanford.edu/people/faculty",
        "college": "School of Engineering / Computer Science",
        "role": "real_literal_negative",
        "note": "真实教师目录，页面里没有目标串",
    },
    "stanford-gsb-faculty": {
        "url": "https://www.gsb.stanford.edu/faculty-research/faculty",
        "college": "Graduate School of Business",
        "role": "real_literal_negative",
        "note": "含 Coley Andrews 等近名，但无 Andrew Ng",
    },
    "stanford-doeer-faculty": {
        "url": "https://sustainability.stanford.edu/people/faculty",
        "college": "Doerr School of Sustainability",
        "role": "real_literal_negative",
        "note": "真实教师目录，长页面",
    },
    "stanford-law-directory": {
        "url": "https://law.stanford.edu/directory/",
        "college": "Stanford Law School",
        "role": "real_literal_negative",
        "note": "真实教师目录",
    },
    "stanford-education-faculty": {
        "url": "https://ed.stanford.edu/faculty",
        "college": "Graduate School of Education",
        "role": "real_literal_negative",
        "note": "真实教师目录",
    },
    "stanford-medicine-profiles": {
        "url": "https://med.stanford.edu/profiles/browse",
        "college": "School of Medicine",
        "role": "real_literal_negative",
        "note": "真实教师目录",
    },
    "stanford-hai-people": {
        "url": "https://hai.stanford.edu/people",
        "college": "Stanford HAI",
        "role": "real_structure_negative",
        "note": "内容由 JS 生成，无浏览器渲染时只抽到 12 个字符——看起来成功、其实空",
    },
}

QUERY = "Provost of Stanford University"
CRITERION = "the current Provost of Stanford University"

NOTES = [
    "页面正文来自一次真实抓取（corpus/*.html），传输层在 L0 是本机 http.server。",
    "目标 'Provost of Stanford University' 的现任者没有任何一个作业页面直接陈述；本实验只验证协调机制。",
    "真实同名陷阱：profiles.stanford.edu/andrew-ng 上的 Andrew Ng 是财务/研究办公室的高级主管，不是 AI 教授。",
    "半合成站点（--slow / --fail）的正文仍是真实抓取的页面，只有延迟/故障是注入的。",
]


@dataclass
class ExperimentConfig:
    sites: List[Site]
    query: str
    criterion: str
    aliases: tuple
    corpus_dir: Path
    notes: List[str]


def _mirror_key(site_name: str) -> str:
    """半合成站点名 -> 它复用的真实页面（镜像路径与复用文件都用这个）。"""
    return site_name.split("--")[0]


def build_sites(level: str = "local_mirror", urls: Optional[Dict[str, str]] = None) -> List[Site]:
    """构造完整站点列表；urls 是 {site_name: 真实 URL}，由镜像服务器起来后注入。"""
    urls = urls or {}
    sites: List[Site] = []

    def add(name: str, expected: str, latency: float = 0.0, fail: Optional[str] = None) -> None:
        meta = REAL_PAGES[_mirror_key(name)]
        sites.append(Site(
            name=name,
            college=str(meta["college"]),
            url=urls.get(name, str(meta["url"])),
            level=level,
            expected=expected,
            latency_seconds=latency,
            fail=fail,
        ))

    # —— 真实站点（同一批语料，镜像路径 = 站点名）——
    add("stanford-profiles-andrew-ng", "found")   # 真实同名陷阱页
    add("stanford-provost-office", "found")       # 网关 A：语义命中（只讲职责，不点名）
    add("stanford-cs-faculty", "not_found")
    add("stanford-gsb-faculty", "not_found")
    add("stanford-doeer-faculty", "not_found")
    add("stanford-law-directory", "not_found")
    add("stanford-education-faculty", "not_found")
    add("stanford-medicine-profiles", "not_found")
    add("stanford-hai-people", "not_found")       # 结构差异：JS 内容为空

    # —— 半合成：延迟注入（真实正文 + 服务端 sleep）——
    add("stanford-provost-office--slow6000ms", "found", latency=6.0)
    add("stanford-profiles-andrew-ng--slow2000ms", "found", latency=2.0)
    add("stanford-law-directory--slow1000ms", "not_found", latency=1.0)

    # —— 半合成：故障注入 ——
    add("stanford-education-faculty--fail-http500", "error", fail="http500")
    add("stanford-doeer-faculty--fail-hang", "error", fail="hang")
    add("stanford-medicine-profiles--fail-malformed", "error", fail="malformed")
    return sites


# —— L1：真实站点直连（不经过本地镜像）——
# 这些就是上面 REAL_PAGES 的原始 URL。用于回答一个 L0 回答不了的问题：
# "把传输层换回真实 HTTPS 之后，协调机制的结论还成立吗？"
LIVE_SITE_ORDER = [
    "stanford-profiles-andrew-ng",
    "stanford-provost-office",
    "stanford-cs-faculty",
    "stanford-gsb-faculty",
    "stanford-law-directory",
    "stanford-education-faculty",
    "stanford-medicine-profiles",
    "stanford-hai-people",
]

# 本机在代理后面，证书链自签名/过期；真实直连必须关校验，这点要在证据里写清楚。
LIVE_NOTES = [
    "L1 live_unverified：worker 直接访问真实 HTTPS 站点，TLS 校验被关闭（本机代理证书链无法通过校验）。",
    "因此本等级证明的是'协调机制在真实网络延迟下依然成立'，不证明'抓取能力等价于浏览器渲染'。",
    "JS 渲染页面（如 hai.stanford.edu/people）在无浏览器时只能拿到骨架文本，这是 L1 的已知局限。",
]


def build_live_sites() -> List[Site]:
    """构造真实直连站点列表（level=live_unverified，url 为真实地址，无镜像路由）。"""
    sites: List[Site] = []
    for name in LIVE_SITE_ORDER:
        meta = REAL_PAGES[name]
        role = str(meta.get("role", ""))
        if role == "real_counterexample":
            expected = "found"      # 同名不同人：确实会命中名字，由证据门禁决定要不要采纳
        elif role == "real_gateway_candidate":
            expected = "found"
        else:
            expected = "not_found"
        sites.append(Site(
            name=name,
            college=str(meta["college"]),
            url=str(meta["url"]),
            level="live_unverified",
            expected=expected,
        ))
    return sites


def corpus_file_for(site: Site, corpus_dir: Path) -> Path:
    return Path(corpus_dir) / f"{_mirror_key(site.name)}.html"
