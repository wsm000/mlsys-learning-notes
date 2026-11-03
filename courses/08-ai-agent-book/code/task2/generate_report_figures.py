"""Create editable SVG explanations; measured chart reads preserved API receipts."""
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET

out = Path(__file__).parent / 'outputs'
style = '<style>text{font-family:Microsoft YaHei, Noto Sans CJK SC, sans-serif;fill:#e8edf6} .muted{fill:#a7b5cb;font-size:17px}.label{font-size:21px}.small{font-size:16px}</style>'
header = '<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="460" viewBox="0 0 1050 460">' + style + '<rect width="1050" height="460" rx="18" fill="#101724"/>'
geometry = header + '''
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs>
<text x="35" y="42" font-size="26">把文字想成箭头：方向相近，不代表规则仍然有效</text>
<path d="M90 355 H665 M110 375 V68" stroke="#39475b" stroke-width="2"/>
<path d="M110 340 L610 340" stroke="#67d5f5" stroke-width="4" marker-end="url(#arrow)"/>
<text x="623" y="347" class="label">问题 q</text>
'''
for angle, color, label in [(12, '#ff8b91', '旧版资料（可能更相近）'), (30, '#8ee6b3', '当前资料')]:
    x = 110 + 500 * math.cos(math.radians(angle))
    y = 340 - 500 * math.sin(math.radians(angle))
    geometry += f'<path d="M110 340 L{x:.1f} {y:.1f}" stroke="{color}" stroke-width="4" marker-end="url(#arrow)"/>'
    label_x, label_y = (420, 220) if angle == 12 else (x + 12, y - 12)
    geometry += f'<text x="{label_x:.1f}" y="{label_y:.1f}" class="label" style="fill:{color}">{label}</text>'
geometry += '''
<path d="M180 340 A70 70 0 0 0 177.6 321.9" fill="none" stroke="#f6d77b" stroke-width="2"/>
<text x="193" y="327" class="small" style="fill:#f6d77b">夹角</text>
<rect x="715" y="145" width="300" height="170" rx="12" fill="#1c283a"/>
<text x="738" y="181" class="label">相似度决定候选排序</text>
<text x="738" y="223" class="label">版本与状态决定有效性</text>
<text x="738" y="269" class="muted">cos 值不是正确概率</text>
<text x="35" y="415" class="muted">二维人为示意，不是真实512维向量的投影，也不是本次实测排名。</text>
<text x="35" y="441" class="muted">检索可认为旧文更相关；应用仍须过滤失效版本。</text></svg>'''
(out / 'embedding_geometry.svg').write_text(geometry, encoding='utf-8')

run = out / 'conditional-rag-20261015-02'
values = []
for mode, label, color in [('neighbors','Top-3 + 相邻块','#67d5f5'), ('condition-search','追加例外检索','#f6d77b'), ('linked-rules','人工规则关联','#8ee6b3')]:
    receipt = json.loads((run / ('generation-' + mode + '.json')).read_text(encoding='utf-8'))
    values.append((mode, label, color, receipt['response']['usage']['prompt_tokens']))
chart = header + '<text x="35" y="45" font-size="27">补齐例外需要更多上下文：实际输入 token</text>'
chart += '<text x="35" y="79" class="muted">同一模型、同一组四题；包含消息格式、元数据与重复提供的证据</text>'
for i, (mode, label, color, value) in enumerate(values):
    y = 120 + i * 85
    w = value / 3200 * 620
    change = '' if i == 0 else f'  (+{(value/values[0][3]-1)*100:.0f}%)'
    chart += f'<text x="35" y="{y+28}" class="label">{label}</text><rect x="240" y="{y}" width="{w:.2f}" height="42" rx="6" fill="{color}"/><text x="{250+w:.2f}" y="{y+28}" class="label">{value:,}{change}</text>'
chart += '<text x="35" y="406" class="muted">来源：修正后的 conditional-rag-20261015-02 / generation-*.json</text>'
chart += '<text x="35" y="436" class="muted">这是单次输入量对照，不是费用账单或模型纯推理延迟。</text></svg>'
(out / 'conditional_context_cost.svg').write_text(chart, encoding='utf-8')
for name in ('embedding_geometry.svg', 'conditional_context_cost.svg'):
    ET.parse(out / name)
print('SVGs created; measured input tokens:', [v[3] for v in values])
