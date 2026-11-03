"""Refresh inspection and produce a reconciled draft, not a final report."""
import hashlib
import json
import runpy
from contextlib import redirect_stdout
from decimal import Decimal
from io import StringIO
from pathlib import Path

root = Path(__file__).resolve().parent
sources = sorted((root / 'data').glob('*.csv'))
before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
with redirect_stdout(StringIO()):
    runpy.run_path(str(root / 'inspect_data.py'))
inspection = json.loads((root / 'inspection.json').read_text(encoding='utf-8'))
totals = {region: Decimal('0') for region in ('华东', '华南', '华北')}
pending = Decimal('0')
pending_rows = []
for row in inspection['rows_detail']:
    if any(issue != 'missing or unknown region' for issue in row['issues']):
        raise ValueError('Non-region validation failure requires review before aggregation')
    amount = Decimal(row['amount'])
    if row['issues']:
        pending += amount
        pending_rows.append(row)
    else:
        totals[row['region']] += amount
known = sum(totals.values(), Decimal('0'))
raw = Decimal(inspection['raw_total_cny'])
if known + pending != raw:
    raise ValueError('Reconciliation failed')
after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
if before != after:
    raise ValueError('Source files changed during generation')
lines = [
    '# 地区净收入报告（草稿，未完成）', '',
    '状态：地区信息待确认，不能作为最终地区汇总交付。', '',
    '范围：本地 01.csv、02.csv、03.csv，共 7 条模拟记录；仅说明这些输入文件的情况。', '',
    '单位：人民币元。净收入包含本期退款；A003 的 -30 元按口径抵扣华东收入。', '',
    '| 地区或状态 | 净收入（元） |', '| --- | ---: |',
]
lines += [f'| {region} | {amount:.2f} |' for region, amount in totals.items()]
lines += [f'| 已知地区小计 | {known:.2f} |',
          f'| 地区待确认（非业务地区） | {pending:.2f} |',
          f'| 原始净收入总额 | {raw:.2f} |', '',
          f'对账通过：{known:.2f} + {pending:.2f} = {raw:.2f}。已知地区小计与待确认金额不重叠。', '',
          '## 待确认明细', '',
          '| 源文件 | CSV 行号 | 事件标识 | 金额（元） | 问题 |',
          '| --- | ---: | --- | ---: | --- |']
lines += [f"| {r['file']} | {r['line']} | {r['order_id']} | {Decimal(r['amount']):.2f} | 地区缺失或未知 |"
          for r in pending_rows]
lines += ['', '最终完成条件：取得 C002 的地区确认或明确处理规则，重跑校验与汇总。不得猜测地区或静默丢弃金额。',
          '', '原始 CSV 未修改。源内金额已对账；业务全集完整性未验证。', '']
(root / 'report-draft.md').write_text('\n'.join(lines), encoding='utf-8')
checks = dict(status='draft_incomplete', final_ready=False,
              source_hashes=before, sources_unchanged=before == after,
              files=inspection['files'], rows=inspection['rows'],
              known_regions={k: str(v) for k, v in totals.items()},
              known_subtotal=str(known), pending_total=str(pending), raw_total=str(raw),
              reconciliation_passed=known + pending == raw, pending_rows=pending_rows)
(root / 'draft-checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(checks, ensure_ascii=False, indent=2))
