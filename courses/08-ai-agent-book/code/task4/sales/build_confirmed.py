"""Apply a source-bound simulation overlay without changing source CSVs."""
import copy
import hashlib
import json
import runpy
from contextlib import redirect_stdout
from decimal import Decimal
from io import StringIO
from pathlib import Path

root = Path(__file__).resolve().parent
sources = sorted((root / 'data').glob('*.csv'))
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
with redirect_stdout(StringIO()):
    runpy.run_path(str(root / 'inspect_data.py'))
inspection = json.loads((root / 'inspection.json').read_text(encoding='utf-8'))
rows = copy.deepcopy(inspection['rows_detail'])
overlay_path = root / 'region-confirmations.json'
overlay = json.loads(overlay_path.read_text(encoding='utf-8'))
applied = []
seen = set()
for entry in overlay['confirmations']:
    key = (entry['source_file'], entry['order_id'], entry['field'])
    if key in seen:
        raise ValueError('Duplicate confirmation')
    seen.add(key)
    if (entry['field'] != 'region'
            or entry['status'] != 'confirmed_in_simulation'
            or not entry['evidence']
            or entry['confirmed_value'] not in {'华东', '华南', '华北'}
            or hashes.get(entry['source_file']) != entry['source_sha256']):
        raise ValueError('Invalid or stale confirmation')
    matches = [r for r in rows if r['file'] == entry['source_file']
               and r['order_id'] == entry['order_id']]
    if len(matches) != 1 or matches[0]['region'] != entry['original_value']:
        raise ValueError('Confirmation target must match exactly once')
    row = matches[0]
    row['region'] = entry['confirmed_value']
    row['issues'] = [i for i in row['issues'] if i != 'missing or unknown region']
    applied.append(entry)

if any(r['issues'] or r['region'] not in {'华东', '华南', '华北'} for r in rows):
    raise ValueError('Unresolved source validation errors')
totals = {region: Decimal('0') for region in ('华东', '华南', '华北')}
for row in rows:
    totals[row['region']] += Decimal(row['amount'])
total = sum(totals.values(), Decimal('0'))
if total != Decimal(inspection['raw_total_cny']):
    raise ValueError('Reconciliation failed')
if hashes != {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}:
    raise ValueError('Source files changed')
lines = ['# 地区净收入报告 v2（模拟确认后）', '',
         '状态：本次三文件教学任务验收通过。地区确认是实验假设，不是真实业务证明。', '',
         '范围：01.csv、02.csv、03.csv，共 7 条记录。单位为人民币元，含本期退款。', '',
         '| 地区 | 净收入（元） |', '| --- | ---: |']
lines += [f'| {region} | {amount:.2f} |' for region, amount in totals.items()]
lines += [f'| 总额 | {total:.2f} |', '',
          '对账通过；待确认金额为 0。三份原始 CSV 保持不变。', '',
          '## 与草稿的差异', '',
          'C002 的 60 元通过独立修订层从“地区待确认”归入华南。华南由 240 元变为 300 元，总额仍为 500 元。', '',
          '修订依据：region-confirmations.json 中的教学模拟确认。按源文件哈希、事件标识和原字段值匹配；源版本变化时拒绝直接套用。', '',
          '覆盖结论限于给定三份输入及本次修订，不证明整个业务的数据全集完整。', '']
report = '\n'.join(lines)
(root / 'report-v2.md').write_text(report, encoding='utf-8')
checks = dict(status='passed_with_simulated_confirmation', final_ready_in_simulation=True,
              source_hashes=hashes, sources_unchanged=True,
              overlay_sha256=hashlib.sha256(overlay_path.read_bytes()).hexdigest(),
              applied_confirmations=applied, rows=len(rows),
              regions={k: str(v) for k, v in totals.items()}, total=str(total),
              pending_total='0', reconciliation_passed=True,
              report_sha256=hashlib.sha256(report.encode('utf-8')).hexdigest())
(root / 'v2-checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(checks, ensure_ascii=False, indent=2))
