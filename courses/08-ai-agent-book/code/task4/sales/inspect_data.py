"""Read-only source inspection for the teaching fixture; no final report."""
import csv
import json
from decimal import Decimal
from pathlib import Path

root = Path(__file__).resolve().parent
files = sorted((root / 'data').glob('*.csv'))
if len(files) != 3:
    raise ValueError('Expected exactly three source files')
rows = []
seen = set()
total = Decimal('0')
for path in files:
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['order_id', 'region', 'event', 'amount', 'currency']:
            raise ValueError(f'Unexpected columns: {path.name}')
        for line, row in enumerate(reader, 2):
            amount = Decimal(row['amount'])
            if not amount.is_finite():
                raise ValueError(f'Non-finite amount: {path.name}:{line}')
            total += amount
            issues = []
            if row['region'] not in {'华东', '华南', '华北'}:
                issues.append('missing or unknown region')
            if not row['order_id'] or row['order_id'] in seen:
                issues.append('missing or duplicate event id')
            seen.add(row['order_id'])
            if row['currency'] != 'CNY':
                issues.append('unexpected currency')
            if row['event'] not in {'sale', 'refund'}:
                issues.append('unknown event')
            if ((row['event'] == 'sale' and amount < 0)
                    or (row['event'] == 'refund' and amount > 0)):
                issues.append('event/amount sign mismatch')
            rows.append(dict(file=path.name, line=line, **row, issues=issues))

errors = [row for row in rows if row['issues']]
result = dict(files=len(files), rows=len(rows), raw_total_cny=str(total),
              errors=errors, rows_detail=rows, final_ready=not errors)
output = json.dumps(result, ensure_ascii=False, indent=2)
(root / 'inspection.json').write_text(output, encoding='utf-8')
print(output)
