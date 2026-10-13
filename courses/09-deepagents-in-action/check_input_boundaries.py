"""Test CLI validation without any model requests or credentials."""
import json
from pathlib import Path
import subprocess
import sys

records = []
for name, question in [('empty', ''), ('too_long', 'x' * 4001)]:
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('agent.py')), question],
                            capture_output=True, text=True, timeout=15)
    passed = result.returncode == 2 and 'question must contain 1' in result.stderr
    records.append({'case': name, 'characters': len(question), 'exit_code': result.returncode,
                    'passed': passed, 'stderr': result.stderr,
                    'layer': 'CLI rejects before model/client initialization'})
Path('outputs').mkdir(exist_ok=True)
Path('outputs/input-boundaries.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
print(json.dumps(records, indent=2))
assert all(r['passed'] for r in records)
