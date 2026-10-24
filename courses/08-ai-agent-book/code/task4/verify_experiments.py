"""Reproduce both experiments in temporary copies using only the standard library."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent
records = {}


def run(args, cwd):
    result = subprocess.run([sys.executable, *args], cwd=cwd,
                            capture_output=True, text=True, encoding='utf-8',
                            errors='replace', timeout=30)
    return result.returncode, result.stdout + result.stderr


def check(condition, message):
    if not condition:
        raise AssertionError(message)


with tempfile.TemporaryDirectory(prefix='chapter5-') as directory:
    work = Path(directory)
    coding = work / 'coding'
    sales = work / 'sales'
    shutil.copytree(root / 'coding', coding, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(root / 'sales', sales, ignore=shutil.ignore_patterns('__pycache__'))
    cart = coding / 'cart.py'
    correct = cart.read_text(encoding='utf-8')
    guard = "    if not (0 <= discount <= 1):\n        raise ValueError('discount must be between 0 and 1')\n"
    check(guard in correct, 'Expected final guard missing')
    variants = {
        'no_guard': (correct.replace(guard, ''), 3),
        'or_guard': (correct.replace('not (0 <= discount <= 1)', 'discount < 0 or discount > 1'), 1),
        'closed_interval': (correct, 0),
    }
    for name, (source, failures) in variants.items():
        cart.write_text(source, encoding='utf-8')
        shutil.rmtree(coding / '__pycache__', ignore_errors=True)
        code, output = run(['-B', '-m', 'unittest', '-v'], coding)
        check('Ran 7 tests' in output, f'{name}: tests did not run')
        check(code == (1 if failures else 0), f'{name}: unexpected exit code')
        check((f'FAILED (failures={failures})' in output) if failures else ('\nOK' in output),
              f'{name}: unexpected test outcome')
        if failures:
            check(output.count('ValueError not raised') == failures, 'Unexpected failure reason')
        if name == 'or_guard':
            check('FAIL: test_nan_discount' in output, 'Expected NaN failure')
        records[name] = dict(tests=7, failures=failures, exit_code=code)
        print(f'{name}: 7 tests, {failures} failures (expected)')

    sources = sorted((sales / 'data').glob('*.csv'))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    code, output = run(['build_draft.py'], sales)
    check(code == 0, output)
    draft = json.loads((sales / 'draft-checks.json').read_text(encoding='utf-8'))
    check(draft['known_subtotal'] == '440' and draft['pending_total'] == '60'
          and draft['raw_total'] == '500' and draft['final_ready'] is False
          and draft['reconciliation_passed'], 'Unexpected draft')
    records['draft'] = {k: draft[k] for k in ('known_subtotal', 'pending_total', 'raw_total', 'final_ready')}
    print('draft: 440 + 60 = 500; final_ready=False')

    code, output = run(['build_confirmed.py'], sales)
    check(code == 0, output)
    final = json.loads((sales / 'v2-checks.json').read_text(encoding='utf-8'))
    check(final['regions'] == {'华东': '120', '华南': '300', '华北': '80'}
          and final['total'] == '500' and final['pending_total'] == '0'
          and final['final_ready_in_simulation'] and final['reconciliation_passed'], 'Unexpected v2')
    check(hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
          'Original CSV content changed')
    records['confirmed'] = dict(regions=final['regions'], total=final['total'],
                               pending_total=final['pending_total'], sources_unchanged=True,
                               simulated_confirmation=True)
    print('v2: 120 + 300 + 80 = 500; source CSVs unchanged')

(root / 'verification.json').write_text(json.dumps(dict(passed=True, checks=records),
                                                   ensure_ascii=False, indent=2), encoding='utf-8')
print('All expected observations reproduced; verification.json saved.')
