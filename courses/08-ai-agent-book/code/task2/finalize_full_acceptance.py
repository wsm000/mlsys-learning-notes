"""Finish audit/report/packaging after the long-running official campaign exits."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

ROOT=Path('/home/simin/projects/task2-memory-rag')
OUT=ROOT/'outputs/full-60x4-amd-20261015-01'
pid=json.loads((OUT/'process.json').read_text())['pid']

def live():
 try:
  stat=Path('/proc')/str(pid)/'stat'
  cmd=Path('/proc')/str(pid)/'cmdline'
  return stat.exists() and stat.read_text().split()[2]!='Z' and str(ROOT/'amd_full_acceptance.py') in cmd.read_bytes().decode(errors='replace')
 except (FileNotFoundError,ProcessLookupError): return False

for resume_pass in range(3):
 while live(): time.sleep(15)
 subprocess.run(['python3',str(ROOT/'audit_full_acceptance.py')],check=True)
 audit=json.loads((OUT/'audit.json').read_text())
 progress=json.loads((OUT/'progress.json').read_text())
 if audit['status']=='passed' or audit['errors'] or progress.get('block_reason') or resume_pass==2: break
 # Only unfinished jobs are retried; completed quality failures are never rerun.
 with (OUT/'campaign.log').open('ab') as log:
  proc=subprocess.Popen(['python3','-u',str(ROOT/'amd_full_acceptance.py')],cwd=ROOT,
    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 pid=proc.pid
 (OUT/'process.json').write_text(json.dumps({'pid':pid,'started_epoch':time.time(),
   'resume_pass':resume_pass+1,'log':str(OUT/'campaign.log')}))
 print('RESUME_UNFINISHED_JOBS pass='+str(resume_pass+1),flush=True)
canonical_path=OUT/'canonical-latest.json'
canonical=json.loads(canonical_path.read_text()) if canonical_path.exists() else {}
canonical_errors=[]
if canonical:
 run_dir=Path(canonical['run_dir'])
 for name,digest in canonical['artifacts'].items():
  path=run_dir/name
  if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
   canonical_errors.append('Artifact hash mismatch: '+name)
 for item in canonical.get('inputs',[]):
  path=Path(item['path'])
  if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
   canonical_errors.append('Input hash mismatch: '+str(path))
  elif path.is_relative_to(ROOT):
   target=OUT/'source'/path.relative_to(ROOT)
   target.parent.mkdir(parents=True,exist_ok=True)
   shutil.copy2(path,target)
 dest=OUT/'official-canonical-evidence'
 dest.mkdir(exist_ok=True)
 for name in ('evidence.json','receipts.json','manifest.json'):
  shutil.copy2(run_dir/name,dest/name)
else:
 canonical_errors.append('Official canonical evidence missing; inspect campaign.log/fatal.json')
final_status='passed' if audit['status']=='passed' and canonical.get('status')=='passed' and not canonical_errors else 'partial'
final={'status':final_status,'coverage_completed':audit['coverage_completed'],'coverage_expected':240,
 'audit_errors':audit['errors'],'canonical_hash_errors':canonical_errors,'finished_epoch':time.time(),
 'meaning':'Experiment coverage/isolation/receipt acceptance; answer quality is a separate statistic.'}
(OUT/'final-status.json').write_text(json.dumps(final,ensure_ascii=False,indent=2))

with (OUT/'case-results.csv').open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['test_id','layer','mode','passed','reward','hallucination_veto','api_attempts','failed_attempts','total_tokens','reported_cost','checkpoint_sha256'],extrasaction='ignore')
 writer.writeheader()
 writer.writerows(audit['rows'])

labels={'notes':'Simple Notes','enhanced_notes':'Enhanced Notes','json_cards':'JSON Cards','advanced_json_cards':'Advanced JSON Cards'}
lines=['# 教材3-1／3-2：60用例四模式全量验收报告','',
 '## 从“记住一条偏好”到“跨会话保持事实”','',
 '本轮用官方三层各20例检查记忆能力：基础回忆、多会话消歧、跨会话综合与主动服务。每段历史只与上一轮记忆状态共同进入写入模型；新会话只看最终记忆。评审模型单独阅读完整权威来源。','',
 f"流程验收状态：**{final_status}**；完成并独立审计 **{audit['coverage_completed']}/240** 个用例×模式。流程通过不代表回答全对。",'',
 '## 固定条件','',
 '官方版本：dbc046eb896ac4e39aa19c7774c8bf49583b89a6。60个YAML，共140段历史会话；四种模式至少1040次调用。DeepSeek-V4-Flash写入与回答，Qwen3.8-Flash-Next独立评审，同为AMD托管。官方提示与评分逻辑保持不变。','',
 '在Python 3.10运行官方campaign，未使用仓库锁定Python 3.12环境。AMD忽略seed，适配器移除了该参数；模型服务随时间变化，不宣称完全确定性复现。','',
 '## 各模式与层次的真实评分','',
 '| 模式 | 层次 | 完成/应有 | 质量通过率 | 平均奖励 | 幻觉否决率 | 总token |','|---|---|---:|---:|---:|---:|---:|']
def percent(value): return '—' if value is None else f'{value:.1%}'
for mode,groups in audit['aggregate'].items():
 for layer,row in groups.items():
  reward='—' if row['mean_reward'] is None else f"{row['mean_reward']:.3f}"
  lines.append(f"| {labels[mode]} | {layer} | {row['completed']}/{row['expected']} | {percent(row['quality_pass_rate'])} | {reward} | {percent(row['hallucination_rate'])} | {row['tokens']:,} |")
lines.extend(['','质量通过要求precision、recall、reasoning均至少3/4且无幻觉否决。proactivity单独诊断。评审是另一模型的判断，不是客观真值；应阅读失败案例的来源、答案与评审证据。缺失用例未当成零分混入均值，应结合完成覆盖率解读。','',
 '## 审计与失败保留','',
 f"独立重建并检查了{audit['counts'].get('states_audited',0)}个会话状态。检查每条写入请求、最终回答请求、评审请求与官方函数生成的预期内容一致；从响应重建的记忆与保存状态一致；重新计算的评分与结果一致。",'',
 'audit.json列出未完成用例与审计错误；campaign.log、live-receipts.jsonl和checkpoints保留失败尝试。提供商cost字段是报告值，不是确认账单实扣。','',
 '## 未通过或需要复核的回答',''])
failures=sorted([r for r in audit['rows'] if not r['passed']],key=lambda r:(r['reward'],r['test_id'],r['mode']))
if not failures: lines.append('当前已完成回答未出现质量未通过记录；这不等于对未知数据的可靠性保证。')
else:
 for row in failures[:20]:
  lines.append(f"- {row['test_id']} / {row['mode']}：reward={row['reward']:.3f}，幻觉否决={row['hallucination_veto']}。详见对应检查点的answer、judge_raw与原始来源。")
lines.extend(['','## 证据','',
 'official-canonical-evidence保留官方evidence、receipts、manifest；canonical-latest.json为官方指针副本；artifact-manifest.json记录本次交付文件哈希。原始官方作者检查点未作为本次结果使用。','',
 '本报告不预设Advanced JSON Cards一定胜出。结论应从本表和具体错误推导，不能把教材的预期趋势写成未观测的结果。未向任何打卡平台提交。',''])
(OUT/'full-acceptance-report.md').write_text('\n'.join(lines),encoding='utf-8')

package_files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ('.zip','.tmp') and p.name not in ('artifact-manifest.json','finalizer.log')]
hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in package_files}
(OUT/'artifact-manifest.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2))
with zipfile.ZipFile(OUT/'full-acceptance-evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
 for path in package_files+[OUT/'artifact-manifest.json']:
  z.write(path,path.relative_to(OUT))
with zipfile.ZipFile(OUT/'full-acceptance-evidence.zip') as z: assert z.testzip() is None
print(json.dumps(final,ensure_ascii=False),flush=True)
