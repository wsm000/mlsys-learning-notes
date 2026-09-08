#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task6-61 Model Architecture Exploration | CPU-first (stdlib only).
对应: 61_Model_Architecture_Exploration.ipynb (4 TODOs).
"""
import json, platform, sys, time
from pathlib import Path

def validate_architecture_baseline(baseline):
    issues = []
    for f in ['params', 'memory_mb', 'step_time_ms', 'score', 'deploy_cost']:
        if f not in baseline:
            issues.append(f'missing: {f}')
    if 'params' in baseline and int(baseline.get('params', 0)) <= 0:
        issues.append('invalid params')
    if 'memory_mb' in baseline and int(baseline.get('memory_mb', 0)) <= 0:
        issues.append('invalid memory_mb')
    if 'step_time_ms' in baseline and float(baseline.get('step_time_ms', 0.0)) <= 0.0:
        issues.append('invalid step_time_ms')
    if 'deploy_cost' in baseline and float(baseline.get('deploy_cost', 0.0)) <= 0.0:
        issues.append('invalid deploy_cost')
    return {'ready': len(issues) == 0, 'issues': issues}

def summarize_architecture_candidates(candidates, baseline_params):
    names, deltas, mods = [], {}, set()
    best, best_score = None, None
    for c in candidates:
        name = str(c.get('name', 'candidate')); score = float(c.get('score', 0.0)); params = int(c.get('params', baseline_params))
        names.append(name); deltas[name] = params - baseline_params
        mods.update(c.get('changed_modules', []))
        if best_score is None or score > best_score:
            best_score, best = score, name
    return {'candidate_count': len(candidates), 'candidate_names': names, 'baseline_params': baseline_params,
            'best_candidate': best, 'param_deltas': deltas, 'changed_module_union': sorted(mods)}

def compare_architecture_pair(baseline, candidate):
    return {'baseline_name': baseline.get('name','baseline'), 'candidate_name': candidate.get('name','candidate'),
            'changed_modules': list(candidate.get('changed_modules', [])),
            'param_delta': int(candidate.get('params',0)) - int(baseline.get('params',0)),
            'memory_delta_mb': int(candidate.get('memory_mb',0)) - int(baseline.get('memory_mb',0)),
            'step_time_delta_ms': round(float(candidate.get('step_time_ms',0.0)) - float(baseline.get('step_time_ms',0.0)), 4),
            'score_delta': float(candidate.get('score',0.0)) - float(baseline.get('score',0.0)),
            'deploy_delta': round(float(candidate.get('deploy_cost',0.0)) - float(baseline.get('deploy_cost',0.0)), 4)}

def recommend_candidate(baseline, candidates, param_budget, max_deploy_delta, max_memory_delta_mb=0, max_step_time_delta_ms=0.0):
    chk = validate_architecture_baseline(baseline)
    if not chk['ready']:
        return {'decision':'reject','recommended_name':None,'reason':'baseline 口径不完整，不能进入候选比较','next_action':'repair_baseline_measurement'}
    feasible = [c for c in candidates if int(c.get('params', 10**9)) <= param_budget]
    if not feasible:
        return {'decision':'reject','recommended_name':None,'reason':'没有候选满足参数预算','next_action':'reduce_candidate_scope'}
    deploy_ok = [c for c in feasible if round(float(c.get('deploy_cost',0.0))-float(baseline.get('deploy_cost',0.0)),4) <= max_deploy_delta]
    if not deploy_ok:
        best = min(feasible, key=lambda x: (round(float(x.get('deploy_cost',0.0))-float(baseline.get('deploy_cost',0.0)),4), int(x.get('memory_mb',10**9)), round(float(x.get('step_time_ms',10**9))-float(baseline.get('step_time_ms',0.0)),4), -float(x.get('score',0.0))))
        return {'decision':'tune','recommended_name':best.get('name','candidate'),'reason':'候选有收益，但部署代价整体超出边界','next_action':'refine_modules_or_capacity'}
    mem_ok = [c for c in deploy_ok if int(c.get('memory_mb',10**9)) - int(baseline.get('memory_mb',0)) <= max_memory_delta_mb]
    step_ok = [c for c in deploy_ok if float(c.get('step_time_ms',10**9)) - float(baseline.get('step_time_ms',0.0)) <= max_step_time_delta_ms]
    pool = [c for c in step_ok if c in mem_ok] or step_ok or mem_ok or deploy_ok
    best = max(pool, key=lambda x: (float(x.get('score',0.0)), -int(x.get('memory_mb',10**9))))
    comp = compare_architecture_pair(baseline, best)
    if comp['score_delta'] > 0 and comp['deploy_delta'] <= max_deploy_delta and comp['memory_delta_mb'] <= max_memory_delta_mb and comp['step_time_delta_ms'] <= max_step_time_delta_ms:
        return {'decision':'accept','recommended_name':best.get('name','candidate'),'reason':'收益、预算和部署代价都达标','next_action':'promote_to_extended_eval'}
    if comp['score_delta'] > 0:
        return {'decision':'tune','recommended_name':best.get('name','candidate'),'reason':'分数提升可用，但显存、step time 或部署代价仍偏高','next_action':'refine_modules_or_capacity'}
    return {'decision':'reject','recommended_name':best.get('name','candidate'),'reason':'候选未带来稳定收益','next_action':'fallback_to_baseline'}

def test_architecture_project_template():
    baseline = {'name':'baseline','params':100,'memory_mb':1200,'step_time_ms':92.0,'score':0.66,'deploy_cost':1.0}
    candidates = [
        {'name':'small_norm','params':96,'memory_mb':1100,'step_time_ms':90.0,'changed_modules':['norm'],'score':0.72,'deploy_cost':1.05},
        {'name':'wide_ffn','params':108,'memory_mb':1320,'step_time_ms':105.0,'changed_modules':['ffn'],'score':0.68,'deploy_cost':1.25},
    ]
    assert validate_architecture_baseline(baseline) == {'ready': True, 'issues': []}
    s = summarize_architecture_candidates(candidates, baseline_params=baseline['params'])
    assert s['candidate_count']==2 and s['best_candidate']=='small_norm' and s['param_deltas']['wide_ffn']==8
    p = compare_architecture_pair(baseline, candidates[0])
    assert p['param_delta']==-4 and p['memory_delta_mb']==-100 and abs(p['score_delta']-0.06)<1e-8 and abs(p['deploy_delta']-0.05)<1e-8
    d = recommend_candidate(baseline, candidates, param_budget=102, max_deploy_delta=0.1, max_memory_delta_mb=0, max_step_time_delta_ms=4.0)
    assert d['decision']=='accept' and d['recommended_name']=='small_norm' and d['next_action']=='promote_to_extended_eval'
    tc = [
        {'name':'memory_heavy','params':101,'memory_mb':1450,'step_time_ms':97.0,'changed_modules':['ffn'],'score':0.78,'deploy_cost':1.08},
        {'name':'balanced','params':100,'memory_mb':1180,'step_time_ms':94.5,'changed_modules':['attn'],'score':0.71,'deploy_cost':1.06},
    ]
    assert recommend_candidate(baseline, tc, param_budget=102, max_deploy_delta=0.1, max_memory_delta_mb=80, max_step_time_delta_ms=4.0)['decision']=='accept'
    assert recommend_candidate(baseline, tc, param_budget=102, max_deploy_delta=0.03, max_memory_delta_mb=80, max_step_time_delta_ms=4.0)['decision']=='tune'
    slow = [{'name':'slow_gain','params':100,'memory_mb':1170,'step_time_ms':99.0,'changed_modules':['attn'],'score':0.73,'deploy_cost':1.05}]
    assert recommend_candidate(baseline, slow, param_budget=102, max_deploy_delta=0.1, max_memory_delta_mb=0, max_step_time_delta_ms=4.0)['decision']=='tune'
    inv = validate_architecture_baseline({'name':'baseline','params':100,'memory_mb':1200,'score':0.66})
    assert inv['ready'] is False and 'missing: step_time_ms' in inv['issues']
    print("PASS test_architecture_project_template")

if __name__ == '__main__':
    t0 = time.perf_counter()
    print("=== Task61 Architecture Exploration | CPU-first ===")
    print(f"host={platform.node()} {platform.platform()} python={sys.version.split()[0]}")
    test_architecture_project_template()
    baseline = {'name':'baseline','params':100,'memory_mb':1200,'step_time_ms':92.0,'score':0.66,'deploy_cost':1.0}
    candidates = [
        {'name':'small_norm','params':96,'memory_mb':1100,'step_time_ms':90.0,'changed_modules':['norm'],'score':0.72,'deploy_cost':1.05},
        {'name':'wide_ffn','params':108,'memory_mb':1320,'step_time_ms':105.0,'changed_modules':['ffn'],'score':0.68,'deploy_cost':1.25},
    ]
    t1=time.perf_counter(); chk=validate_architecture_baseline(baseline); t_chk=(time.perf_counter()-t1)*1000
    t1=time.perf_counter(); summ=summarize_architecture_candidates(candidates, baseline['params']); t_sum=(time.perf_counter()-t1)*1000
    pair=compare_architecture_pair(baseline, candidates[0])
    decision=recommend_candidate(baseline, candidates, param_budget=102, max_deploy_delta=0.1, max_memory_delta_mb=0, max_step_time_delta_ms=4.0)
    print(f"baseline_check={chk} ({t_chk:.3f}ms)")
    print(f"summary={summ} ({t_sum:.3f}ms)")
    print(f"pair_small_norm={pair}")
    print(f"decision={decision}")
    report={'project':'61_model_architecture_exploration','baseline':baseline,'candidates':candidates,
            'summary':summ,'pair_example':pair,'decision':decision,
            'environment':{'device':'cpu','platform':platform.platform(),'python':sys.version.split()[0]}}
    outdir=Path("打卡材料/task6_60_61"); outdir.mkdir(parents=True, exist_ok=True)
    (outdir/"61_architecture_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"total_wall={(time.perf_counter()-t0)*1000:.1f}ms; saved to {outdir}")
    print("DONE 61")
