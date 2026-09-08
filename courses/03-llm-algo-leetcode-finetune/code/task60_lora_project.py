#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task6-60 LoRA Fine-Tuning Project | CPU-first template run (stdlib only).
对应: 60_LoRA_Fine_Tuning_Project.ipynb (5 TODOs) + fine-tuning-project/v1 报告.
本机: Windows CPU, 无 torch/numpy 依赖, 所有资源数为模板演示口径 + 真实函数耗时测量.
RUN_REAL_TRAINING 保持 False (与 Notebook Step6 默认一致).
"""
import json, platform, sys, time, hashlib
from pathlib import Path

def audit_sft_examples(examples, max_total_chars):
    seen = set()
    total_chars = 0
    empty_response_count = 0
    duplicate_count = 0
    over_length_count = 0
    for example in examples:
        prompt = example.get('prompt', '')
        response = example.get('response', '')
        pair = (prompt, response)
        total = len(prompt) + len(response)
        total_chars += total
        if not response.strip():
            empty_response_count += 1
        if pair in seen:
            duplicate_count += 1
        else:
            seen.add(pair)
        if total > max_total_chars:
            over_length_count += 1
    total_samples = len(examples)
    avg_total_chars = total_chars / total_samples if total_samples else 0.0
    return {'total_samples': total_samples, 'empty_response_count': empty_response_count,
            'duplicate_count': duplicate_count, 'over_length_count': over_length_count,
            'avg_total_chars': round(avg_total_chars, 2)}

def loss_mask_report(attention_mask, labels, ignore_index=-100):
    mask_flat = [v for row in attention_mask for v in row]
    labels_flat = [v for row in labels for v in row]
    if len(mask_flat) != len(labels_flat):
        raise ValueError('attention_mask and labels must have the same number of tokens')
    total_tokens = len(labels_flat)
    non_padding_tokens = sum(1 for m in mask_flat if m == 1)
    supervised_tokens = sum(1 for lb in labels_flat if lb != ignore_index)
    padding_supervised_tokens = sum(1 for m, lb in zip(mask_flat, labels_flat) if m == 0 and lb != ignore_index)
    supervised_ratio = supervised_tokens / non_padding_tokens if non_padding_tokens else 0.0
    return {'total_tokens': total_tokens, 'non_padding_tokens': non_padding_tokens,
            'supervised_tokens': supervised_tokens, 'padding_supervised_tokens': padding_supervised_tokens,
            'supervised_ratio': round(supervised_ratio, 4)}

def build_lora_project_config(base_model, target_modules, rank, alpha, dropout, learning_rate, micro_batch_size, accum_steps, scheduler):
    return {'base_model': base_model, 'target_modules': target_modules, 'rank': rank, 'alpha': alpha,
            'dropout': dropout, 'learning_rate': learning_rate, 'micro_batch_size': micro_batch_size,
            'accum_steps': accum_steps, 'effective_batch_size': micro_batch_size * accum_steps, 'scheduler': scheduler}

def lora_trainable_params(in_dim, out_dim, rank):
    return rank * (in_dim + out_dim)

def full_linear_params(in_dim, out_dim):
    return in_dim * out_dim

def lora_param_ratio(in_dim, out_dim, rank):
    return lora_trainable_params(in_dim, out_dim, rank) / full_linear_params(in_dim, out_dim)

def summarize_lora_project(baseline_metrics, lora_metrics):
    param_reduction = 1.0 - lora_metrics['trainable_params'] / baseline_metrics['trainable_params']
    memory_delta = baseline_metrics['peak_mem_mb'] - lora_metrics['peak_mem_mb']
    time_delta = baseline_metrics['step_time_ms'] - lora_metrics['step_time_ms']
    train_loss_delta = lora_metrics['final_train_loss'] - baseline_metrics['final_train_loss']
    val_loss_delta = lora_metrics['final_val_loss'] - baseline_metrics['final_val_loss']
    return {'param_reduction': round(param_reduction, 4), 'peak_mem_delta_mb': round(memory_delta, 2),
            'step_time_delta_ms': round(time_delta, 2), 'final_train_loss_delta': round(train_loss_delta, 4),
            'final_val_loss_delta': round(val_loss_delta, 4)}

def build_adapter_artifact_record(adapter_path, tokenizer_path, merge_checked, sanity_generation_checked):
    return {'adapter_path': adapter_path, 'tokenizer_path': tokenizer_path, 'merge_checked': merge_checked,
            'sanity_generation_checked': sanity_generation_checked}

def build_lora_project_report(config, baseline, candidates, quality, resources, artifacts, decision, environment=None):
    return {'schema_version': 'fine-tuning-project/v1', 'project': '60_lora_fine_tuning', 'stage': 'project_decision',
            'config': config, 'baseline': baseline, 'candidates': candidates, 'quality': quality,
            'resources': resources, 'artifacts': artifacts, 'decision': decision, 'environment': environment or {}}

def check_lora_project_readiness(data_audit, mask_report, artifact_record):
    issues = []
    if data_audit['empty_response_count'] > 0:
        issues.append('empty_response')
    if data_audit['duplicate_count'] > 0:
        issues.append('duplicate_examples')
    if mask_report['padding_supervised_tokens'] > 0:
        issues.append('padding_supervised')
    if mask_report['supervised_tokens'] == 0:
        issues.append('no_supervised_tokens')
    if not artifact_record['merge_checked']:
        issues.append('merge_not_checked')
    if not artifact_record['sanity_generation_checked']:
        issues.append('sanity_generation_not_checked')
    return {'ready': len(issues) == 0, 'issues': issues}

def recommend_lora_decision(summary, readiness, min_param_reduction=0.5, max_val_loss_delta=0.03, min_peak_mem_delta_mb=128.0, min_step_time_delta_ms=-3.0):
    memory_gain_ok = summary['peak_mem_delta_mb'] >= min_peak_mem_delta_mb
    speed_not_too_bad = summary['step_time_delta_ms'] >= min_step_time_delta_ms
    if not readiness['ready']:
        decision, reason = 'tune', '数据、loss mask 或 adapter 交付检查未通过，先修复项目可信度问题。'
    elif summary['param_reduction'] < min_param_reduction:
        decision, reason = 'reject', '参数节省不足，LoRA 没有带来足够训练成本收益。'
    elif summary['final_val_loss_delta'] > max_val_loss_delta:
        decision, reason = 'tune', '参数节省达标，但验证集 loss 损失偏大，优先调 rank、target modules 或学习率。'
    elif not (memory_gain_ok or speed_not_too_bad):
        decision, reason = 'tune', '参数节省和验证损失可接受，但显存收益偏弱且速度恶化，优先继续调 rank、插层范围或 batch 配置。'
    else:
        decision, reason = 'accept', '参数节省达标，验证集损失可接受，交付检查通过，可以保留当前 LoRA 配置。'
    return {'decision': decision, 'reason': reason}

def test_lora_project_template():
    examples = [
        {'prompt': '问：什么是 LoRA？', 'response': '答：LoRA 是低秩适配方法。'},
        {'prompt': '问：如何检查 loss？', 'response': '答：检查 labels 中参与监督的 token。'},
        {'prompt': '问：什么是 LoRA？', 'response': '答：LoRA 是低秩适配方法。'},
        {'prompt': '问：空回答？', 'response': ''},
    ]
    audit = audit_sft_examples(examples, max_total_chars=30)
    assert audit['total_samples'] == 4
    assert audit['empty_response_count'] == 1
    assert audit['duplicate_count'] == 1
    assert audit['over_length_count'] == 1
    mask = [[1, 1, 1, 0], [1, 1, 0, 0]]
    labels = [[-100, 7, 8, -100], [-100, 9, -100, 3]]
    report = loss_mask_report(mask, labels)
    assert report['total_tokens'] == 8
    assert report['non_padding_tokens'] == 5
    assert report['supervised_tokens'] == 4
    assert report['padding_supervised_tokens'] == 1
    assert report['supervised_ratio'] == 0.8
    config = build_lora_project_config('tiny-llama', ['q_proj', 'v_proj'], 8, 16, 0.05, 2e-4, 2, 4, 'wsd-cosine')
    assert config['effective_batch_size'] == 8
    assert lora_trainable_params(8, 8, 2) == 32
    assert full_linear_params(8, 8) == 64
    assert abs(lora_param_ratio(8, 8, 2) - 0.5) < 1e-12
    baseline = {'trainable_params': 1000, 'step_time_ms': 20.0, 'peak_mem_mb': 1024.0, 'final_train_loss': 0.40, 'final_val_loss': 0.50}
    lora = {'trainable_params': 100, 'step_time_ms': 22.0, 'peak_mem_mb': 768.0, 'final_train_loss': 0.42, 'final_val_loss': 0.52}
    summary = summarize_lora_project(baseline, lora)
    assert summary['param_reduction'] == 0.9
    assert summary['peak_mem_delta_mb'] == 256.0
    assert summary['step_time_delta_ms'] == -2.0
    assert summary['final_train_loss_delta'] == 0.02
    assert summary['final_val_loss_delta'] == 0.02
    artifact = build_adapter_artifact_record('outputs/lora-adapter', 'outputs/tokenizer', True, True)
    pr = build_lora_project_report({'model': 'tiny-llama', 'dtype': 'bf16', 'seed': 42}, baseline, [{'name': 'lora', **lora}],
        {'train_loss': 0.42, 'val_loss': 0.52, 'task_metrics': {}}, {'trainable_params': 100, 'peak_memory_mb': 768.0, 'step_time_ms': 22.0},
        {'adapter': artifact}, {'decision': 'accept', 'reason': 'test'})
    for s in ('config', 'baseline', 'candidates', 'quality', 'resources', 'artifacts', 'decision', 'environment'):
        assert s in pr
    clean_audit = {'total_samples': 2, 'empty_response_count': 0, 'duplicate_count': 0, 'over_length_count': 0, 'avg_total_chars': 12.0}
    clean_report = {'total_tokens': 8, 'non_padding_tokens': 5, 'supervised_tokens': 3, 'padding_supervised_tokens': 0, 'supervised_ratio': 0.6}
    readiness = check_lora_project_readiness(clean_audit, clean_report, artifact)
    assert readiness['ready'] is True
    dirty = check_lora_project_readiness(audit, report, artifact)
    assert dirty['ready'] is False and 'empty_response' in dirty['issues'] and 'padding_supervised' in dirty['issues']
    assert recommend_lora_decision(summary, readiness)['decision'] == 'accept'
    assert recommend_lora_decision(summary, dirty)['decision'] == 'tune'
    ws = dict(summary); ws['final_val_loss_delta'] = 0.08
    assert recommend_lora_decision(ws, readiness)['decision'] == 'tune'
    ws2 = dict(summary); ws2['param_reduction'] = 0.2
    assert recommend_lora_decision(ws2, readiness)['decision'] == 'reject'
    ws3 = dict(summary); ws3['peak_mem_delta_mb'] = 32.0; ws3['step_time_delta_ms'] = -6.0
    assert recommend_lora_decision(ws3, readiness)['decision'] == 'tune'
    print("PASS test_lora_project_template")

if __name__ == '__main__':
    t0 = time.perf_counter()
    print("=== Task60 LoRA Project | CPU-first ===")
    print(f"host={platform.node()} {platform.platform()} python={sys.version.split()[0]}")
    test_lora_project_template()
    # 演示用干净数据走 accept 分支 + 真实耗时
    examples = [
        {'prompt': '问：什么是 LoRA？', 'response': '答：低秩旁路，只训 A/B。'},
        {'prompt': '问：如何检查 loss？', 'response': '答：看 labels!=-100 且 mask==1。'},
    ]
    t1 = time.perf_counter(); audit = audit_sft_examples(examples, max_total_chars=64); t_audit = (time.perf_counter()-t1)*1000
    t1 = time.perf_counter(); mrep = loss_mask_report([[1,1,1,0]],[[-100,7,8,-100]]); t_mask = (time.perf_counter()-t1)*1000
    config = build_lora_project_config('tiny-llama', ['q_proj','v_proj'], 8, 16, 0.05, 2e-4, 2, 4, 'wsd-cosine')
    print("audit:", audit, f"({t_audit:.3f}ms)")
    print("mask:", mrep, f"({t_mask:.3f}ms)")
    print("config:", config)
    for hs, rk in [(4096,8),(4096,16),(8192,16)]:
        tr = lora_trainable_params(hs,hs,rk); tot = full_linear_params(hs,hs); ra = lora_param_ratio(hs,hs,rk)
        print(f"hidden={hs}, rank={rk} -> trainable={tr:,}, full={tot:,}, ratio={ra:.4%}")
    baseline = {'trainable_params': 1000, 'step_time_ms': 20.0, 'peak_mem_mb': 1024.0, 'final_train_loss': 0.40, 'final_val_loss': 0.50}
    lora = {'trainable_params': 100, 'step_time_ms': 22.0, 'peak_mem_mb': 768.0, 'final_train_loss': 0.42, 'final_val_loss': 0.52}
    summary = summarize_lora_project(baseline, lora)
    artifact = build_adapter_artifact_record('outputs/lora-adapter', 'outputs/tokenizer', True, True)
    readiness = check_lora_project_readiness(audit, mrep, artifact)
    decision = recommend_lora_decision(summary, readiness)
    print("summary:", summary); print("readiness:", readiness); print("decision:", decision)
    env = {'device': 'cpu', 'platform': platform.platform(), 'python': sys.version.split()[0],
           'run_real_training': False, 'note': 'CPU-first template; resource numbers are template demo, timing is real measured'}
    report = build_lora_project_report({'model':'tiny-llama','dtype':'bf16','seed':42}, baseline, [{'name':'lora', **lora}],
        {'train_loss': 0.42, 'val_loss': 0.52, 'task_metrics': {}}, {'trainable_params': 100, 'peak_memory_mb': 768.0, 'step_time_ms': 22.0},
        {'adapter': artifact}, decision, env)
    outdir = Path("打卡材料/task6_60_61"); outdir.mkdir(parents=True, exist_ok=True)
    (outdir/"60_lora_project_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (outdir/"60_audit_mask_summary.json").write_text(json.dumps({'audit': audit, 'mask': mrep, 'summary': summary, 'readiness': readiness}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"total_wall={(time.perf_counter()-t0)*1000:.1f}ms; saved to {outdir}")
    print("DONE 60")
