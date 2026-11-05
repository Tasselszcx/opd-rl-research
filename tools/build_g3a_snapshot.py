"""Publish aggregate clean-v4 development evidence; never reads a remote machine."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


MODES = {'plain': 'Baseline', 'answer_only': '只给答案',
         'full_reasoning': '完整解法', 'error_targeted': '针对错误提示'}
ROUTE_FIELDS = ['task', 'seed', 'student_start_step', 'guidance', 'status',
                'completed_updates', 'target_updates', 'current_global_step',
                'latest_development_update', 'dev_pass_at_1_pct', 'dev_pass_at_8_pct']
DEV_FIELDS = ['route_kind', 'student_start_step', 'guidance', 'additional_step',
              'global_step', 'pass_at_1_pct', 'pass_at_8_pct', 'mean_response_tokens',
              'output_limit_pct', 'noncap_missing_answer_pct']


def build(snapshot, source_commit, expected_scope_sha):
    """Use an explicit allowlist so private responses/processes cannot leak."""
    assert snapshot['campaign'] == 'student-state-guidance-03-kk-clean-v4', 'Wrong campaign'
    assert snapshot['scope_sha256'] == expected_scope_sha, 'Scope changed'
    scope = snapshot['active_scope']
    assert scope['tasks'] == ['kk'] and scope['training_seeds'] == [42]
    assert scope['states'] == [30, 90] and set(scope['modes']) == set(MODES)
    assert scope['formal_branches'] == 8 and scope['branch_additional_steps'] == 150
    assert scope['legacy_reference_allowed'] is False
    assert not snapshot['pending_or_invalid'], 'Collector deferred incomplete evidence'
    assert len(source_commit) == 40 and all(c in '0123456789abcdef' for c in source_commit)
    runs = {}
    development = []
    for run in snapshot['runs']:
        start, mode = run['start_step'], run['guidance']
        assert run['task'] == 'kk' and run['training_seed'] == 42
        assert (start == 0 and mode == 'plain') or (start in (30, 90) and mode in MODES)
        assert (start, mode) not in runs, 'Duplicate route'
        assert not run.get('reused_reference'), 'Reused reference is forbidden'
        cfg = run['public_config']
        assert cfg['clean_reference'] is True and cfg['eos_mode'] == 'semantic'
        assert cfg['starting_checkpoint_origin'] == ('official_base' if start == 0 else 'clean_reference_v4')
        count = len(run['train'])
        assert 0 <= count <= 150
        assert [p['additional_step'] for p in run['train']] == list(range(1, count + 1))
        assert [p['global_step'] for p in run['train']] == list(range(start + 1, start + count + 1))
        assert run['status'] in ('running', 'starting', 'completed', 'failed')
        if run['status'] == 'completed':
            assert count == 150, 'Completion requires all 150 updates'
        runs[(start, mode)] = run
        seen = set()
        for point in run['dev']:
            step = point['additional_step']
            assert step not in seen
            seen.add(step)
            assert step in (0, 25, 50, 75, 100, 125, 150) and step <= count
            assert point['global_step'] == start + step
            assert point['questions'] == 512 and point['samples'] == 8
            if step == 0:
                continue  # A starting grade cannot masquerade as a trained grade.
            development.append(dict(route_kind='reference' if start == 0 else 'formal',
                student_start_step=start, guidance=mode, additional_step=step,
                global_step=point['global_step'], pass_at_1_pct=100 * point['pass_at_1'],
                pass_at_8_pct=100 * point['pass_at_8'], mean_response_tokens=point['mean_response_tokens'],
                output_limit_pct=100 * point['output_limit_fraction'],
                noncap_missing_answer_pct=100 * point['no_final_before_limit_fraction']))
    reference = runs.get((0, 'plain'))
    assert reference is not None, 'A fresh reference must exist'
    if any(start > 0 for start, _ in runs):
        assert reference['status'] == 'completed' and len(reference['train']) == 150, 'Reference dependency is incomplete'
    routes = []
    counts = dict(completed=0, running=0, pending=0, failed=0)
    for start in (30, 90):
        for mode in MODES:
            run = runs.get((start, mode))
            state = run['status'] if run else 'waiting_clean_reference'
            count = len(run['train']) if run else 0
            dev = [p for p in development if p['student_start_step'] == start and p['guidance'] == mode]
            latest = max(dev, key=lambda p: p['additional_step']) if dev else None
            routes.append(dict(task='kk', seed=42, student_start_step=start, guidance=mode,
                status=state, completed_updates=count, target_updates=150,
                current_global_step=start + count if run else None,
                latest_development_update=latest['additional_step'] if latest else None,
                dev_pass_at_1_pct=latest['pass_at_1_pct'] if latest else None,
                dev_pass_at_8_pct=latest['pass_at_8_pct'] if latest else None))
            counts[{'waiting_clean_reference': 'pending', 'starting': 'running'}.get(state, state)] += 1
    reference_dev = [p for p in development if p['route_kind'] == 'reference']
    summary = dict(checked_beijing=snapshot['checked_beijing'], campaign='K&K clean-v4',
        training_source_commit=source_commit, phase='formal_training' if any(k[0] for k in runs) else 'fresh_reference_training',
        formal_branches=8, counts=counts, routes=routes, development=development,
        reference=dict(status=reference['status'], completed_updates=len(reference['train']), target_updates=150,
            source='official Qwen3-1.7B-Base; checkpoint resume disabled', full_fork_checkpoints=[30, 90],
            gpus=4, counts_as_formal_branch=False,
            latest_development=max(reference_dev, key=lambda p: p['additional_step']) if reference_dev else None),
        repairs_active_from_reference_step0=['semantic EOS alignment and supervision', 'actual newline serialization',
            'explicit development seed forwarding', 'reject legacy reference reuse'],
        selected_extra_teacher_format_reminder=False, independent_final_test_complete=False,
        state_guidance_interaction_available=False,
        sampling_scope='Development512x8 is separate from fixed-endpoint test1024x32. Step0 grades are excluded.',
        source_snapshot_sha256=hashlib.sha256(json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        scope_sha256=expected_scope_sha,
        swanlab_url='https://swanlab.cn/@Tasselszcx/OPD+RL/runs/d865445ade5e2c67474c',
        historical_results='legacy-format-v3-20261105/README.md')
    return summary


def write(summary, output):
    output.mkdir(parents=True, exist_ok=True)
    existing = output / 'summary.json'
    if existing.exists():
        assert not json.loads(existing.read_text()).get('independent_final_test_complete'), 'Do not overwrite completed final analysis'
    existing.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (output / 'README.md').write_text(
        '# clean-v4当前快照\n\n'
        '[结果解释](../../docs/g3a-results.md) · [summary.json](summary.json) · [八组进度](formal-progress.csv) · '
        '[开发点](development.csv) · [50／100／150节点](milestones.csv) · [验证摘要](validation.json)\n\n'
        '公共参考从官方Base重新生成30／90起点，单列记录，不计入八组正式对照。开发CSV仅包含训练后测验，'
        '每行明确参考／正式路线、起点、测验步数和全局步；空白代表尚未验证，不填入旧成绩或step0分数。\n\n'
        '[旧format-v3探索](legacy-format-v3-20261105/README.md)与[更早18路线快照](legacy-20261104/LEGACY_STATUS.md)'
        '单独保留。权重、原始回答、凭据、进程身份和机器路径不发布。\n')
    for filename, fields, rows in [
        ('formal-progress.csv', ROUTE_FIELDS, summary['routes']),
        ('development.csv', DEV_FIELDS, summary['development']),
        ('milestones.csv', DEV_FIELDS, [p for p in summary['development'] if p['additional_step'] in (50, 100, 150)])]:
        with (output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    ref = summary['reference']
    lines = ['# 当前进度与结果', '', f"快照时间：{summary['checked_beijing']}，**clean-v4**。静态快照；[SwanLab新轮]({summary['swanlab_url']})同步运行日志。", '',
        f"公共参考从官方Base重新训练，完成**{ref['completed_updates']}／150步**，使用4张GPU，不计入正式八组。参考完成并保留新30／90状态后，先跑30起点四组，再跑90起点四组，各追加150步。", '',
        '| 学生起点 | 指导 | 已完成／计划更新 | 状态 | 最近开发测于追加步 | pass@1／pass@8 |', '|---:|---|---:|---|---:|---|']
    for route in summary['routes']:
        status = {'waiting_clean_reference': '等待公共参考完成', 'running': '训练中', 'starting': '初始化', 'completed': '已完成', 'failed': '失败'}[route['status']]
        score = '待验证' if route['latest_development_update'] is None else f"{route['dev_pass_at_1_pct']:.2f}%／{route['dev_pass_at_8_pct']:.2f}%"
        lines.append(f"| {route['student_start_step']} | {MODES[route['guidance']]} | {route['completed_updates']}／150 | {status} | {route['latest_development_update'] or '—'} | {score} |")
    lines += ['', f"正式八组完成**{summary['counts']['completed']}／8**。分数对应表内测验步数，不代表当前未测步数；单训练种子，暂不下指导优劣或状态交互结论。", '',
        '## 训练后开发成绩', '', '| 路线 | 起点 | 指导 | 追加步／全局步 | pass@1 | pass@8 | 平均token | 长度上限率 | 未触限缺答案标签 |', '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for point in summary['development']:
        lines.append(f"| {'公共参考' if point['route_kind']=='reference' else '正式对照'} | {point['student_start_step']} | {MODES[point['guidance']]} | {point['additional_step']}／{point['global_step']} | {point['pass_at_1_pct']:.2f}% | {point['pass_at_8_pct']:.2f}% | {point['mean_response_tokens']:.1f} | {point['output_limit_pct']:.2f}% | {point['noncap_missing_answer_pct']:.2f}% |")
    if not summary['development']:
        lines += ['', '尚无新轮训练后开发测验，不用旧成绩或step0分数代替。']
    lines += ['', '开发集固定512题×8次，每25步记录；固定终点另做独立测试1024题×32次。缺答案标签为0不代表逻辑答案全对，所有不合法或错误答案仍按原评分规则计错。', '',
        '## 历史与检查', '', '修复从参考训练第1步生效，禁止旧参考起点。旧轮结果、调试对照和成本保留在[历史探索](../results/g3a/legacy-format-v3-20261105/README.md)，不计入新轮。', '',
        '[新轮快照](../results/g3a/summary.json) · [八组进度](../results/g3a/formal-progress.csv) · [开发点](../results/g3a/development.csv) · [检查范围](validation.md) · [实验设计](g3a-study.md)']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--training-source-commit', required=True)
    parser.add_argument('--expected-scope-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--results-document', type=Path, required=True)
    parser.add_argument('--figures', type=Path)
    args = parser.parse_args()
    summary = build(json.loads(args.snapshot.read_text()), args.training_source_commit, args.expected_scope_sha256)
    document = write(summary, args.output)
    if args.figures:
        starts = sorted({0} | {p['student_start_step'] for p in summary['development']})
        for start in starts:
            for extension in ('png', 'svg'):
                source = args.figures / f'kk_state{start}_curves.{extension}'
                if source.is_file():
                    destination = args.output / 'figures' / source.name
                    destination.parent.mkdir(exist_ok=True)
                    shutil.copy2(source, destination)
                    if extension == 'svg':
                        destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines()) + '\n')
            if (args.output / 'figures' / f'kk_state{start}_curves.png').exists():
                label = '公共参考（不计入八组）' if start == 0 else f'起点{start}四组'
                document += f'\n![{label}](../results/g3a/figures/kk_state{start}_curves.png)\n'
    args.results_document.parent.mkdir(parents=True, exist_ok=True)
    args.results_document.write_text(document)
    print(json.dumps({'status': 'prepared', 'formal_counts': summary['counts'],
        'reference_updates': summary['reference']['completed_updates'], 'trained_development_points': len(summary['development']),
        'remote_actions': False, 'uploaded': False}))


if __name__ == '__main__':
    main()
