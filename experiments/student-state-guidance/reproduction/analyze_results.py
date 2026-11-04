"""Analyze all four fixed final checkpoints, separating seen and unseen suites."""
from pathlib import Path
import csv
import json
import numpy as np
from run_phase import ROOT, atomic

METHODS = ['grpo', 'opd', 'opd_then_rl', 'kdrl_anneal']
LABELS = {'grpo': '纯 RL', 'opd': '纯 OPD', 'opd_then_rl': '60 OPD → 90 RL', 'kdrl_anneal': 'KDRL-Annealing'}


def main():
    if json.loads((ROOT / 'study_state.json').read_text())['status'] != 'completed':
        raise RuntimeError('Training and fixed evaluation must both complete')
    output, arrays = {}, {}
    # Check that continuation consumed the same fixed training order.
    def batches(name):
        path = ROOT / 'runs' / name / 'checkpoints/metrics.jsonl'
        return {r['step']: r['prompt_indices'] for r in
                (json.loads(line) for line in path.read_text().splitlines())}
    orders = {m: batches(n) for m, n in [('grpo', 'grpo-001'), ('opd', 'opd-001'),
              ('opd_then_rl', 'opd-then-rl-001'), ('kdrl_anneal', 'kdrl-anneal-001')]}
    assert set(orders['grpo']) == set(range(1, 151))
    for method in ['opd', 'kdrl_anneal']:
        assert orders[method] == orders['grpo'], 'Training order differs: ' + method
    assert orders['opd_then_rl'] == {k: v for k, v in orders['grpo'].items() if k > 60}
    trained_indices = {i for batch in orders['grpo'].values() for i in batch}
    author_eval_actually_seen = sum((i + 40) in trained_indices for i in range(512))
    paired_rows = []
    for suite in ['author_test', 'unseen_test']:
        output[suite], arrays[suite] = {}, {}
        expected_hashes = {r['index']: r['question_sha256'] for r in
            (json.loads(s) for s in (ROOT / 'data' / f'{suite}.jsonl').read_text().splitlines())}
        for method in METHODS:
            by_question = {}
            for file in sorted((ROOT / 'evaluations' / method).glob('rank_*.jsonl')):
                for line in file.read_text().splitlines():
                    row = json.loads(line)
                    if row['suite'] != suite:
                        continue
                    assert row['question_sha256'] == expected_hashes[row['index']]
                    scores = by_question.setdefault(row['index'], {})
                    assert row['sample'] not in scores, 'Duplicate evaluation response'
                    assert row['score'] in (0, 1)
                    scores[row['sample']] = row['score']
            assert set(by_question) == set(range(512))
            assert all(set(s) == set(range(32)) for s in by_question.values())
            matrix = np.array([[by_question[i][j] for j in range(32)] for i in range(512)])
            arrays[suite][method] = matrix.mean(axis=1)
            output[suite][method] = {'pass_at_1': float(matrix.mean()),
                                    'pass_at_32': float(matrix.max(axis=1).mean()),
                                    'questions': 512, 'responses': 16384}
        for i in range(512):
            paired_rows.append({'suite': suite, 'index': i, 'question_sha256': expected_hashes[i],
                                **{m: float(arrays[suite][m][i]) for m in METHODS}})
    comparisons = {}
    rng = np.random.default_rng(20261019)
    for suite in arrays:
        comparisons[suite] = {}
        for baseline in ['grpo', 'opd', 'kdrl_anneal']:
            delta = arrays[suite]['opd_then_rl'] - arrays[suite][baseline]
            means = rng.choice(delta, size=(10000, 512), replace=True).mean(axis=1)
            comparisons[suite][baseline] = {'difference': float(delta.mean()),
                'question_bootstrap_ci95': np.quantile(means, [0.025, 0.975]).tolist()}
    events = [json.loads(p.read_text()) for p in ROOT.glob('**/launcher_record.json')]
    inherited_ledger = ROOT / 'prior_cost_ledger.json'
    prior_events = json.loads(inherited_ledger.read_text())['events'] if inherited_ledger.exists() else []
    all_events = [*prior_events, *events]
    result = {'status': 'completed', 'metrics': output, 'comparisons': comparisons,
              'total_allocated_gpu_hours': sum(e.get('allocated_gpu_hours', 0) for e in all_events),
              'current_schedule_gpu_hours': sum(e.get('allocated_gpu_hours', 0) for e in events),
              'prior_schedule_and_setup_gpu_hours': sum(e.get('allocated_gpu_hours', 0) for e in prior_events),
              'estimated_cost_events': sum(bool(e.get('cost_estimated')) for e in all_events),
              'single_training_seed': True,
              'training_order_matches_all_four_methods': True,
              'author_test_questions_actually_consumed_during_training': author_eval_actually_seen,
              'bootstrap_scope': 'paired question means over 32 completions; excludes training-seed uncertainty',
              'warning': 'author_test 512/512 overlaps training in the released code; only unseen_test tests unseen questions'}
    atomic(ROOT / 'study_results.json', result)
    with (ROOT / 'paired_question_scores.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['suite', 'index', 'question_sha256', *METHODS])
        writer.writeheader(); writer.writerows(paired_rows)
    report = '# Sequential Beats Joint：K&K 四组复现结果\n'
    report += '原发布代码的评测 seed=42 导致 512/512 道题包含在训练集中，因此同时报告原评测口径和预先固定的独立题目结果。\n'
    report += f'在这次实际消耗的 150 个训练 batch 中，原评测题目有 {author_eval_actually_seen}/512 道被用过；四组的训练题顺序核对一致。\n'
    for suite, title in [('author_test', '原评测题目（与训练集重合）'), ('unseen_test', '独立题目（seed=1000000，无训练题重合）')]:
        report += f'\n**{title}**\n\n| 方法 | pass@1 | pass@32 |\n|---|---:|---:|\n'
        for method in METHODS:
            row = output[suite][method]
            report += f"| {LABELS[method]} | {row['pass_at_1']:.2%} | {row['pass_at_32']:.2%} |\n"
        report += '\nOPD→RL 相对各对照的 pass@1 差值与题目级配对 bootstrap 95% 区间：\n\n'
        for baseline, comp in comparisons[suite].items():
            lo, hi = comp['question_bootstrap_ci95']
            report += f"- 相对 {LABELS[baseline]}：{100*comp['difference']:+.2f} 个百分点，[{100*lo:+.2f}, {100*hi:+.2f}]。\n"
    report += f"\n累计记账 GPU 时间 {result['total_allocated_gpu_hours']:.3f} 小时，包含兼容检查、失败尝试、正式训练和评测，其中 {result['estimated_cost_events']} 条前期检查成本为显式标注的估计。四组按论文总步数对齐；这不代表总计算量相同。\n"
    if prior_events:
        report += f"用户授权将调度从单组 8 卡改成两组各 4 卡并行；原 8 卡尝试与准备工作计 {result['prior_schedule_and_setup_gpu_hours']:.3f} GPU 小时，已保留并计入总账。四组最终比较均来自统一 4 卡配置，未混入中止的 8 卡训练结果。\n"
    report += '本轮保留作者代码中的 ref 前向：纯 RL 的更新不使用 teacher 信号，但仍有 teacher 计算开销。OPD→RL 复用了纯 OPD 在 step 60 的真实 checkpoint，研究总账不重复收费该前缀。\n'
    report += '这里只运行一套训练随机性配置；题目 bootstrap 不是训练种子区间。B300、现代依赖、恢复的遗漏源码与显式随机种子均有记录。不能把数值差异单独解释成论文结论成立或失败。\n'
    report += '未按测试分数选择 checkpoint：全部使用预定全局 step 150。没有增加调参、多种子或其他方法。\n'
    (ROOT / 'REPRODUCTION_RESULTS.md').write_text(report)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
