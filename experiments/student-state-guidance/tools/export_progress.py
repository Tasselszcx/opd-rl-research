"""Read existing outputs and export portable development tables and curves.

No model is loaded, no generation is requested, and no training file is modified.
"""
import argparse
import csv
import json
from pathlib import Path
from collect_progress import collect

LABELS = {'plain': 'Plain OPD', 'answer_only': 'Answer only',
          'full_reasoning': 'Full reasoning', 'key_steps': 'Key steps',
          'error_targeted': 'Error targeted'}
COLORS = dict(zip(LABELS, ['#2563eb', '#64748b', '#dc2626', '#059669', '#9333ea']))


def export(root, output, plot=True):
    data = collect(root)
    output.mkdir(parents=True, exist_ok=True)
    points, routes = [], []
    for run in data['runs']:
        last = run['dev'][-1] if run['dev'] else None
        routes.append({'run': run['run'], 'task': run['task'], 'training_seed': run['training_seed'],
                       'student_start_step': run['start_step'], 'guidance': run['guidance'],
                       'status': run['status'], 'completed_updates': len(run['train']),
                       'reused_reference':run.get('reused_reference',False),
                       'latest_validation_update': last['additional_step'] if last else None})
        for point in run['dev']:
            points.append({'run': run['run'], 'task': run['task'], 'training_seed': run['training_seed'],
                           'student_start_step': run['start_step'], 'guidance': run['guidance'],
                           'additional_step': point['additional_step'], 'global_step': point['global_step'],
                           'pass_at_1_pct': 100 * point['pass_at_1'], 'pass_at_8_pct': 100 * point['pass_at_8'],
                           'mean_response_tokens': point['mean_response_tokens'],
                           'output_limit_pct': 100 * point['output_limit_fraction'],
                           'noncap_missing_answer_pct':100*point['no_final_before_limit_fraction'] if 'no_final_before_limit_fraction' in point else None,
                           'reused_reference':run.get('reused_reference',False)})
    for filename, rows in [('routes.csv', routes), ('development.csv', points),
                           ('milestones.csv', [p for p in points if not p['reused_reference'] and p['additional_step'] in [50, 100, 150]])]:
        if not rows:
            continue
        with (output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {'checked_beijing': data['checked_beijing'],
               'measurement': 'development only: 512 questions x 8 samples; final tests are separate',
               'routes': routes, 'development': points, 'deferred_or_invalid': data['pending_or_invalid'],
               'scope_sha256': data['scope_sha256'], 'input_sha256': data['source_sha256']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for task, state in sorted({(r['task'], r['start_step']) for r in data['runs']}):
            fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout='constrained')
            for run in [r for r in data['runs'] if (r['task'], r['start_step']) == (task, state)]:
                dev, train = run['dev'], run['train']
                label = ('Historical reference' if run.get('reused_reference') else LABELS[run['guidance']]) + ' / seed' + str(run['training_seed'])
                color = COLORS[run['guidance']]
                for ax, key in zip(axes[0], ['pass_at_1', 'pass_at_8']):
                    ax.plot([p['additional_step'] for p in dev], [100 * p[key] for p in dev],
                            marker='o', color=color, label=label)
                for ax, key in zip(axes[1], ['critic/score/mean', 'response_length/mean']):
                    ax.plot([p['additional_step'] for p in train], [p[key] for p in train], color=color)
            for ax, title in zip(axes.flat, ['Development pass@1 (%)', 'Development pass@8 (%)',
                                             'Training reward (not development accuracy)', 'Training response length (tokens)']):
                ax.set_title(title)
                ax.set_xlabel('Additional training updates')
                ax.set_xlim(-2, 152)
                ax.grid(alpha=.2)
            for ax in axes[0]:
                ax.set_ylim(0, 100)
            axes[0, 0].legend(fontsize=8)
            fig.suptitle('G3a / ' + task + ' / student start ' + str(state) + '\n' +
                         data['checked_beijing'][:16] + ' / development only')
            for extension in ['png', 'svg']:
                fig.savefig(output / (task + '_state' + str(state) + '.' + extension), dpi=160)
            plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    root, output = args.root.resolve(), args.out.resolve()
    protected = [root / p for p in ['runs', 'checks', 'data', 'source_data', 'evaluations']]
    if output == root or root.is_relative_to(output) or any(output.is_relative_to(p) for p in protected):
        parser.error('--out must be a separate report directory, not training/input storage')
    result = export(root, output, not args.no_plots)
    print(json.dumps({'routes': len(result['routes']), 'development_points': len(result['development']),
                      'deferred_or_invalid': len(result['deferred_or_invalid']), 'gpu_generation_performed': False}))


if __name__ == '__main__':
    main()
