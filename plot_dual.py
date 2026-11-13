import argparse
import json
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

large_size = 30
small_size = 30

plt.rcParams.update({
    'figure.autolayout': True,
    'font.size': small_size,
    'axes.labelsize': large_size,
    'xtick.labelsize': small_size,
    'ytick.labelsize': small_size,
    'axes.linewidth': 1.1,
    'xtick.major.width': 1.1,
    'ytick.major.width': 1.1,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'axes.labelpad': 2.0,
})

DEFAULT_LW = 2.8
DEFAULT_MS = 8.5
DEFAULT_STAR_MEW = 0.7

# Larger markers at 1000 RANSAC iterations: stars without ENM, X with ENM
HIGHLIGHT_ITERS = 1000
HIGHLIGHT_MS = 20.0
HIGHLIGHT_MEW = 1.0

LEGEND_SCALE = 2.3
LEGEND_LW = DEFAULT_LW * LEGEND_SCALE
LEGEND_MS = DEFAULT_MS * LEGEND_SCALE

# Colors of the paper's figures
COLORS = {'5pE': '#1f77b4', '5pF-V-Umlaut': '#d62728'}

# Top row: all methods, bottom row: top performers
EXPERIMENT_GROUPS = [['5pE', '5pF-V-Umlaut', '5pF-V-Umlaut + ENM (5p)'],
                     ['5pE', '5pF-V-Umlaut + ENM (5p)']]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--title', type=str, required=True)
    parser.add_argument('json_files', nargs='+', help='one json per scene, results of all scenes are pooled')

    return parser.parse_args()


def load_results(json_files):
    # (experiment, iterations) -> runtimes and pose errors pooled over all files
    out = {}
    for json_file in json_files:
        print(f'Loading: {json_file}')
        with open(json_file, 'r') as f:
            results = json.load(f)
        if isinstance(results, dict):
            results = results['results']
        for x in results:
            runtimes, errs = out.setdefault((x['experiment'], x['info']['iterations']), ([], []))
            runtimes.append(x['info']['runtime'])
            errs.append(max(x['R_err'], x['t_err']))
    return out


def get_curve(data, experiment):
    iterations_list = sorted(it for exp, it in data.keys() if exp == experiment and it is not None)
    xs, maa, med = [], [], []
    for iterations in iterations_list:
        runtimes, errs = data[(experiment, iterations)]
        errs = np.array(errs)
        errs[np.isnan(errs)] = 180
        xs.append(np.mean(runtimes))
        maa.append(np.mean(np.array([np.sum(errs < t) / len(errs) for t in range(1, 11)])))
        med.append(np.median(errs))
    return iterations_list, xs, maa, med


def draw(ax, curves, experiments, metric):
    for experiment in experiments:
        iterations_list, xs, maa, med = curves[experiment]
        ys = maa if metric == 'maa' else med
        color = COLORS[experiment.split(' ')[0]]
        enm = 'ENM' in experiment
        ax.semilogx(xs, ys, label=experiment, marker='*', color=color, linestyle='dashed' if enm else 'solid',
                    linewidth=DEFAULT_LW, markersize=DEFAULT_MS,
                    markeredgewidth=DEFAULT_STAR_MEW, markeredgecolor='black')
        if HIGHLIGHT_ITERS in iterations_list:
            i = iterations_list.index(HIGHLIGHT_ITERS)
            ax.semilogx([xs[i]], [ys[i]], marker='X' if enm else '*', linestyle='None', color=color,
                        markersize=HIGHLIGHT_MS, markeredgewidth=HIGHLIGHT_MEW, markeredgecolor='black', zorder=5)

    ax.set_xlabel('Mean runtime (ms)', fontsize=small_size)
    if metric == 'maa':
        ax.set_ylabel('mAA@10$^\\circ$', fontsize=small_size)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    else:
        ax.set_ylabel('Median Pose Error ($^\\circ$)', fontsize=small_size)
    ax.grid(True, alpha=0.3)


def main():
    args = parse_args()
    data = load_results(args.json_files)

    curves = {exp: get_curve(data, exp) for exp in EXPERIMENT_GROUPS[0]}
    for exp, (iterations_list, xs, maa, med) in curves.items():
        for row in zip(iterations_list, xs, maa, med):
            print(f'{exp:24s} iters {row[0]:5d}  runtime {row[1]:8.2f} ms  mAA@10 {row[2]:.4f}  median {row[3]:.3f}')

    fig, axes = plt.subplots(2, 2, figsize=(2 * 8.6, 2 * 6.4))
    for row, experiments in zip(axes, EXPERIMENT_GROUPS):
        draw(row[0], curves, experiments, 'maa')
        draw(row[1], curves, experiments, 'median')

    handles = [Line2D([0], [0], color=c, marker='*', markeredgecolor='black', linewidth=LEGEND_LW, markersize=LEGEND_MS)
               for c in COLORS.values()]
    handles += [Line2D([0], [0], color='0.25', linestyle=s, linewidth=LEGEND_LW) for s in ['solid', 'dashed']]
    labels = list(COLORS.keys()) + ['w/o ENM', 'w/ ENM']
    # two columns filled column-wise: methods in the first legend row, line styles in the second
    order = [0, 2, 1, 3]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=2.8, columnspacing=2.0)

    os.makedirs('figs', exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(f'figs/{args.title}_dual_experiments.{ext}', bbox_inches='tight', pad_inches=0.06)
    print(f'Saved: figs/{args.title}_dual_experiments.pdf')


if __name__ == '__main__':
    main()
