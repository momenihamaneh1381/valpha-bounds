import argparse
import os
import pickle
import sys

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nnlib.nnlib.matplotlib_utils import set_default_configs

set_default_configs(plt, seaborn)


class NestedDict(dict):
    def __missing__(self, key):
        self[key] = type(self)()
        return self[key]


def fit_exp(x, y):
    log_x = np.log(x)
    log_y = np.log(y)

    res_y = np.zeros_like(log_y)
    a, b = np.zeros(log_y.shape[1]), np.zeros(log_y.shape[1])
    for i in range(log_y.shape[1]):
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y[:, i])

        b[i] = slope
        a[i] = np.exp(intercept)
        res_y[:, i] = a[i] * (x ** b[i])

    print(a.mean(), b.mean())
    return res_y


def plot_line(ax, xs, data, label, m=None, ls=None, lw=None):
    mean, std = np.nanmean(data, axis=1), np.nanstd(data, axis=1)
    ax.plot(xs, mean, label=label, marker=m, ls=ls, lw=lw)
    if data.shape[1] > 1:
        ax.fill_between(xs, mean-std, mean+std, alpha=0.15)


SAMPLEWISE_METHOD_LABELS = {
    'theorem8_samplewise_loss_difference_valpha_bound': 'Theorem 8',
    'theorem9_samplewise_single_loss_valpha_bound': 'Theorem 9',
    'theorem10_samplewise_loss_pair_valpha_bound': 'Theorem 10',
    'theorem11_samplewise_binarized_single_loss_valpha_bound': 'Theorem 11',
    'theorem12_samplewise_binarized_loss_difference_valpha_bound': 'Theorem 12',
    'theorem13_samplewise_binarized_loss_pair_valpha_bound': 'Theorem 13',
}

ORIGINAL_BOUND_LABELS = {
    'bound_square': 'Square-Root',
    'bound_fast': 'Fast-Rate',
    'bound_bkl': 'Binary KL',
    'bound_cmi': 'CMI (Oracle)',
    'bound_cshi': 'CSHI (Oracle)',
    'bound_cjsi': 'CJSI (Oracle)',
    'bound_bjs': 'Binary JS (Ours)',
}


def samplewise_gap_bound(result, method, binary):
    """Return the T8--T13 estimated increment for one loss type."""
    if method not in SAMPLEWISE_METHOD_LABELS:
        raise ValueError(f"unsupported sample-wise method: {method}")
    item = result.get(method)
    if item is None:
        return None
    return item.get(f"best_gap_bound_{'01' if binary else 'ce'}")


def original_bound_gap(result, method, binary):
    loss_idx = 0 if binary else 1
    loss_key = 'loss_01' if binary else 'loss_ce'
    if method not in result:
        return None
    return result[method][loss_idx] - result[loss_key][0]


def plot_theorem_bounds(exp_name='mnist-4vs9-CNN-valpha-main', binary=True,
                        ns=None, epoch=None, num_Z=None, results_dir='results',
                        methods=None, figures_dir='../figures',
                        include_original_bounds=True):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    if methods is None:
        methods = list(SAMPLEWISE_METHOD_LABELS)
    original_methods = list(ORIGINAL_BOUND_LABELS) if include_original_bounds else []

    results_file_path = os.path.join(results_dir, exp_name, 'results.pkl')
    with open(results_file_path, 'rb') as f:
        results = pickle.load(f)

    if ns is None:
        ns = sorted(results.keys())
    xs = np.arange(len(ns))

    max_num_Z = num_Z
    if max_num_Z is None:
        max_num_Z = max(len(results[n][epoch if epoch is not None else sorted(results[n].keys())[-1]]) for n in ns)

    loss_key = 'loss_01' if binary else 'loss_ce'
    error_data = np.full((len(ns), max_num_Z), np.nan)
    original_data = {
        method: np.full((len(ns), max_num_Z), np.nan)
        for method in original_methods
    }
    method_data = {
        method: np.full((len(ns), max_num_Z), np.nan)
        for method in methods
    }

    for i, n in enumerate(ns):
        selected_epoch = epoch if epoch is not None else sorted(results[n].keys())[-1]
        seed_results = results[n][selected_epoch]
        for j, result in enumerate(seed_results[:max_num_Z]):
            L = result[loss_key]
            error_data[i, j] = abs(L[1] - L[0])
            for method in original_methods:
                value = original_bound_gap(result=result, method=method, binary=binary)
                if value is not None:
                    original_data[method][i, j] = value
            for method in methods:
                value = samplewise_gap_bound(result=result, method=method, binary=binary)
                if value is not None:
                    method_data[method][i, j] = value

    plot_line(ax, xs, error_data, 'Error', None, ':', 4)
    original_markers = {
        'bound_square': None,
        'bound_fast': '^',
        'bound_bkl': 's',
        'bound_cmi': '*',
        'bound_cshi': 'd',
        'bound_cjsi': 'p',
        'bound_bjs': 'o',
    }
    for method in original_methods:
        data = original_data[method]
        if np.all(np.isnan(data)):
            print(f"Skipping {method}: not present in {results_file_path}")
            continue
        plot_line(ax, xs, data, ORIGINAL_BOUND_LABELS[method],
                  original_markers.get(method), '--', 1)

    markers = {
        'theorem8_samplewise_loss_difference_valpha_bound': '*',
        'theorem9_samplewise_single_loss_valpha_bound': 's',
        'theorem10_samplewise_loss_pair_valpha_bound': 'h',
        'theorem11_samplewise_binarized_single_loss_valpha_bound': '8',
        'theorem12_samplewise_binarized_loss_difference_valpha_bound': '<',
        'theorem13_samplewise_binarized_loss_pair_valpha_bound': '>',
    }
    for method in methods:
        data = method_data[method]
        if np.all(np.isnan(data)):
            print(f"Skipping {method}: not present in {results_file_path}")
            continue
        plot_line(ax, xs, data, SAMPLEWISE_METHOD_LABELS[method], markers.get(method))

    ax.set_xlabel('n')
    ax.set_xticks(xs)
    ax.set_xticklabels(ns)
    ax.set_ylabel('gap or bound increment')
    ax.legend()

    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(os.path.join(figures_dir, f'{exp_name}_theorem_bounds_{("B" if binary else "C")}.pdf'),
                format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_gauss(name, binary):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    results_dir = "results"
    ns = [10, 20, 30, 50, 80, 100, 200]
    num_Z = 50

    data = np.zeros((len(ns), 8, num_Z))
    for i in range(len(ns)):
        results_file_path = os.path.join(results_dir, name, f'results_{ns[i]}.pkl')
        with open(results_file_path, 'rb') as f:
            results = pickle.load(f)

        for j in range(num_Z):
            result = results[j]
            L = result['loss_01' if binary else 'loss_ce']

            data[i, 0, j] = abs(L[1] - L[0])
            data[i, 1, j] = result['bound_square'][0 if binary else 1] - L[0]
            data[i, 2, j] = result['bound_fast'][0 if binary else 1] - L[0]
            data[i, 3, j] = result['bound_bkl'][0 if binary else 1] - L[0]
            data[i, 4, j] = result['bound_cmi'][0 if binary else 1] - L[0]
            data[i, 5, j] = result['bound_cshi'][0 if binary else 1] - L[0]
            data[i, 6, j] = result['bound_cjsi'][0 if binary else 1] - L[0]
            data[i, 7, j] = result['bound_bjs'][0 if binary else 1] - L[0]

    fit_curve = fit_exp(ns, data[:, 0, :])
    plot_line(ax, ns, data[:, 0, :], 'Error', None, ':', 4)
    # plot_line(ax, ns, data[:, 2, :], 'Fast-Rate', '^')
    # plot_line(ax, ns, data[:, 3, :], 'Binary KL', 's')
    # plot_line(ax, ns, data[:, 7, :], 'Binary JS (Ours)', 'o')
    # plot_line(ax, ns, data[:, 4, :], 'CMI (Oracle)', '*')
    # plot_line(ax, ns, data[:, 5, :], 'CSHI (Oracle)', 'd')
    # plot_line(ax, ns, data[:, 6, :], 'CJSI (Oracle)', 'p')
    plot_line(ax, ns, fit_curve, 'Fit')

    ax.set_xlabel('n')
    if np.max(data) > 1:
        ax.set_ylim(top=1)
    ax.legend()

    fig.savefig(f'../figures/{name}_{("B" if binary else "C")}.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_n(name, binary, ep=0.0):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    results_dir = "results"
    if name == "mnist":
        ns, num_Z, epochs, exp_name = [75, 250, 1000, 4000], 5, 200, 'mnist-4vs9-CNN'
    elif name == "mnist-noise":
        ns, num_Z, epochs = [75, 250, 1000, 4000], 5, 200
        exp_name = f'mnist-4vs9-CNN-noise-{ep}'
        name += f'-{ep}'
    else:
        ns, num_Z, epochs, exp_name = [1000, 5000, 20000], 2, 40, 'cifar10-pretrained-resnet50'

    with open(os.path.join(results_dir, exp_name, 'results.pkl'), 'rb') as f:
        results = pickle.load(f)

    data = np.full((len(ns), 8, num_Z), np.nan)
    for i, n in enumerate(ns):
        for j in range(num_Z):
            result = results[n][epochs][j]
            L = result['loss_01' if binary else 'loss_ce']
            data[i, 0, j] = abs(L[1] - L[0])
            for k, method in enumerate(ORIGINAL_BOUND_LABELS, 1):
                data[i, k, j] = result[method][0 if binary else 1] - L[0]

    fit_curve = fit_exp(np.asarray(ns), data[:, 0, :])
    xs = np.arange(len(ns))
    plot_line(ax, xs, data[:, 0, :], 'Error', None, ':', 4)
    plot_line(ax, xs, fit_curve, 'Fit')
    ax.set_xlabel('n')
    ax.set_xticks(xs)
    ax.set_xticklabels(ns)
    if np.max(data) > 1:
        ax.set_ylim(top=1)
    if not name.startswith("mnist-noise") or ep < 0.1:
        ax.legend()
    fig.savefig(f'../figures/{name}_n_{("B" if binary else "C")}.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_epoch(name, binary):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    results_dir = "results"
    if name == "mnist":
        epochs = np.arange(1, 11) * 4
        num_Z = 5
        n = 4000
        exp_name = 'mnist-4vs9-CNN-LD'
    else:
        epochs = np.arange(1, 9) * 2
        num_Z = 1
        n = 20000
        exp_name = 'cifar10-pretrained-resnet50-LD'

    results_file_path = os.path.join(results_dir, exp_name, 'results.pkl')
    with open(results_file_path, 'rb') as f:
        results = pickle.load(f)

    data = np.zeros((len(epochs), 8, num_Z))
    for i in range(len(epochs)):

        for j in range(num_Z):
            result = results[n][epochs[i]][j]
            L = result['loss_01' if binary else 'loss_ce']

            data[i, 0, j] = abs(L[1] - L[0])
            data[i, 1, j] = result['bound_square'][0 if binary else 1] - L[0]
            data[i, 2, j] = result['bound_fast'][0 if binary else 1] - L[0]
            data[i, 3, j] = result['bound_bkl'][0 if binary else 1] - L[0]
            data[i, 4, j] = result['bound_cmi'][0 if binary else 1] - L[0]
            data[i, 5, j] = result['bound_cshi'][0 if binary else 1] - L[0]
            data[i, 6, j] = result['bound_cjsi'][0 if binary else 1] - L[0]
            data[i, 7, j] = result['bound_bjs'][0 if binary else 1] - L[0]

    plot_line(ax, epochs, data[:, 0, :], 'Error', None, ':', 4)
    # plot_line(ax, epochs, data[:, 1, :], 'Square-Root')
    plot_line(ax, epochs, data[:, 2, :], 'Fast-Rate', '^')
    plot_line(ax, epochs, data[:, 3, :], 'Binary KL', 's')
    plot_line(ax, epochs, data[:, 7, :], 'Binary JS (Ours)', 'o')
    plot_line(ax, epochs, data[:, 4, :], 'CMI (Oracle)', '*')
    plot_line(ax, epochs, data[:, 5, :], 'CSHI (Oracle)', 'd')
    plot_line(ax, epochs, data[:, 6, :], 'CJSI (Oracle)', 'p')

    ax.set_xlabel('Epoch')
    if np.max(data) > 1:
        ax.set_ylim(top=1)

    fig.savefig(f'../figures/{name}_epoch_{("B" if binary else "C")}.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--theorem_bounds', action='store_true',
                        help='plot theorem/baseline bound increments from a parsed results.pkl')
    parser.add_argument('--exp_name', default='mnist-4vs9-CNN-valpha-main')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--figures_dir', default='../figures')
    parser.add_argument('--loss_type', choices=['01', 'ce'], default='01')
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--hide_original_bounds', action='store_true',
                        help='only plot T8--T13, without original BinaryJS bounds')
    args = parser.parse_args()

    if args.theorem_bounds:
        plot_theorem_bounds(exp_name=args.exp_name,
                            binary=args.loss_type == '01',
                            epoch=args.epoch,
                            results_dir=args.results_dir,
                            figures_dir=args.figures_dir,
                            include_original_bounds=not args.hide_original_bounds)
        raise SystemExit

    plot_gauss("two", True)
    # plot_gauss("two", False)
    plot_gauss("ten", True)
    # plot_gauss("ten", False)

    plot_n('mnist', True)
    # plot_n('mnist', False)
    plot_n('cifar10', True)
    # plot_n('cifar10', False)

    # plot_epoch('mnist', True)
    # plot_epoch('mnist', False)
    # plot_epoch('cifar10', True)
    # plot_epoch('cifar10', False)

    # plot_n('mnist-noise', True, 0.05)
    # plot_n('mnist-noise', False, 0.05)
    # plot_n('mnist-noise', True, 0.1)
    # plot_n('mnist-noise', False, 0.1)
    # plot_n('mnist-noise', True, 0.15)
    # plot_n('mnist-noise', False, 0.15)

    pass
