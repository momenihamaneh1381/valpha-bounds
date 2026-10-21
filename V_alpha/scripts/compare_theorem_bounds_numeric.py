# scripts/compare_theorem_bounds_numeric.py

import argparse
import pickle
import sys
import types
import numpy as np


class NestedDict(dict):
    def __missing__(self, key):
        self[key] = type(self)()
        return self[key]


parse_results_module = types.ModuleType("scripts.parse_results")
parse_results_module.NestedDict = NestedDict
sys.modules.setdefault("scripts.parse_results", parse_results_module)


METHODS = [
    "theorem8_samplewise_loss_difference_valpha_bound",
    "theorem9_samplewise_single_loss_valpha_bound",
    "theorem10_samplewise_loss_pair_valpha_bound",
    "theorem11_samplewise_binarized_single_loss_valpha_bound",
    "theorem12_samplewise_binarized_loss_difference_valpha_bound",
    "theorem13_samplewise_binarized_loss_pair_valpha_bound",
]

LABELS = {
    "theorem8_samplewise_loss_difference_valpha_bound": "T8",
    "theorem9_samplewise_single_loss_valpha_bound": "T9",
    "theorem10_samplewise_loss_pair_valpha_bound": "T10",
    "theorem11_samplewise_binarized_single_loss_valpha_bound": "T11",
    "theorem12_samplewise_binarized_loss_difference_valpha_bound": "T12",
    "theorem13_samplewise_binarized_loss_pair_valpha_bound": "T13",
}


def latest_epoch(results, n, epoch):
    if epoch is not None:
        return epoch
    return sorted(results[n].keys())[-1]


def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    return float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def get_signed_gap(result, loss_type):
    train, ghost = result[f"loss_{loss_type}"]
    return float(ghost - train)


def get_gap_magnitude(result, loss_type):
    return abs(get_signed_gap(result, loss_type))


def get_bound(result, method, loss_type):
    proxy = result.get(method)
    if proxy is None:
        return np.nan
    return float(proxy.get(f"best_gap_bound_{loss_type}", np.nan))


def get_best_alpha(result, method, loss_type):
    proxy = result.get(method)
    if proxy is None:
        return np.nan
    return proxy.get(f"best_alpha_{loss_type}", np.nan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/mnist-4vs9-CNN-valpha-main/results.pkl")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--loss_type", choices=["01", "ce"], default="01")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        results = pickle.load(f)

    ns = sorted(results.keys())

    header = [
        "n",
        "signed_gap_mean",
        "abs_gap_mean",
        "abs_gap_std",
    ]
    for m in METHODS:
        label = LABELS[m]
        header += [
            f"{label}_mean",
            f"{label}_std",
            f"{label}_minus_error",
            f"{label}_ratio",
            f"{label}_best_alpha_mode",
        ]
    print(",".join(header))

    for n in ns:
        epoch = latest_epoch(results, n, args.epoch)
        seed_results = results[n][epoch]

        signed_gaps = [get_signed_gap(r, args.loss_type) for r in seed_results]
        abs_gaps = [abs(g) for g in signed_gaps]
        signed_gap_mean = float(np.mean(signed_gaps))
        err_mean, err_std = mean_std(abs_gaps)

        if err_mean <= 1e-12:
            raise ValueError(
                f"mean |empirical gap| is too small for a stable ratio at n={n}"
            )

        row = [
            str(n),
            f"{signed_gap_mean:.8g}",
            f"{err_mean:.8g}",
            f"{err_std:.8g}",
        ]

        for m in METHODS:
            vals = [get_bound(r, m, args.loss_type) for r in seed_results]
            vals = [v for v in vals if np.isfinite(v)]
            if len(vals) == 0:
                row += ["nan", "nan", "nan", "nan", "nan"]
                continue

            b_mean, b_std = mean_std(vals)
            diff = b_mean - err_mean
            ratio = b_mean / err_mean

            alphas = [get_best_alpha(r, m, args.loss_type) for r in seed_results]
            alpha_mode = max(set(alphas), key=alphas.count)

            row += [
                f"{b_mean:.8g}",
                f"{b_std:.8g}",
                f"{diff:.8g}",
                f"{ratio:.8g}",
                str(alpha_mode),
            ]

        print(",".join(row))


if __name__ == "__main__":
    main()