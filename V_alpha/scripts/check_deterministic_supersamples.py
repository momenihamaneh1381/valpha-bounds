"""Preflight augmented runs for fixed deterministic supersample evaluation."""

import argparse
import os
import pickle

from scripts.parse_results import (
    get_supersample_identity,
    select_supersample_for_evaluation,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--ns", nargs="+", type=int, required=True)
    parser.add_argument("--n_seeds", type=int, required=True)
    parser.add_argument("--n_selector_seeds", type=int, required=True)
    args = parser.parse_args()

    failures = []
    checked = 0
    for n in args.ns:
        for seed in range(args.n_seeds):
            expected_indices = None
            for selector in range(args.n_selector_seeds):
                run_dir = os.path.join(
                    args.results_dir,
                    args.exp_name,
                    f"n={n},seed={seed},S_seed={selector}",
                )
                path = os.path.join(run_dir, "saved_data.pkl")
                if not os.path.isfile(path):
                    failures.append(f"missing {path}")
                    continue
                try:
                    with open(path, "rb") as f:
                        saved_data = pickle.load(f)
                    dataset, key = select_supersample_for_evaluation(
                        saved_data,
                        require_deterministic=True,
                    )
                    if key != "all_examples_wo_data_aug":
                        raise ValueError(f"unexpected evaluation key {key!r}")
                    indices = get_supersample_identity(saved_data, dataset)
                    if indices is not None:
                        indices = tuple(int(x) for x in indices)
                        if expected_indices is None:
                            expected_indices = indices
                        elif indices != expected_indices:
                            raise ValueError(
                                "ordered raw supersample indices differ across selector runs"
                            )
                    checked += 1
                except Exception as exc:  # reviewer-facing aggregate preflight
                    failures.append(f"{path}: {exc}")

    if failures:
        print("Deterministic supersample preflight FAILED:")
        for item in failures[:50]:
            print(f"  - {item}")
        if len(failures) > 50:
            print(f"  ... and {len(failures) - 50} more")
        raise SystemExit(1)

    print(
        f"Deterministic supersample preflight passed for {checked} saved runs "
        f"under {args.exp_name}."
    )


if __name__ == "__main__":
    main()
