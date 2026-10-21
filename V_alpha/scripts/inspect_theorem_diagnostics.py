import argparse
import pickle


class NestedDict(dict):
    def __missing__(self, key):
        self[key] = type(self)()
        return self[key]

def latest_epoch_for_n(results, n, requested_epoch=None):
    epochs = sorted(results[n].keys())
    if requested_epoch is not None:
        if requested_epoch not in results[n]:
            raise KeyError(
                f"epoch {requested_epoch} not found for n={n}; available epochs={epochs}"
            )
        return requested_epoch
    return epochs[-1]






def print_theorem8(result):
    key = 'theorem8_samplewise_loss_difference_valpha_bound'
    if key not in result:
        print("Theorem 8: missing")
        return

    r8 = result[key]
    print("\n=== Theorem 8 diagnostics ===")
    print("method:", r8.get('method'))
    print("conditions_on:", r8.get('conditions_on'))
    print("num selector seeds:", r8.get('num_selector_seeds'))
    print("num supersample pairs:", r8.get('num_supersample_pairs'))
    print("coordinatewise:", r8.get('coordinatewise'))
    print("selector prior:", r8.get('selector_prior'))
    print("bounded M 01:", r8.get('bounded_loss_M_01'))
    print("bounded M ce:", r8.get('bounded_loss_M_ce'))
    print("loss difference info 01:", r8.get('loss_difference_info_01'))
    print("loss difference info ce:", r8.get('loss_difference_info_ce'))
    print("warnings:", r8.get('warnings'))

    for item in r8.get('by_alpha', []):
        print(
            "alpha:", item.get('alpha'),
            "V01:", item.get('valpha_01'),
            "Vce:", item.get('valpha_ce'),
            "gap_bound_01:", item.get('gap_bound_01'),
            "gap_bound_ce:", item.get('gap_bound_ce'),
            "coordinate summary 01:", item.get('valpha_coordinate_summary_01'),
            "coordinate summary ce:", item.get('valpha_coordinate_summary_ce'),
        )

def print_theorem9(result):
    key = 'theorem9_samplewise_single_loss_valpha_bound'
    if key not in result:
        print("Theorem 9: missing")
        return

    r9 = result[key]
    print("\n=== Theorem 9 diagnostics ===")
    print("method:", r9.get('method'))
    print("conditions_on:", r9.get('conditions_on'))
    print("selected_form:", r9.get('selected_form'))
    print("num selector seeds:", r9.get('num_selector_seeds'))
    print("num supersample pairs:", r9.get('num_supersample_pairs'))
    print("coordinatewise:", r9.get('coordinatewise'))
    print("selector prior:", r9.get('selector_prior'))
    print("bounded M 01:", r9.get('bounded_loss_M_01'))
    print("bounded M ce:", r9.get('bounded_loss_M_ce'))
    print("center 01:", r9.get('center_01'))
    print("center ce:", r9.get('center_ce'))
    print("single loss info 01:", r9.get('single_loss_info_01'))
    print("single loss info ce:", r9.get('single_loss_info_ce'))
    print("warnings:", r9.get('warnings'))

    for item in r9.get('by_alpha', []):
        print(
            "alpha:", item.get('alpha'),
            "V01:", item.get('valpha_01'),
            "Vce:", item.get('valpha_ce'),
            "weighted_01:", item.get('weighted_gap_bound_01'),
            "weighted_ce:", item.get('weighted_gap_bound_ce'),
            "bounded_01:", item.get('bounded_gap_bound_01'),
            "bounded_ce:", item.get('bounded_gap_bound_ce'),
            "gap_bound_01:", item.get('gap_bound_01'),
            "gap_bound_ce:", item.get('gap_bound_ce'),
            "coordinate summary 01:", item.get('valpha_coordinate_summary_01'),
            "coordinate summary ce:", item.get('valpha_coordinate_summary_ce'),
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        default='results/mnist-4vs9-CNN-valpha-main/results.pkl',
        help='path to parsed results.pkl',
    )
    parser.add_argument(
        '--n',
        type=int,
        default=75,
        help='sample size n to inspect',
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=None,
        help='epoch to inspect; default is latest available for this n',
    )
    parser.add_argument(
        '--seed_index',
        type=int,
        default=0,
        help='outer supersample seed index inside results[n][epoch]',
    )
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        results = pickle.load(f)

    epoch = latest_epoch_for_n(results, args.n, args.epoch)
    seed_results = results[args.n][epoch]

    if args.seed_index >= len(seed_results):
        raise IndexError(
            f"seed_index={args.seed_index} out of range; "
            f"available seed results={len(seed_results)}"
        )

    result = seed_results[args.seed_index]

    print("=== Selected parsed result ===")
    print("input:", args.input)
    print("n:", args.n)
    print("epoch:", epoch)
    print("seed_index:", args.seed_index)
    print("available keys:", sorted(result.keys()))
    print("loss_01:", result.get('loss_01'))
    print("loss_ce:", result.get('loss_ce'))

    print_theorem8(result)
    print_theorem9(result)


if __name__ == '__main__':
    main()