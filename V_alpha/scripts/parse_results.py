import argparse
import os
import pickle
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from nnlib.nnlib import utils

from modules.bound_utils import (
    estimate_mi_bound_discrete,
    estimate_mi_bound_continuous,
    estimate_theorem8_samplewise_loss_difference_valpha_bound,
    estimate_theorem9_samplewise_single_loss_valpha_bound,
    estimate_theorem10_samplewise_loss_pair_valpha_bound,
    estimate_theorem11_samplewise_binarized_single_loss_valpha_bound,
    estimate_theorem12_samplewise_binarized_loss_difference_valpha_bound,
    estimate_theorem13_samplewise_binarized_loss_pair_valpha_bound,
)


class NestedDict(dict):
    def __missing__(self, key):
        self[key] = type(self)()
        return self[key]


CE_CLIP_MIN = 0.0
CE_CLIP_MAX = 1.0


def clipped_cross_entropy(preds, labels):
    """Per-example evaluation CE used by the paper: min(max(CE, 0), 1)."""
    return F.cross_entropy(preds, labels, reduction='none').clamp(
        CE_CLIP_MIN,
        CE_CLIP_MAX,
    )


def _namespace_or_dict_value(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _dataset_source_indices(dataset):
    include_indices = getattr(dataset, 'include_indices', None)
    if include_indices is None:
        return None
    return np.asarray(include_indices)


def get_supersample_identity(saved_data, dataset):
    """Return the strongest available ordered identity for a saved supersample."""
    metadata = saved_data.get('supersample_metadata')
    if isinstance(metadata, dict) and metadata.get('raw_train_indices') is not None:
        return np.asarray(metadata['raw_train_indices'])
    return _dataset_source_indices(dataset)


def select_supersample_for_evaluation(saved_data, require_deterministic=False):
    """Return the fixed supersample dataset to use for post-training evaluation.

    Historical augmented CIFAR runs store ``all_examples_wo_data_aug``. That
    object contains the same raw 2n examples as ``all_examples`` but uses the
    deterministic test transform. Using ``all_examples`` for those runs would
    resample random crops/flips separately for every model evaluation and would
    therefore violate the fixed-supersample protocol.
    """
    deterministic_key = 'all_examples_wo_data_aug'
    if deterministic_key in saved_data:
        dataset = saved_data[deterministic_key]
        training_view = saved_data.get('all_examples')
        if training_view is None:
            raise KeyError("saved_data is missing required 'all_examples'")
        if len(dataset) != len(training_view):
            raise ValueError(
                'Deterministic evaluation supersample has a different length '
                'from the training-view supersample.'
            )

        train_indices = _dataset_source_indices(training_view)
        eval_indices = _dataset_source_indices(dataset)
        if (train_indices is not None and eval_indices is not None
                and not np.array_equal(train_indices, eval_indices)):
            raise ValueError(
                'Deterministic evaluation supersample does not contain the same '
                'raw examples/order as the training-view supersample.'
            )
        return dataset, deterministic_key

    if 'all_examples' not in saved_data:
        raise KeyError("saved_data is missing required 'all_examples'")

    training_aug = saved_data.get('training_data_augmentation')
    if training_aug is None:
        training_aug = _namespace_or_dict_value(saved_data.get('args'), 'data_augmentation', False)

    expected_key = saved_data.get('evaluation_supersample_key')
    if expected_key == deterministic_key or training_aug or require_deterministic:
        raise ValueError(
            'A deterministic fixed supersample is required for evaluation, but '
            "saved_data.pkl does not contain 'all_examples_wo_data_aug'. "
            'Restore the original saved_data.pkl produced by the augmented run; '
            'do not evaluate theorem quantities on random augmented views.'
        )

    return saved_data['all_examples'], 'all_examples'


# Backward-compatibility only: historical .mdl checkpoints saved the original
# callable in model.args["ld_schedule_fn"] with these module-qualified names.
# torch.load(weights_only=False) resolves those names while unpickling, even
# for StandardClassifier checkpoints which do not use an LD schedule. Do not
# remove or rename these symbols unless historical checkpoints are migrated.
# The parser does not use these schedules for any estimator or diagnostic.
def mnist_ld_schedule(lr, beta, iteration):
    if iteration % 100 == 0:
        lr = lr * 0.9
    beta = min(4000, max(100, 10 * np.exp(iteration / 100)))
    return lr, beta


def cifar_resnet50_ld_schedule(lr, beta, iteration):
    if iteration % 300 == 0:
        lr = lr * 0.9
    beta = min(16000, max(100, 10 * np.exp(iteration / 300)))
    return lr, beta


def register_legacy_main_pickle_globals():
    """Resolve __main__.* schedule references in trusted notebook checkpoints.

    Some archived .mdl files contain function globals recorded by pickle as
    __main__.mnist_ld_schedule (or __main__.cifar_resnet50_ld_schedule).
    Importing the functions from scripts.parse_results does not satisfy that
    lookup: unpickling resolves attributes on the module named in the file.
    Avoid overwriting a symbol that the notebook already defines.
    """
    import __main__

    for name in ('mnist_ld_schedule', 'cifar_resnet50_ld_schedule'):
        if not hasattr(__main__, name):
            setattr(__main__, name, globals()[name])


def load_methods_for_parsing():
    # load_methods_for_parsing() is also used by the Colab evaluation notebook
    # immediately before utils.load(). Register legacy notebook-level globals
    # before torch.load(weights_only=False) resolves their pickle references.
    register_legacy_main_pickle_globals()
    try:
        import methods
        return methods
    except ModuleNotFoundError as exc:
        if exc.name != 'sklearn':
            raise

    # Parsing only needs model construction/loading. The methods package imports
    # visualization helpers that require sklearn, but visualize() is not used here.
    visualizations = types.ModuleType('nnlib.nnlib.visualizations')

    def plot_predictions(*args, **kwargs):
        raise RuntimeError('visualization is unavailable while parsing without sklearn')

    visualizations.plot_predictions = plot_predictions
    sys.modules['nnlib.nnlib.visualizations'] = visualizations

    import methods
    return methods


def compute_risk(preds, mask, dataset):
    labels = [y for x, y in dataset]
    labels = torch.tensor(labels).long()
    indices0 = 2 * np.arange(len(mask)) + mask
    indices1 = 2 * np.arange(len(mask)) + 1 - mask

    LB = utils.to_numpy((preds.argmax(dim=1) != labels).to(torch.int))
    LC = utils.to_numpy(clipped_cross_entropy(preds, labels))
    return LB[indices0].mean(), LB[indices1].mean(), LC[indices0].mean(), LC[indices1].mean()


def compute_loss(preds, num, dataset):
    labels = [y for x, y in dataset]
    labels = torch.tensor(labels).long()
    indices0 = 2 * np.arange(num)
    indices1 = indices0 + 1

    LB = utils.to_numpy((preds.argmax(dim=1) != labels).to(torch.int))
    LC = utils.to_numpy(clipped_cross_entropy(preds, labels))
    return LB[indices0], LB[indices1], LC[indices0], LC[indices1]







def get_results_for_fixed_z(n, epoch, seed, args):
    methods = load_methods_for_parsing()

    LBts, LBvs = [], []
    LCts, LCvs = [], []
    LB0s, LB1s = [], []
    LC0s, LC1s = [], []
    masks = []

    evaluation_dataset_keys = []
    fixed_source_indices = None

    for S_seed in range(args.n_S_seeds):
        dir_name = f'n={n},seed={seed},S_seed={S_seed}'
        dir_path = os.path.join(args.results_dir, args.exp_name, dir_name)
        if not os.path.exists(dir_path):
            print(f"Did not find results for {dir_name}")
            continue

        with open(os.path.join(dir_path, 'saved_data.pkl'), 'rb') as f:
            saved_data = pickle.load(f)

        model = utils.load(
            path=os.path.join(dir_path, 'checkpoints', f'epoch{epoch - 1}.mdl'),
            methods=methods,
            device=args.device,
        )

        all_examples, evaluation_dataset_key = select_supersample_for_evaluation(
            saved_data,
            require_deterministic=getattr(
                args,
                'require_deterministic_supersample_eval',
                False,
            ),
        )
        evaluation_dataset_keys.append(evaluation_dataset_key)

        current_source_indices = get_supersample_identity(saved_data, all_examples)
        if current_source_indices is not None:
            if fixed_source_indices is None:
                fixed_source_indices = current_source_indices.copy()
            elif not np.array_equal(fixed_source_indices, current_source_indices):
                raise ValueError(
                    f'Selector run {dir_name} does not use the same ordered raw '
                    'supersample as the other selector runs for this seed.'
                )

        # Predictions on the fixed 2n supersample examples.
        preds = utils.apply_on_dataset(
            model=model,
            dataset=all_examples,
            batch_size=args.batch_size,
        )['pred']

        mask = saved_data['mask']
        masks.append(mask)

        LBt, LBv, LCt, LCv = compute_risk(
            preds=preds,
            mask=mask,
            dataset=all_examples,
        )

        LBts.append(LBt)
        LBvs.append(LBv)
        LCts.append(LCt)
        LCvs.append(LCv)

        LB0, LB1, LC0, LC1 = compute_loss(
            preds=preds,
            num=n,
            dataset=all_examples,
        )
        LB0s.append(LB0)
        LB1s.append(LB1)
        LC0s.append(LC0)
        LC1s.append(LC1)

    if len(masks) == 0:
        raise ValueError(
            f"No parsed runs found for n={n}, epoch={epoch}, seed={seed} "
            f"under results/{args.exp_name}"
        )

    if len(set(evaluation_dataset_keys)) != 1:
        raise ValueError(
            'Mixed supersample evaluation protocols were found across selector '
            f'runs: {sorted(set(evaluation_dataset_keys))}'
        )
    evaluation_dataset_key = evaluation_dataset_keys[0]

    LBt = np.mean(LBts)
    LBv = np.mean(LBvs)
    LCt = np.mean(LCts)
    LCv = np.mean(LCvs)

    LB0s = np.array(LB0s)
    LB1s = np.array(LB1s)
    LC0s = np.array(LC0s)
    LC1s = np.array(LC1s)
    masks = np.array(masks)


    bb_square, bb_fast, bb_bkl, bb_cmi, bb_cshi, bb_cjsi, bb_bjs = estimate_mi_bound_discrete(
        n,
        masks,
        LB0s,
        LB1s,
        LBt,
        LBv,
    )
    bc_square, bc_fast, bc_bkl, bc_cmi, bc_cshi, bc_cjsi, bc_bjs = estimate_mi_bound_continuous(
        n,
        masks,
        LC0s,
        LC1s,
        LCt,
        LCv,
    )

    result = {
        'loss_01': [LBt, LBv],
        'loss_ce': [LCt, LCv],
        'evaluation_protocol': {
            'supersample_dataset_key': evaluation_dataset_key,
            'deterministic_supersample_eval': True,
            'ce_clip_interval': [CE_CLIP_MIN, CE_CLIP_MAX],
            'signed_gap_definition': 'ghost_loss - train_loss',
            'ratio_denominator': 'abs(ghost_loss - train_loss)',
        },

        'bound_square': [bb_square, bc_square],
        'bound_fast': [bb_fast, bc_fast],
        'bound_bkl': [bb_bkl, bc_bkl],
        'bound_cmi': [bb_cmi, bc_cmi],
        'bound_cshi': [bb_cshi, bc_cshi],
        'bound_cjsi': [bb_cjsi, bc_cjsi],
        'bound_bjs': [bb_bjs, bc_bjs],
    }

    if getattr(args, 'include_theorem8_samplewise_loss_difference_valpha_bound', False):
        result['theorem8_samplewise_loss_difference_valpha_bound'] = (
            estimate_theorem8_samplewise_loss_difference_valpha_bound(
                n=n,
                masks=masks,
                L0_01=LB0s,
                L1_01=LB1s,
                L0_ce=LC0s,
                L1_ce=LC1s,
                train_loss_01=LBt,
                train_loss_ce=LCt,
                alphas=args.valpha_alphas,
                num_bins=getattr(args, 'theorem8_num_bins', 30),
                max_exact_values=getattr(args, 'theorem8_max_exact_values', 20),
                bounded_loss_M_01=getattr(args, 'theorem8_bounded_loss_M_01', 1.0),
                bounded_loss_M_ce=getattr(args, 'theorem8_bounded_loss_M_ce', 1.0),
            )
        )
    if getattr(args, 'include_theorem9_samplewise_single_loss_valpha_bound', False):
        result['theorem9_samplewise_single_loss_valpha_bound'] = (
            estimate_theorem9_samplewise_single_loss_valpha_bound(
                n=n,
                masks=masks,
                L0_01=LB0s,
                L0_ce=LC0s,
                L1_01=LB1s,
                L1_ce=LC1s,
                train_loss_01=LBt,
                train_loss_ce=LCt,
                alphas=args.valpha_alphas,
                num_bins=getattr(args, 'theorem9_num_bins', 30),
                max_exact_values=getattr(args, 'theorem9_max_exact_values', 20),
                bounded_loss_M_01=getattr(args, 'theorem9_bounded_loss_M_01', 1.0),
                bounded_loss_M_ce=getattr(args, 'theorem9_bounded_loss_M_ce', 1.0),
                center_01=getattr(args, 'theorem9_center_01', 0.5),
                center_ce=getattr(args, 'theorem9_center_ce', 0.5),
                use_weighted=not getattr(args, 'theorem9_use_bounded_form', False),
                symmetrization=getattr(args, 'theorem9_symmetrization', 'average'),
            )
        )
    if getattr(args, 'include_theorem10_samplewise_loss_pair_valpha_bound', False):
        result['theorem10_samplewise_loss_pair_valpha_bound'] = (
            estimate_theorem10_samplewise_loss_pair_valpha_bound(
                n=n,
                masks=masks,
                L0_01=LB0s,
                L1_01=LB1s,
                L0_ce=LC0s,
                L1_ce=LC1s,
                train_loss_01=LBt,
                train_loss_ce=LCt,
                alphas=args.valpha_alphas,
                num_bins=getattr(args, 'theorem10_num_bins', 30),
                max_exact_values=getattr(args, 'theorem10_max_exact_values', 20),
                bounded_loss_M_01=getattr(args, 'theorem10_bounded_loss_M_01', 1.0),
                bounded_loss_M_ce=getattr(args, 'theorem10_bounded_loss_M_ce', 1.0),
                use_weighted=not getattr(args, 'theorem10_use_bounded_form', False),
            )
        )

    if getattr(args, 'include_theorem11_samplewise_binarized_single_loss_valpha_bound', False):
        result['theorem11_samplewise_binarized_single_loss_valpha_bound'] = (
            estimate_theorem11_samplewise_binarized_single_loss_valpha_bound(
                n=n,
                masks=masks,
                L0_01=LB0s,
                L0_ce=LC0s,
                L1_01=LB1s,
                L1_ce=LC1s,
                train_loss_01=LBt,
                train_loss_ce=LCt,
                alphas=args.valpha_alphas,
                bounded_loss_M_01=getattr(args, 'theorem11_bounded_loss_M_01', 1.0),
                bounded_loss_M_ce=getattr(args, 'theorem11_bounded_loss_M_ce', 1.0),
                symmetrization=getattr(args, 'theorem11_symmetrization', 'average'),
            )
        )

    if getattr(args, 'include_theorem12_samplewise_binarized_loss_difference_valpha_bound', False):
        result['theorem12_samplewise_binarized_loss_difference_valpha_bound'] = (
            estimate_theorem12_samplewise_binarized_loss_difference_valpha_bound(
                n=n,
                masks=masks,
                L0_01=LB0s,
                L1_01=LB1s,
                L0_ce=LC0s,
                L1_ce=LC1s,
                train_loss_01=LBt,
                train_loss_ce=LCt,
                alphas=args.valpha_alphas,
                bounded_loss_M_01=getattr(args, 'theorem12_bounded_loss_M_01', 1.0),
                bounded_loss_M_ce=getattr(args, 'theorem12_bounded_loss_M_ce', 1.0),
                use_weighted=not getattr(args, 'theorem12_use_bounded_form', False),
            )
        )

    if getattr(args, 'include_theorem13_samplewise_binarized_loss_pair_valpha_bound', False):
        result['theorem13_samplewise_binarized_loss_pair_valpha_bound'] = (
            estimate_theorem13_samplewise_binarized_loss_pair_valpha_bound(
                n=n,
                masks=masks,
                L0_01=LB0s,
                L1_01=LB1s,
                L0_ce=LC0s,
                L1_ce=LC1s,
                train_loss_01=LBt,
                train_loss_ce=LCt,
                alphas=args.valpha_alphas,
                bounded_loss_M_01=getattr(args, 'theorem13_bounded_loss_M_01', 1.0),
                bounded_loss_M_ce=getattr(args, 'theorem13_bounded_loss_M_ce', 1.0),
                use_weighted=not getattr(args, 'theorem13_use_bounded_form', False),
            )
        )
    return result


def get_results(n, epoch, args):
    results = []
    for seed in range(args.n_seeds):
        results.append(get_results_for_fixed_z(n=n, epoch=epoch, seed=seed, args=args))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', default='cuda', help='specifies the main device')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument(
        '--require_deterministic_supersample_eval',
        action='store_true',
        help=(
            "require saved_data.pkl to contain the deterministic "
            "'all_examples_wo_data_aug' supersample; intended for augmented "
            "CIFAR-10 runs"
        ),
    )
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--ns', nargs='+', type=int, default=None,
                        help='optional override for training set sizes to parse')
    parser.add_argument('--n_seeds', type=int, default=None,
                        help='optional override for number of supersample seeds to parse')
    parser.add_argument('--n_S_seeds', type=int, default=None,
                        help='optional override for number of selector seeds to parse')
    parser.add_argument('--epochs', nargs='+', type=int, default=None,
                        help='optional override for checkpoint epochs to parse')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='optional override for number of classes')
    parser.add_argument('--valpha_alphas', nargs='+', type=float, default=[1.05, 1.2, 1.5, 2.0, 3.0],
                        help='alpha values for the coordinatewise Theorems 8--13 estimators')


    parser.add_argument(
        '--include_theorem8_samplewise_loss_difference_valpha_bound',
        action='store_true',
        help='include Theorem 8 sample-wise loss-difference V_alpha bound',
    )

    parser.add_argument(
        '--theorem8_num_bins',
        type=int,
        default=30,
        help='number of bins for Theorem 8 scalar loss-difference discretization',
    )

    parser.add_argument(
        '--theorem8_max_exact_values',
        type=int,
        default=20,
        help='use exact scalar support coding for Theorem 8 when the number of unique loss differences is at most this value',
    )

    parser.add_argument(
        '--theorem8_bounded_loss_M_01',
        type=float,
        default=1.0,
        help='bounded-loss constant M for Theorem 8 0-1 loss',
    )

    parser.add_argument(
        '--theorem8_bounded_loss_M_ce',
        type=float,
        default=1.0,
        help='bounded-loss constant M for Theorem 8 clamped cross-entropy loss',
    )
    parser.add_argument(
        '--include_theorem9_samplewise_single_loss_valpha_bound',
        action='store_true',
        help='include Theorem 9 sample-wise single-loss V_alpha bound',
    )

    parser.add_argument(
        '--theorem9_num_bins',
        type=int,
        default=30,
        help='number of bins for Theorem 9 scalar single-loss discretization',
    )

    parser.add_argument(
        '--theorem9_max_exact_values',
        type=int,
        default=20,
        help='use exact scalar support coding for Theorem 9 when the number of unique losses is at most this value',
    )

    parser.add_argument(
        '--theorem9_bounded_loss_M_01',
        type=float,
        default=1.0,
        help='bounded-loss constant M for Theorem 9 0-1 loss',
    )

    parser.add_argument(
        '--theorem9_bounded_loss_M_ce',
        type=float,
        default=1.0,
        help='bounded-loss constant M for Theorem 9 clamped cross-entropy loss',
    )

    parser.add_argument(
        '--theorem9_center_01',
        type=float,
        default=0.5,
        help='center c for Theorem 9 weighted 0-1 loss bound',
    )

    parser.add_argument(
        '--theorem9_center_ce',
        type=float,
        default=0.5,
        help='center c for Theorem 9 weighted CE loss bound',
    )

    parser.add_argument(
        '--theorem9_use_bounded_form',
        action='store_true',
        help='use the bounded M * V_alpha form for Theorem 9 instead of the sharper weighted form',
    )
    parser.add_argument(
        '--include_theorem10_samplewise_loss_pair_valpha_bound',
        action='store_true',
        help='include Theorem 10 sample-wise loss-pair V_alpha bound',
    )
    parser.add_argument('--theorem10_num_bins', type=int, default=30)
    parser.add_argument('--theorem10_max_exact_values', type=int, default=20)
    parser.add_argument('--theorem10_bounded_loss_M_01', type=float, default=1.0)
    parser.add_argument('--theorem10_bounded_loss_M_ce', type=float, default=1.0)
    parser.add_argument(
        '--theorem10_use_bounded_form',
        action='store_true',
        help='use bounded M*V_alpha form for Theorem 10 instead of weighted form',
    )

    parser.add_argument(
        '--include_theorem11_samplewise_binarized_single_loss_valpha_bound',
        action='store_true',
        help='include Theorem 11 sample-wise binarized single-loss V_alpha bound',
    )
    parser.add_argument('--theorem11_bounded_loss_M_01', type=float, default=1.0)
    parser.add_argument('--theorem11_bounded_loss_M_ce', type=float, default=1.0)

    parser.add_argument(
        '--include_theorem12_samplewise_binarized_loss_difference_valpha_bound',
        action='store_true',
        help='include Theorem 12 sample-wise binarized loss-difference V_alpha bound',
    )
    parser.add_argument('--theorem12_bounded_loss_M_01', type=float, default=1.0)
    parser.add_argument('--theorem12_bounded_loss_M_ce', type=float, default=1.0)
    parser.add_argument(
        '--theorem12_use_bounded_form',
        action='store_true',
        help='use bounded M*V_alpha form for Theorem 12 instead of weighted form',
    )

    parser.add_argument(
        '--include_theorem13_samplewise_binarized_loss_pair_valpha_bound',
        action='store_true',
        help='include Theorem 13 sample-wise binarized loss-pair V_alpha bound',
    )
    parser.add_argument('--theorem13_bounded_loss_M_01', type=float, default=1.0)
    parser.add_argument('--theorem13_bounded_loss_M_ce', type=float, default=1.0)
    parser.add_argument(
        '--theorem13_use_bounded_form',
        action='store_true',
        help='use bounded M*V_alpha form for Theorem 13 instead of weighted form',
    )
    parser.add_argument(
        '--theorem9_symmetrization',
        choices=['none', 'average', 'max'],
        default='average',
        help='conditional T9: average is theorem-aligned; max is looser; none is diagnostic only',
    )

    parser.add_argument(
        '--theorem11_symmetrization',
        choices=['none', 'average', 'max'],
        default='average',
        help='conditional T11: average is theorem-aligned; max is looser; none is diagnostic only',
    )
    args = parser.parse_args()
    print(args)

    override_fields = ['ns', 'n_seeds', 'n_S_seeds', 'epochs', 'num_classes']
    overrides = {field: getattr(args, field) for field in override_fields}
    required_overrides_supplied = all(value is not None for value in overrides.values())

    if args.exp_name in {"mnist-4vs9-CNN", "mnist-4vs9-CNN-valpha-main"} or args.exp_name.startswith("mnist-4vs9-CNN-noise"):
        defaults = {
            'n_seeds': 5,
            'n_S_seeds': 30,
            'ns': [75, 250, 1000, 4000],
            'epochs': [200],
            'num_classes': 2,
        }
    elif args.exp_name == 'mnist-4vs9-CNN-LD':
        defaults = {
            'n_seeds': 5,
            'n_S_seeds': 30,
            'ns': [4000],
            'epochs': list(np.arange(1, 11) * 4),
            'num_classes': 2,
            'batch_size': 100,
        }
    elif args.exp_name in {'cifar10-pretrained-resnet50', 'cifar10-pretrained-resnet50-valpha-main'}:
        defaults = {
            'n_seeds': 2,
            'n_S_seeds': 40,
            'ns': [1000, 5000, 20000],
            'epochs': [40],
            'num_classes': 10,
        }
    elif args.exp_name == 'cifar10-pretrained-resnet50-LD':
        defaults = {
            'n_seeds': 1,
            'n_S_seeds': 1,
            'ns': [20000],
            'epochs': list(np.arange(1, 9) * 2),
            'num_classes': 10,
            'batch_size': 64,
        }
    else:
        if not required_overrides_supplied:
            missing = [field for field, value in overrides.items() if value is None]
            raise ValueError(
                f"Unexpected exp_name: {args.exp_name}. Custom experiment names require "
                f"all parse overrides: --ns, --n_seeds, --n_S_seeds, --epochs, --num_classes. "
                f"Missing: {missing}"
            )
        defaults = {}

    for field, default_value in defaults.items():
        if getattr(args, field, None) is None:
            setattr(args, field, default_value)

    args.epochs = list(args.epochs)

    # Parse baseline information bounds and enabled T8--T13 estimators.
    results = NestedDict()  # indexing with n, epoch
    for n in tqdm(args.ns):
        results[n][args.epochs[-1]] = get_results(n=n, epoch=args.epochs[-1], args=args)
    for epoch in tqdm(args.epochs[:-1]):
        results[args.ns[-1]][epoch] = get_results(n=args.ns[-1], epoch=epoch, args=args)
    results_file_path = os.path.join(args.results_dir, args.exp_name, 'results.pkl')
    with open(results_file_path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
