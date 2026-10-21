import argparse
import json
import os
import pickle
import time

import numpy as np
from torch.utils.data import Subset

from nnlib.nnlib import training, metrics, callbacks, utils
from nnlib.nnlib.data_utils.wrappers import SubsetDataWrapper, LabelSubsetWrapper, ResizeImagesWrapper, UniformNoiseWrapper, LabelMappingWrapper
from nnlib.nnlib.data_utils.base import get_loaders_from_datasets, get_input_shape
from modules.data_utils import load_data_from_arguments
import methods


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


def _resolve_raw_dataset_indices(dataset):
    """Map wrapper-relative indices back to the underlying raw dataset.

    The returned indices are suitable for auditing a supersample independently
    of stochastic transforms. Unknown wrapper types that expose ``dataset`` are
    treated as index-preserving.
    """
    positions = np.arange(len(dataset), dtype=np.int64)
    current = dataset

    while hasattr(current, 'dataset'):
        if isinstance(current, SubsetDataWrapper):
            positions = np.asarray(current.include_indices, dtype=np.int64)[positions]
        elif isinstance(current, LabelSubsetWrapper):
            positions = np.asarray(current.valid_indices, dtype=np.int64)[positions]
        elif isinstance(current, Subset):
            positions = np.asarray(current.indices, dtype=np.int64)[positions]
        current = current.dataset

    return positions, current


def _cifar10_supersample_metadata(dataset, args):
    """Portable audit metadata for the fixed CIFAR-10 supersample."""
    raw_indices, raw_dataset = _resolve_raw_dataset_indices(dataset)
    targets = getattr(raw_dataset, 'targets', None)
    if targets is None:
        raise RuntimeError('Could not recover CIFAR-10 raw labels for supersample metadata.')
    raw_labels = np.asarray(targets, dtype=np.int64)[raw_indices]

    return {
        'dataset': 'CIFAR10',
        'source_split': 'train',
        'raw_train_indices': raw_indices,
        'raw_labels': raw_labels,
        'train_validation_split_seed': int(args.seed),
        'train_validation_ratio': 0.2,
        'supersample_selection_seed': int(args.seed),
        'deterministic_evaluation_preprocessing': {
            'augmentation': False,
            'to_tensor': True,
            'normalize_mean': [0.485, 0.456, 0.406],
            'normalize_std': [0.229, 0.224, 0.225],
            'resize': [224, 224] if args.resize_to_imagenet else None,
            'resize_mode': 'bilinear' if args.resize_to_imagenet else None,
            'align_corners': False if args.resize_to_imagenet else None,
        },
        'training_augmentation': {
            'random_horizontal_flip': bool(args.data_augmentation),
            'random_crop_size': 32 if args.data_augmentation else None,
            'random_crop_padding': 4 if args.data_augmentation else None,
        },
    }


def load_data(args):
    all_examples, _, _, _ = load_data_from_arguments(args, build_loaders=False)

    # select labels if needed
    if args.which_labels is not None:
        all_examples = LabelSubsetWrapper(all_examples, which_labels=args.which_labels)

        if args.error_prob > 0:
            all_examples = UniformNoiseWrapper(all_examples, error_prob=args.error_prob, num_classes=2, seed=args.seed)

    # resize if needed
    if args.resize_to_imagenet:
        all_examples = ResizeImagesWrapper(all_examples, size=(224, 224))

    # select 2n examples (tilde{z})
    assert len(all_examples) >= 2 * args.n
    np.random.seed(args.seed)
    include_indices = np.random.choice(range(len(all_examples)), size=2 * args.n, replace=False)
    all_examples = SubsetDataWrapper(all_examples, include_indices=include_indices)

    return all_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--device', '-d', default='cuda', help='specifies the main device')
    parser.add_argument('--all_device_ids', nargs='+', type=str, default=None,
                        help="If not None, this list specifies devices for multiple GPU training. "
                             "The first device should match with the main device (args.device).")
    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--epochs', '-e', type=int, default=400)
    parser.add_argument('--stopping_param', type=int, default=2**30)
    parser.add_argument('--save_iter', '-s', type=int, default=10)
    parser.add_argument('--vis_iter', '-v', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--S_seed', type=int, default=42)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--deterministic', action='store_true', dest='deterministic')

    # data parameters
    parser.add_argument('--dataset', '-D', type=str, default='corrupt4_mnist')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--data_augmentation', '-A', action='store_true', dest='data_augmentation')
    parser.set_defaults(data_augmentation=False)
    parser.add_argument('--error_prob', type=float, default=0.0)
    parser.add_argument('--n', '-n', type=int, required=True, default='Number of training examples')
    parser.add_argument('--which_labels', nargs='+', default=None, type=int)
    parser.add_argument('--clean_validation', action='store_true', default=False)
    parser.add_argument('--resize_to_imagenet', action='store_true', default=False)

    # hyper-parameters
    parser.add_argument('--model_class', '-m', type=str, default='StandardClassifier')
    parser.add_argument('--load_from', type=str, default=None)

    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.0, help='momentum')

    parser.add_argument('--ld_lr', type=float, help='initial learning rate of Langevin dynamics')
    parser.add_argument('--ld_beta', type=float, help='initial inverse temperature of LD')
    parser.add_argument('--ld_track_grad_variance', dest='ld_track_grad_variance', action='store_true')
    parser.add_argument('--ld_track_every_iter', type=int, default=1)
    args = parser.parse_args()
    print(args)

    log_dir = f'results/{args.exp_name}/n={args.n},seed={args.seed},S_seed={args.S_seed}'
    save_path = f'{log_dir}/saved_data.pkl'
    if os.path.exists(save_path):
        return

    # Load data
    all_examples = load_data(args)

    # select the train/val split (S)
    np.random.seed(args.S_seed)
    mask = np.random.randint(2, size=(args.n,))
    train_indices = 2*np.arange(args.n) + mask
    val_indices = 2*np.arange(args.n) + (1-mask)
    train_data = SubsetDataWrapper(all_examples, include_indices=train_indices)
    val_data = SubsetDataWrapper(all_examples, include_indices=val_indices)

    if args.deterministic:
        num_workers = 0  # to make sure data shuffling is always done the same way
    else:
        num_workers = 2 # I changed this from 4 due to errors

    train_loader, val_loader, test_loader = get_loaders_from_datasets(train_data=train_data,
                                                                      val_data=val_data,
                                                                      test_data=None,
                                                                      batch_size=args.batch_size,
                                                                      num_workers=num_workers)

    # Options
    optimization_args = {
        'optimizer': {
            'name': args.optimizer,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum
        }
    }

    with open(args.config, 'r') as f:
        architecture_args = json.load(f)

    model_class = getattr(methods, args.model_class)

    if args.deterministic:
        utils.set_seed(42)

    ld_schedule_fn = mnist_ld_schedule
    if args.exp_name == 'cifar10-pretrained-resnet50-LD':
        ld_schedule_fn = cifar_resnet50_ld_schedule

    model = model_class(input_shape=get_input_shape(train_loader.dataset),
                        architecture_args=architecture_args,
                        device=args.device,
                        load_from=args.load_from,
                        ld_lr=args.ld_lr,
                        ld_beta=args.ld_beta,
                        ld_schedule_fn=ld_schedule_fn,
                        ld_track_grad_variance=args.ld_track_grad_variance,
                        ld_track_every_iter=args.ld_track_every_iter)

    metrics_list = [metrics.Accuracy(output_key='pred')]
    if args.dataset == 'imagenet':
        metrics_list.append(metrics.TopKAccuracy(k=5, output_key='pred'))

    callbacks_list = [callbacks.SaveBestWithMetric(metric=metrics_list[0], partition='val', direction='max')]

    training_started = time.perf_counter()
    training.train(model=model,
                   train_loader=train_loader,
                   val_loader=val_loader,
                   epochs=args.epochs,
                   save_iter=args.save_iter,
                   vis_iter=args.vis_iter,
                   optimization_args=optimization_args,
                   log_dir=log_dir,
                   args_to_log=args,
                   metrics=metrics_list,
                   callbacks=callbacks_list,
                   device_ids=args.all_device_ids)
    training_wall_time_seconds = time.perf_counter() - training_started

    training_data_augmentation = bool(args.data_augmentation)
    save_data = {
        'args': args,
        'mask': mask,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'all_examples': all_examples,
        'metrics': metrics_list,
        # Explicit metadata avoids relying on mutable argparse state when the
        # deterministic evaluation copy is constructed below.
        'training_data_augmentation': training_data_augmentation,
        'training_wall_time_seconds': float(training_wall_time_seconds),
        'supersample_source_indices': np.asarray(all_examples.include_indices),
    }

    # Store portable source-level metadata in addition to the historical
    # pickled dataset object. This makes the exact CIFAR-10 supersample and its
    # deterministic evaluation preprocessing auditable without invoking random
    # transforms.
    if args.dataset == 'cifar10':
        save_data['supersample_metadata'] = _cifar10_supersample_metadata(
            all_examples,
            args,
        )

    # If training used stochastic augmentation, persist a second view of the
    # *same raw supersample* with augmentation disabled. The theorem estimators
    # must evaluate every trained model on one fixed supersample rather than on
    # newly sampled random crops/flips. Do not mutate ``args``: old versions did
    # so, which made saved_data['args'].data_augmentation incorrectly appear
    # false even when training used augmentation.
    if training_data_augmentation:
        eval_args = argparse.Namespace(**vars(args))
        eval_args.data_augmentation = False
        all_examples_wo_data_aug = load_data(eval_args)

        train_raw_indices, _ = _resolve_raw_dataset_indices(all_examples)
        eval_raw_indices, _ = _resolve_raw_dataset_indices(all_examples_wo_data_aug)
        if not np.array_equal(train_raw_indices, eval_raw_indices):
            raise RuntimeError(
                'Deterministic evaluation supersample does not match the raw '
                'training supersample indices/order.'
            )

        save_data['all_examples_wo_data_aug'] = all_examples_wo_data_aug
        save_data['evaluation_supersample_key'] = 'all_examples_wo_data_aug'
    else:
        save_data['evaluation_supersample_key'] = 'all_examples'

    with open(save_path, 'wb') as f:
        pickle.dump(obj=save_data, file=f)


if __name__ == '__main__':
    main()
