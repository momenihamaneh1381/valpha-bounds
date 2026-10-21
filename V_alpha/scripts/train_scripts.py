import argparse
import os
import subprocess
import time

import torch

from nnlib.nnlib.data_utils.base import register_parser as register_fn

local_functions = {}  # storage for registering the functions below


def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    try:
        # Get list of GPUs from env, split by ',' and remove empty string ''
        # To handle the case when there is one extra comma: `CUDA_VISIBLE_DEVICES=0,1,2,3, python3 ...`
        available_gpus = [x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x != '']
    except Exception:
        # If the env variable is not set, we use all GPUs
        available_gpus = [str(x) for x in range(torch.cuda.device_count())]
    n_gpus = len(available_gpus)
    procs_by_gpu = [None] * n_gpus

    try:
        while True:
            for idx, gpu_idx in enumerate(available_gpus):
                proc = procs_by_gpu[idx]
                if (proc is None) or (proc.poll() is not None):
                    # Nothing is running on this GPU; launch a command.
                    cmd = next(commands)
                    print(cmd)
                    new_proc = subprocess.Popen(
                        f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                    procs_by_gpu[idx] = new_proc
                    break
            time.sleep(0.1)
    except StopIteration:
        pass

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()


def iter_seed(command_prefix, ns, n_seeds, n_S_seeds):
    for n in ns:
        for seed in range(n_seeds):
            for S_seed in range(n_S_seeds):
                yield command_prefix + f"--n {n} --seed {seed} --S_seed {S_seed};"


#######################################################################################
#
#     MNIST 4 vs 9, CNN
#
# exp_name: mnist-4vs9-CNN
#######################################################################################

@register_fn(local_functions, 'mnist-4vs9-CNN')
def foo(**kwargs):
    config_file = 'configs/binary-mnist-4layer-CNN.json'
    batch_size = 128
    n_epochs = 200
    save_iter = 20
    exp_name = "mnist-4vs9-CNN"
    dataset = 'mnist'
    which_labels = '4 9'

    command_prefix = f"python -um scripts.train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"--which_labels {which_labels} --deterministic "

    n_seeds = 5
    n_S_seeds = 30
    ns = [75, 250, 1000, 4000]

    return iter_seed(command_prefix, ns, n_seeds, n_S_seeds)


@register_fn(local_functions, 'mnist-4vs9-CNN-noise')
def foo(error_prob=0, **kwargs):
    config_file = 'configs/binary-mnist-4layer-CNN.json'
    batch_size = 128
    n_epochs = 200
    save_iter = 20
    exp_name = f"mnist-4vs9-CNN-noise-{error_prob}"
    dataset = 'mnist'
    which_labels = '4 9'

    command_prefix = f"python -um scripts.train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"--which_labels {which_labels} --deterministic --error_prob {error_prob} "

    n_seeds = 5
    n_S_seeds = 30
    ns = [75, 250, 1000, 4000]

    return iter_seed(command_prefix, ns, n_seeds, n_S_seeds)


@register_fn(local_functions, 'mnist-4vs9-CNN-LD')
def foo(**kwargs):
    config_file = 'configs/binary-mnist-4layer-CNN.json'
    batch_size = 100
    n_epochs = 40
    save_iter = 4
    exp_name = "mnist-4vs9-CNN-LD"
    dataset = 'mnist'
    which_labels = '4 9'
    ld_lr = 0.004
    ld_beta = 10.0

    command_prefix = f"python -um scripts.train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"--which_labels {which_labels} -m LangevinDynamics --ld_lr {ld_lr} " \
                     f"--ld_beta {ld_beta} "

    n_seeds = 5
    n_S_seeds = 30
    ns = [4000]

    return iter_seed(command_prefix, ns, n_seeds, n_S_seeds)


@register_fn(local_functions, 'cifar10-pretrained-resnet50')
def foo(**kwargs):
    config_file = 'configs/pretrained-resnet50-cifar10.json'
    batch_size = 64
    n_epochs = 40
    save_iter = n_epochs
    exp_name = "cifar10-pretrained-resnet50"
    dataset = 'cifar10'
    optimizer = 'sgd'
    lr = 0.01
    momentum = 0.9

    command_prefix = f"python -um scripts.train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"-m StandardClassifier -A --resize_to_imagenet --optimizer {optimizer} " \
                     f"--lr {lr} --momentum {momentum} "

    n_seeds = 2
    n_S_seeds = 40
    ns = [1000, 5000, 20000]

    return iter_seed(command_prefix, ns, n_seeds, n_S_seeds)


@register_fn(local_functions, 'cifar10-pretrained-resnet50-LD')
def foo(**kwargs):
    config_file = 'configs/pretrained-resnet50-cifar10.json'
    batch_size = 64
    n_epochs = 16
    save_iter = 2
    exp_name = "cifar10-pretrained-resnet50-LD"
    dataset = 'cifar10'
    ld_lr = 0.01
    ld_beta = 10.0

    command_prefix = f"python -um scripts.train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"-m LangevinDynamics -A --resize_to_imagenet  " \
                     f"--ld_lr {ld_lr} --ld_beta {ld_beta} "

    n_seeds = 1
    n_S_seeds = 40
    ns = [20000]

    return iter_seed(command_prefix, ns, n_seeds, n_S_seeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_names', '-E', type=str, nargs='+', required=True)
    parser.add_argument('--error_prob', type=float, default=0.0)
    args = parser.parse_args()

    for exp_name in args.exp_names:
        assert exp_name in local_functions
        commands = local_functions[exp_name](**vars(args))
        multi_gpu_launcher(commands)


if __name__ == '__main__':
    main()
