import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from modules.bound_utils import estimate_mi_bound_discrete, estimate_mi_bound_continuous
from nnlib.nnlib import utils


def SuperSample(num=2, num_class=2, d=2, sep=1.0):
    samples = make_classification(n_samples=2 * num, n_features=d, n_redundant=0, n_classes=num_class, n_informative=d, n_clusters_per_class=1, class_sep=sep, flip_y=0)

    for i in range(int(num_class)):
        cluster = samples[0][samples[1] == i]
        targets = np.zeros(len(cluster)) + i
        if i == 0:
            inputs = cluster
            labels = targets
        else:
            labels = np.append(labels, targets)
            inputs = np.concatenate((inputs, cluster), axis=0)

    X_0, X_1, y_0, y_1 = train_test_split(
        inputs, labels, test_size=0.5, random_state=42)
    return X_0, X_1, y_0, y_1


def compute_loss(model, X, y):
    preds = model(X)
    LB = utils.to_numpy((preds.argmax(dim=1) != y).to(torch.int))
    LC = utils.to_numpy(F.cross_entropy(preds, y, reduction='none').clamp(0, 1))
    return LB, LC


def TrainClassification(X_0, X_1, y_0, y_1, U, d, num_class):
    class MultiClassification(torch.nn.Module):
        def __init__(self, input_dim, output_dim):
            super(MultiClassification, self).__init__()
            self.linear = torch.nn.Linear(input_dim, output_dim)

        def forward(self, x):
            x = torch.flatten(x, 1)
            x = self.linear(x)
            return x

    criterion = torch.nn.CrossEntropyLoss()
    epochs = 300
    input_dim = d
    output_dim = num_class
    learning_rate = 0.01
    model = MultiClassification(input_dim, output_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    X_train = (1 - U).reshape(-1, 1) * X_0 + U.reshape(-1, 1) * X_1
    y_train = (1 - U) * y_0 + U * y_1
    X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train).type(torch.LongTensor)

    X_val = U.reshape(-1, 1) * X_0 + (1 - U).reshape(-1, 1) * X_1
    y_val = U * y_0 + (1 - U) * y_1
    X_val, y_val = torch.Tensor(X_val), torch.Tensor(y_val).type(torch.LongTensor)

    X_0, y_0, X_1, y_1 = torch.Tensor(X_0), torch.Tensor(y_0).type(torch.LongTensor), torch.Tensor(X_1), torch.Tensor(y_1).type(torch.LongTensor)

    for epoch in range(int(epochs)):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total = y_train.size(0)
            correct = np.sum(outputs.data.max(1)[1].detach().view_as(y_train).numpy() == y_train.detach().numpy())
            accuracy = 100 * correct / total
            if accuracy >= 99:
                break

    with torch.no_grad():
        LB0, LC0 = compute_loss(model, X_0, y_0)
        LB1, LC1 = compute_loss(model, X_1, y_1)
        LBt, LCt = compute_loss(model, X_train, y_train)
        LBv, LCv = compute_loss(model, X_val, y_val)

    return LB0, LC0, LB1, LC1, LBt, LCt, LBv, LCv


def RunClassExp(n=2, n_class=2, dim=2, diff=1.0, num_Z=50, num_U=100):
    results = []

    for i in tqdm(range(int(num_Z))):
        X_0, X_1, y_0, y_1 = SuperSample(num=n, num_class=n_class, d=dim, sep=diff)

        LBts, LBvs = [], []
        LCts, LCvs = [], []
        LB0s, LB1s = [], []
        LC0s, LC1s = [], []
        masks = []

        for j in tqdm(range(int(num_U))):
            U = np.random.binomial(size=n, n=1, p=0.5)
            LB0, LC0, LB1, LC1, LBt, LCt, LBv, LCv = TrainClassification(X_0, X_1, y_0, y_1, U, dim, n_class)

            LBts.append(LBt)
            LBvs.append(LBv)
            LCts.append(LCt)
            LCvs.append(LCv)
            LB0s.append(LB0)
            LB1s.append(LB1)
            LC0s.append(LC0)
            LC1s.append(LC1)
            masks.append(U)

        LBt = np.mean(LBts)
        LBv = np.mean(LBvs)
        LCt = np.mean(LCts)
        LCv = np.mean(LCvs)

        LB0s = np.array(LB0s)
        LB1s = np.array(LB1s)
        LC0s = np.array(LC0s)
        LC1s = np.array(LC1s)
        masks = np.array(masks)

        bb_square, bb_fast, bb_bkl, bb_cmi, bb_cshi, bb_cjsi, bb_bjs = estimate_mi_bound_discrete(n, masks, LB0s, LB1s, LBt, LBv)
        bc_square, bc_fast, bc_bkl, bc_cmi, bc_cshi, bc_cjsi, bc_bjs = estimate_mi_bound_continuous(n, masks, LC0s, LC1s, LCt, LCv)

        results.append({
            'loss_01': [LBt, LBv],
            'loss_ce': [LCt, LCv],

            'bound_square': [bb_square, bc_square],
            'bound_fast': [bb_fast, bc_fast],
            'bound_bkl': [bb_bkl, bc_bkl],
            'bound_cmi': [bb_cmi, bc_cmi],
            'bound_cshi': [bb_cshi, bc_cshi],
            'bound_cjsi': [bb_cjsi, bc_cjsi],
            'bound_bjs': [bb_bjs, bc_bjs],
        })

    return results


def PlotBound(n_class, dim, diff, num_Z, num_U, sample, exp_name, results_dir):
    print("------Class:", n_class, " Difficulty:", 1 / diff, "------")

    for num in sample:
        results = RunClassExp(n=num, n_class=n_class, dim=dim, diff=diff, num_Z=num_Z, num_U=num_U)
        results_file_path = os.path.join(results_dir, exp_name, f'results_{num}.pkl')
        os.makedirs(os.path.join(results_dir, exp_name), exist_ok=True)
        with open(results_file_path, 'wb') as f:
            pickle.dump(results, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.set_defaults(parse=True)
    args = parser.parse_args()
    print(args)

    sample = [10, 20, 30, 50, 80, 100, 200]
    if args.exp_name in ["two"]:
        PlotBound(n_class=2, dim=5, diff=1, num_Z=50, num_U=100, sample=sample, exp_name=args.exp_name, results_dir=args.results_dir)
    elif args.exp_name in ["ten"]:
        PlotBound(n_class=10, dim=5, diff=4, num_Z=50, num_U=100, sample=sample, exp_name=args.exp_name, results_dir=args.results_dir)
    else:
        raise ValueError(f"Unexpected exp_name: {args.exp_name}")


if __name__ == '__main__':
    main()
