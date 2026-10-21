"""Analytical canonical-channel comparisons and alpha trade-off; MI is in nats."""
from pathlib import Path
import argparse
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_figure_style import save_paper_figure


def binary_entropy(p):
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = -p * np.log(p) - (1-p) * np.log(1-p)
    return np.where((p == 0) | (p == 1), 0.0, value)


def z_mutual_information(p, q):
    return binary_entropy((1-p)*q) - q*binary_entropy(p)


def v_alpha(alpha, p, q):
    w0 = ((1-q)*np.abs(q-p*q)**alpha
          + q*np.abs(p+q-1-p*q)**alpha)**(1/alpha)
    return w0+w0  # Binary output deviations are equal in magnitude.


def loss_norm(alpha, w, q):
    conjugate = 1/(1-1/alpha)
    population_loss = (1-q)*w**2 + q*(w-1)**2
    return (q*np.abs(population_loss-(w-1)**2)**conjugate
            + (1-q)*np.abs(population_loss-w**2)**conjugate)**(1/conjugate)


def channel_values():
    p = np.linspace(0, 1, 1001)
    alpha = np.linspace(1.01, 20, 1000)
    q = .8
    sensitivity = np.maximum(loss_norm(alpha, 0, q), loss_norm(alpha, 1, q))
    dependence = v_alpha(alpha, .3, q)
    return {
        "p": p,
        "bsc_v2": np.abs(1-2*p),
        "bsc_mi": np.sqrt(2*np.maximum(np.log(2)-binary_entropy(p),0)),
        "bec_v2": 1-p,
        "bec_mi": np.sqrt(2*np.log(2)*(1-p)),
        "z_v2": v_alpha(2, p, .5)*np.maximum(loss_norm(2, 0, .5), loss_norm(2, 1, .5)),
        "z_mi": np.sqrt(2*.25*z_mutual_information(p, .5)),
        "alpha": alpha,
        "sensitivity": sensitivity,
        "dependence": dependence,
        "product": sensitivity*dependence,
        "gap": np.full_like(alpha, 2*q*(1-.3)*(1-q)),
        "mi_tradeoff": np.full_like(alpha, np.sqrt(2*.25*z_mutual_information(.3, q))),
    }


def draw_pair(ax, x, bound, mi, xlabel, ymax, *, z_channel=False):
    # BSC/BEC already omit sqrt(sigma^2/n). Remove the same factor
    # from the Z-channel's full bounds: sqrt((1/4)/1) = 1/2.
    if z_channel:
        normalization_factor = np.sqrt(.25 / 1.0)
        bound = np.asarray(bound) / normalization_factor
        mi = np.asarray(mi) / normalization_factor
    bound_label = r"$V_2(S;W)$"
    mi_label = r"$\sqrt{2I(S;W)}$"
    ax.plot(x, bound, "b-", linewidth=2, label=bound_label)
    ax.plot(x, mi, "r--", linewidth=2, label=mi_label)
    ax.set(xlim=(0,1), ylim=(0,ymax), xlabel=xlabel, ylabel="Normalized bound")
    ax.set_xticks(np.linspace(0,1,6))
    ax.legend(loc="best", fontsize=8)


def make_channel_triptych(data):
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.4))
    specs = [
        ("(a) BSC", "bsc_v2", "bsc_mi", r"$p$", 1.5, False),
        ("(b) BEC", "bec_v2", "bec_mi", r"$\epsilon$", 1.5, False),
        ("(c) Z-channel", "z_v2", "z_mi", r"$p$", 1.5, True),
    ]
    for ax, (title, ours, mi, xlabel, ymax, is_z) in zip(axes, specs):
        draw_pair(ax, data["p"], data[ours], data[mi], xlabel, ymax, z_channel=is_z)
        ax.set_title(title, fontsize=12)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent/"paper_figures")
    args = parser.parse_args()
    d = channel_values()
    specs = [(6, "Canonical-channels-1x3", make_channel_triptych(d))]
    fig,ax=plt.subplots(figsize=(9.2,5.0), constrained_layout=True)
    ax.plot(d["alpha"],d["sensitivity"],color=(.6,.1,.15),linestyle=":",linewidth=2,
            label=r"$\sup_{w\in\mathcal{W}}\|L_S(w)-L_\mu(w)\|_{\alpha^\prime}$")
    ax.plot(d["alpha"],d["dependence"],"m--",linewidth=2,label=r"$V_\alpha(S;W)$")
    ax.plot(d["alpha"],d["product"],"b-.",linewidth=2,
            label=r"$\sup_w\|L_S(w)-L_\mu(w)\|_{\alpha^\prime}\cdot V_\alpha(S;W)$")
    ax.plot(d["alpha"],d["gap"],"k-",linewidth=2,label=r"$\overline{\mathrm{gen}}$")
    ax.plot(d["alpha"],d["mi_tradeoff"],"ro-",linewidth=2,markersize=2,
            label=r"$\sqrt{\frac{2\sigma^2}{n}I(W;S)}$")
    ax.set(xlim=(0,20),ylim=(.2,1.1),xlabel=r"$\alpha$",ylabel="Function values")
    ax.set_xticks(np.arange(0,21,2))
    ax.legend(loc="best")
    specs.append((7,"Trade-off",fig))
    reports=[]
    for number,name,figure in specs:
        reports.append(save_paper_figure(figure,args.output_dir/name,number))
        plt.close(figure)
    args.output_dir.mkdir(exist_ok=True,parents=True)
    np.savez(args.output_dir/"channel_curves.npz",**d)
    print(json.dumps(reports,indent=2))

if __name__ == "__main__":
    main()
