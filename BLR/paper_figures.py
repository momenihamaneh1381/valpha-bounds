"""Redraw manuscript Figures 4--5 from the cached BLR summary."""
from pathlib import Path
import argparse
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_figure_style import save_paper_figure

BLR_FIGSIZE = (9.2, 5.0)

def plot_curve_with_band(
    ax,
    x,
    y,
    yerr,
    *,
    fmt,
    label,
    linewidth=2,
    capsize=3,
    alpha_band=0.14,
):
    """Plot a discrete curve with error bars and a light uncertainty band."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    line = ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt=fmt,
        linewidth=linewidth,
        capsize=capsize,
        label=label,
    )
    color = line.lines[0].get_color()
    ax.fill_between(
        x,
        np.maximum(y - yerr, 0.0),
        y + yerr,
        color=color,
        alpha=alpha_band,
        linewidth=0,
    )
    return line

def plot_conditional_output_paper_figures(summary_frame):
    fig, ax = plt.subplots(figsize=BLR_FIGSIZE, constrained_layout=True)
    dims = summary_frame["dimension"].to_numpy(dtype=float)

    plot_curve_with_band(
        ax,
        dims,
        summary_frame["empirical_generalization"],
        summary_frame["empirical_generalization_stderr"],
        fmt="o-",
        label="Empirical generalization",
    )
    plot_curve_with_band(
        ax,
        dims,
        summary_frame["mi_bound"],
        summary_frame["mi_bound_stderr"],
        fmt="h--",
        label="MI",
    )
    plot_curve_with_band(
        ax,
        dims,
        summary_frame["cmi_bound"],
        summary_frame["cmi_bound_stderr"],
        fmt="s--",
        label="CMI",
    )
    plot_curve_with_band(
        ax,
        dims,
        summary_frame["rigorous_conditional_v_alpha_bound"],
        summary_frame["rigorous_conditional_v_alpha_bound_stderr"],
        fmt="d-",
        label=r"Conditional theorem-valid $V_\alpha$ (Thm. 5)",
    )
    plot_curve_with_band(
        ax,
        dims,
        summary_frame["proxy_conditional_v_alpha_bound"],
        summary_frame["proxy_conditional_v_alpha_bound_stderr"],
        fmt="x-",
        label=r"Conditional finite-candidate approx. $V_\alpha$ (Thm. 5)",
    )
    plot_curve_with_band(
        ax,
        dims,
        summary_frame["corollary1_parameter_v_alpha_bound"],
        summary_frame["corollary1_parameter_v_alpha_bound_stderr"],
        fmt="*:",
        label=r"Parameter Cor. 2 $V_\alpha$ (Cor. 2)",
    )
    plot_curve_with_band(
        ax,
        dims,
        summary_frame["proxy_theorem1_v_alpha_bound"],
        summary_frame["proxy_theorem1_v_alpha_bound_stderr"],
        fmt="P-",
        label=r"Parameter finite-candidate approx. $V_\alpha$ (Thm. 1)",
    )
    plot_curve_with_band(
        ax,
        dims,
        summary_frame["fixed_parameter_v2_bound"],
        summary_frame["fixed_parameter_v2_bound_stderr"],
        fmt="v--",
        label=r"Parameter fixed $V_2$ ($\alpha=2$) (Thm. 2)",
    )

    ax.set_xlabel("Dimension d")
    ax.set_ylabel("Bound / empirical gap")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    fig_alpha, ax_alpha = plt.subplots(figsize=BLR_FIGSIZE, constrained_layout=True)
    ax_alpha.plot(
        dims,
        summary_frame["rigorous_conditional_selected_alpha"],
        "d-",
        linewidth=2,
        label=r"Conditional theorem-valid $\alpha^*$",
    )
    ax_alpha.plot(
        dims,
        summary_frame["proxy_conditional_selected_alpha"],
        "x-",
        linewidth=2,
        label=r"Conditional finite-candidate $\alpha^*$",
    )
    ax_alpha.plot(
        dims,
        summary_frame["corollary1_parameter_selected_alpha"],
        "*:",
        linewidth=2,
        label=r"Parameter Cor. 2 $\alpha^*$",
    )
    ax_alpha.plot(
        dims,
        summary_frame["proxy_theorem1_selected_alpha"],
        "P-",
        linewidth=2,
        label=r"Parameter finite-candidate $\alpha^*$",
    )
    ax_alpha.axhline(2.0, color="0.45", linestyle=":", linewidth=1.5, label=r"Reference $\alpha=2$")
    ax_alpha.set_xlabel("Dimension d")
    ax_alpha.set_ylabel(r"Selected $\alpha^*$")
    ax_alpha.grid(True, alpha=0.3)
    ax_alpha.legend(fontsize=8)

    return fig, fig_alpha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--input", type=Path, default=here / "outputs/blr_conditional_split_conditional_bounds_extended.csv")
    parser.add_argument("--output-dir", type=Path, default=here / "outputs")
    args = parser.parse_args()
    summary = pd.read_csv(args.input)
    fig, fig_alpha = plot_conditional_output_paper_figures(summary)
    report = [save_paper_figure(fig, args.output_dir / "blr_conditional_split_conditional_bounds_extended", 4),
              save_paper_figure(fig_alpha, args.output_dir / "blr_conditional_split_conditional_bounds_extended_alpha_selected", 5)]
    plt.close(fig)
    plt.close(fig_alpha)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
