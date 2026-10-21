"""Reproduce the seven paper figures from included results and analytical formulas."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["all", "deep-learning", "blr", "canonical"], default="all")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = (args.output_dir or root / "figures").resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1", "PYTHON_BIN": sys.executable}
    if args.only in {"all", "deep-learning"}:
        subprocess.run(["bash", "reproduce/figures_from_aggregates.sh"], cwd=root / "V_alpha", env={**env, "FIGURES_DIR": str(out)}, check=True)
    if args.only in {"all", "blr"}:
        with tempfile.TemporaryDirectory(prefix="tmlr-blr-") as temporary:
            tmp = Path(temporary)
            subprocess.run([sys.executable, "-B", "paper_figures.py", "--output-dir", str(tmp)], cwd=root / "BLR", env=env, check=True)
            for suffix in ["", "_alpha_selected"]:
                shutil.copy2(tmp / f"blr_conditional_split_conditional_bounds_extended{suffix}.pdf", out / f"06_extended_split_conditional_bounds_extended{suffix}.pdf")
    if args.only in {"all", "canonical"}:
        with tempfile.TemporaryDirectory(prefix="tmlr-canonical-") as temporary:
            tmp = Path(temporary)
            subprocess.run([sys.executable, "-B", "plot_figures.py", "--output-dir", str(tmp)], cwd=root / "canonical_channels", env=env, check=True)
            shutil.copy2(tmp / "Canonical-channels-1x3.pdf", out / "Canonical-channels.pdf")
            shutil.copy2(tmp / "Trade-off.pdf", out / "Trade-off.pdf")
    print(f"Figures saved to {out}")


if __name__ == "__main__":
    main()
