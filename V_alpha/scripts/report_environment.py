"""Write a machine-readable reproduction environment report."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys

import torch


PACKAGES = [
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "Pillow",
    "torch",
    "torchvision",
    "tensorboard",
    "tqdm",
    "nbformat",
    "nbclient",
    "ipykernel",
    "tensorflow",
    "tensorflow-datasets",
]


def package_versions():
    out = {}
    for name in PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def nvidia_smi_metadata():
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ]
    try:
        proc = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    rows = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", maxsplit=1)]
        rows.append({
            "name": parts[0],
            "driver_version": parts[1] if len(parts) > 1 else None,
        })
    return rows


def build_report():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "torch": {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            ),
            "visible_cuda_devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
        },
        "nvidia_smi": nvidia_smi_metadata(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="optional JSON output path")
    args = parser.parse_args()

    report = build_report()
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
