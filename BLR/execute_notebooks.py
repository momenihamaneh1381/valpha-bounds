from __future__ import annotations

import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


def execute_notebook(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    nbformat.write(notebook, path)


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parent
    mpl_config_dir = root / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    if argv:
        notebook_paths = [Path(arg).resolve() for arg in argv]
    else:
        notebook_paths = [root / "BLR.ipynb"]

    for path in notebook_paths:
        print(f"Executing {path} ...")
        execute_notebook(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
