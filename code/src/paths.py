"""Resolve the code root, dataset folder, and output directories."""

import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = CODE_ROOT / "outputs" / "results"
FIGURE_DIR = CODE_ROOT / "outputs" / "figures"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_FILE = "experiment_1.csv"


def find_data_dir() -> Path:
    env = os.environ.get("TEACHPEER_DATA_DIR")
    candidates = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            CODE_ROOT / "data",
            CODE_ROOT.parent / "TeachPeer_dataset",
            CODE_ROOT.parent / "dataset",
        ]
    )
    for path in candidates:
        if (path / REQUIRED_FILE).is_file():
            return path
    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find the TeachPeer dataset.\n"
        "Copy the CSV files into paper_submission_code/data/, keep "
        "TeachPeer_dataset next to this folder, or set TEACHPEER_DATA_DIR.\n"
        f"Looked in:\n{searched}"
    )


DATA_DIR = find_data_dir()
