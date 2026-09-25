"""
Run the complete analysis pipeline.

Usage:
    python run_all_analyses.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

STEPS = [
    ("0_prepare_data.py", "Data preparation and validation"),
    ("1_baseline_llm_simulation.py", "Experiment 1: baseline LLM comparison"),
    ("2_llm_peer_generation.py", "LLM-peer essay validation"),
    ("3_statistical_analysis.py", "Experiment 2: statistical analysis"),
]


def run_command(cmd, description):
    print("\n" + "=" * 72)
    print(description)
    print("=" * 72)
    try:
        subprocess.run(cmd, check=True, cwd=ROOT)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Failed with exit code {exc.returncode}")
        return False
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")
        return False


def main():
    results = {}
    for script, description in STEPS:
        results[script] = run_command([sys.executable, str(SRC / script)], description)

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    for name, success in results.items():
        print(f"  {name:<32} {'ok' if success else 'failed'}")

    if not all(results.values()):
        sys.exit(1)
    print("\nResults written to outputs/results/")
    print("Figures written to outputs/figures/")


if __name__ == "__main__":
    main()
