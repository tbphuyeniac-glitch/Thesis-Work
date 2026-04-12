from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import argparse


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_step(title: str, args: list[str]) -> None:
    print("\n" + "=" * 80, flush=True)
    print(title, flush=True)
    print("=" * 80, flush=True)
    print(" ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny runner for the thesis IRP-LT BiGAT pipeline.")
    parser.add_argument(
        "--stage",
        choices=["all", "generate", "train", "test", "evaluate"],
        default="all",
        help="Run the whole tiny pipeline or only one stage.",
    )
    args = parser.parse_args()

    data_dir = "GNN/data/irplt_tiny"
    model_dir = "GNN/trained_models/irplt_tiny/bigat/0"
    checkpoint = f"{model_dir}/best_model.pt"
    test_csv = "GNN/results/irplt_tiny_test.csv"
    selected_json = "GNN/results/irplt_tiny_selected_columns.json"

    if args.stage in ("all", "generate"):
        run_step("STEP 1 - Generate tiny IRP-LT BiGAT dataset", [
        PYTHON,
        "GNN/02_generate_dataset.py",
        "--out-dir",
        data_dir,
        "--train-size",
        "8",
        "--valid-size",
        "4",
        "--test-size",
        "4",
        "--n-columns",
        "8",
        "--n-need-constraints",
        "4",
        "--n-surplus-constraints",
        "4",
        "--max-pairs-per-column",
        "2",
        "--overwrite",
        ])

    if args.stage in ("all", "train"):
        run_step("STEP 2 - Train BiGAT and save loss chart", [
        PYTHON,
        "GNN/03_train_bigat.py",
        "--data-dir",
        data_dir,
        "--out-dir",
        model_dir,
        "--epochs",
        "5",
        "--hidden-dim",
        "16",
        "--device",
        "cpu",
        ])

    if args.stage in ("all", "test"):
        run_step("STEP 3 - Test trained BiGAT on tiny test split", [
        PYTHON,
        "GNN/04_test.py",
        "--data-dir",
        data_dir,
        "--checkpoint",
        checkpoint,
        "--split",
        "test",
        "--out-file",
        test_csv,
        "--device",
        "cpu",
        ])

    if args.stage in ("all", "evaluate"):
        run_step("STEP 4 - Export selected top-k LT columns", [
        PYTHON,
        "GNN/evaluate_result.py",
        "--data-dir",
        data_dir,
        "--checkpoint",
        checkpoint,
        "--split",
        "test",
        "--top-k",
        "3",
        "--out-file",
        selected_json,
        "--device",
        "cpu",
        ])

    print("\n" + "=" * 80, flush=True)
    print(f"DONE - stage={args.stage}", flush=True)
    print("=" * 80, flush=True)
    print(f"Training log: {ROOT / model_dir / 'log.txt'}", flush=True)
    print(f"Training history JSON: {ROOT / model_dir / 'training_history.json'}", flush=True)
    print(f"Training loss chart: {ROOT / model_dir / 'training_loss_curve.png'}", flush=True)
    print(f"Model checkpoint: {ROOT / checkpoint}", flush=True)
    print(f"Test metrics CSV: {ROOT / test_csv}", flush=True)
    print(f"Selected columns JSON: {ROOT / selected_json}", flush=True)
    print("\nOpen the JSON file to see which LT candidate columns the GNN ranked highest.", flush=True)


if __name__ == "__main__":
    main()
