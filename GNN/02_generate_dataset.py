from __future__ import annotations

"""Generate synthetic IRP-LT BiGAT graph samples for debugging/pretraining.

These samples use heuristic labels. They are useful for smoke tests and
optional warm-up only; final thesis training should use solver-derived
teacher rows exported from the column-generation pipeline.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

import utilities


def generate_split(
    out_root: Path,
    split: str,
    n_samples: int,
    rng: np.random.Generator,
    n_columns: int,
    n_need_constraints: int,
    n_surplus_constraints: int,
    max_pairs_per_column: int,
) -> None:
    split_dir = out_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, n_samples + 1):
        sample = utilities.make_synthetic_irplt_graph(
            rng=rng,
            n_columns=int(rng.integers(max(4, n_columns // 2), n_columns + 1)),
            n_need_constraints=int(rng.integers(max(2, n_need_constraints // 2), n_need_constraints + 1)),
            n_surplus_constraints=int(rng.integers(max(2, n_surplus_constraints // 2), n_surplus_constraints + 1)),
            max_pairs_per_column=max_pairs_per_column,
        )
        utilities.save_graph_sample(sample, split_dir / f"sample_{idx:05d}.pkl")
    print(f"Wrote {n_samples} {split} samples to {split_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic/debug IRP-LT BiGAT graph samples, not final teacher-labeled thesis data.")
    parser.add_argument("--out-dir", default="GNN/data/irplt_synthetic_debug", help="Output synthetic/debug dataset root.")
    parser.add_argument("--seed", type=utilities.valid_seed, default=0)
    parser.add_argument("--train-size", type=int, default=200)
    parser.add_argument("--valid-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=60)
    parser.add_argument("--n-columns", type=int, default=80)
    parser.add_argument("--n-need-constraints", type=int, default=4)
    parser.add_argument("--n-surplus-constraints", type=int, default=4)
    parser.add_argument("--max-pairs-per-column", type=int, default=2)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    for split, n_samples in [
        ("train", args.train_size),
        ("valid", args.valid_size),
        ("test", args.test_size),
    ]:
        generate_split(
            out_root=out_root,
            split=split,
            n_samples=n_samples,
            rng=rng,
            n_columns=args.n_columns,
            n_need_constraints=args.n_need_constraints,
            n_surplus_constraints=args.n_surplus_constraints,
            max_pairs_per_column=args.max_pairs_per_column,
        )

    print("Column features:", utilities.COLUMN_FEATURE_NAMES)
    print("Constraint features:", utilities.CONSTRAINT_FEATURE_NAMES)
    print("Edge features:", utilities.EDGE_FEATURE_NAMES)


if __name__ == "__main__":
    main()
