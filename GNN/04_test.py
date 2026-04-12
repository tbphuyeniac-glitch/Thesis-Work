from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

import utilities
from models.attention.model import BiGATColumnScorer


def load_checkpoint(path: str | Path, device: str):
    checkpoint = torch.load(path, map_location=device)
    config = dict(checkpoint["config"])
    dropout = config.pop("dropout", 0.0)
    model = BiGATColumnScorer(**config, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def apply_stats(samples, stats):
    normalized, _ = utilities.normalize_dataset(samples, stats=stats)
    return normalized


def evaluate(model, samples, device: str):
    rows = []
    for sample_id, sample in enumerate(samples):
        graph = {key: value.to(device) for key, value in sample.items()}
        with torch.no_grad():
            logits = model(
                graph["column_features"],
                graph["constraint_features"],
                graph["edge_index_col_to_con"],
                graph["edge_attr_col_to_con"],
            )
        labels = graph["labels"].detach().cpu()
        logits_cpu = logits.detach().cpu()
        metrics = utilities.binary_metrics(logits_cpu, labels)
        metrics.update(utilities.topk_accuracy(logits_cpu, labels, ks=(1, 3, 5)))
        metrics["sample_id"] = sample_id
        rows.append(metrics)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained IRP-LT BiGAT model.")
    parser.add_argument("--data-dir", default="GNN/data/irplt_tiny")
    parser.add_argument("--checkpoint", default="GNN/trained_models/irplt_tiny/bigat/0/best_model.pt")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--out-file", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    samples = utilities.load_split(args.data_dir, args.split)
    if not samples:
        raise RuntimeError(f"No {args.split} samples found under {args.data_dir}")
    samples = apply_stats(samples, checkpoint["normalization"])
    rows = evaluate(model, samples, args.device)

    summary = {}
    for key in rows[0]:
        if key == "sample_id":
            continue
        values = [row[key] for row in rows if not np.isnan(row[key])]
        summary[key] = float(np.mean(values)) if values else float("nan")

    print(f"Evaluated {len(rows)} {args.split} samples")
    for key, value in summary.items():
        print(f"{key}: {value:.6f}")

    out_file = args.out_file
    if out_file is None:
        out_file = f"GNN/results/irplt_tiny_{args.split}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "accuracy", "precision", "recall", "f1", "top1_hit", "top3_hit", "top5_hit"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-sample metrics to {out_path}")


if __name__ == "__main__":
    main()
