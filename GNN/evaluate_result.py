from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import utilities
from models.attention.model import BiGATColumnScorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize IRP-LT BiGAT predictions and top-k selected columns.")
    parser.add_argument("--data-dir", default="GNN/data/irplt_tiny")
    parser.add_argument("--checkpoint", default="GNN/trained_models/irplt_tiny/bigat/0/best_model.pt")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out-file", default="GNN/results/irplt_tiny_selected_columns.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = dict(checkpoint["config"])
    dropout = config.pop("dropout", 0.0)
    model = BiGATColumnScorer(**config, dropout=dropout).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    raw_files = utilities.iter_sample_files(args.data_dir, args.split)
    samples = [utilities.load_graph_sample(path) for path in raw_files]
    samples, _ = utilities.normalize_dataset(samples, stats=checkpoint["normalization"])

    output = []
    hit_count = 0
    comparable = 0
    for path, sample in zip(raw_files, samples):
        graph = {key: value.to(args.device) for key, value in sample.items()}
        with torch.no_grad():
            logits = model(
                graph["column_features"],
                graph["constraint_features"],
                graph["edge_index_col_to_con"],
                graph["edge_attr_col_to_con"],
            )
            probs = torch.sigmoid(logits).detach().cpu()

        k_eff = min(args.top_k, probs.numel())
        top = torch.topk(probs, k_eff)
        labels = sample["labels"].detach().cpu()
        positives = set(torch.nonzero(labels > 0.5, as_tuple=False).flatten().tolist())
        selected = []
        for rank, (idx, score) in enumerate(zip(top.indices.tolist(), top.values.tolist()), start=1):
            selected.append({
                "rank": rank,
                "column_id": int(idx),
                "score": float(score),
                "label": float(labels[idx].item()),
            })
        if positives:
            comparable += 1
            if any(row["column_id"] in positives for row in selected):
                hit_count += 1
        output.append({
            "sample": str(path),
            "n_columns": int(probs.numel()),
            "n_positive_columns": int(len(positives)),
            "selected_columns": selected,
        })

    summary = {
        "split": args.split,
        "n_samples": len(output),
        "top_k": args.top_k,
        "top_k_hit_rate": hit_count / comparable if comparable else float("nan"),
    }
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": output}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved selected-column report to {out_path}")


if __name__ == "__main__":
    main()
