from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import utilities
from models.attention.model import BiGATColumnScorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize IRP-LT BiGAT adaptive self-defined-k selected columns.")
    parser.add_argument("--data-dir", default="GNN/data/irplt_teacher")
    parser.add_argument("--checkpoint", nargs="+", default=["GNN/trained_models/irplt_teacher/bigat/pairwise_rank/best_model.pt"])
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--selection-mode", default="cumulative_mass", choices=["cumulative_mass", "relative_threshold"])
    parser.add_argument("--mass-threshold", type=float, default=0.80)
    parser.add_argument("--relative-threshold", type=float, default=0.85)
    parser.add_argument("--min-keep", type=int, default=1)
    parser.add_argument("--max-keep", type=int, default=None)
    parser.add_argument("--out-file", default="GNN/results/irplt_teacher_selected_columns.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    raw_files = utilities.iter_sample_files(args.data_dir, args.split)
    if not raw_files:
        raise RuntimeError(f"No {args.split} samples found under {args.data_dir}")

    checkpoint_reports = []
    for checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        config = dict(checkpoint["config"])
        dropout = config.pop("dropout", 0.0)
        model = BiGATColumnScorer(**config, dropout=dropout).to(args.device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.eval()

        samples = [utilities.load_graph_sample(path) for path in raw_files]
        samples, _ = utilities.normalize_dataset(samples, stats=checkpoint["normalization"])

        output = []
        hit_count = 0
        comparable = 0
        selected_sizes = []
        false_positives = []
        precisions = []
        recalls = []
        f1s = []
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

            selected_list, info = utilities.adaptive_select_indices(
                probs.tolist(),
                selection_mode=args.selection_mode,
                mass_threshold=args.mass_threshold,
                relative_threshold=args.relative_threshold,
                min_keep=args.min_keep,
                max_keep=args.max_keep,
            )
            order = info["ordered_indices"]
            selected_set = set(selected_list)
            labels = sample.get("labels_binary", sample["labels"]).detach().cpu()
            positives = set(torch.nonzero(labels > 0.5, as_tuple=False).flatten().tolist())
            selected = []
            for rank, idx in enumerate(selected_list, start=1):
                selected.append({
                    "rank": rank,
                    "column_id": int(idx),
                    "score": float(probs[idx].item()),
                    "label": float(labels[idx].item()),
                })
            ranked_columns = [
                {
                    "rank": rank,
                    "column_id": int(idx),
                    "score": float(probs[idx].item()),
                    "label": float(labels[idx].item()),
                    "selected": int(idx) in selected_set,
                }
                for rank, idx in enumerate(order, start=1)
            ]
            if positives:
                comparable += 1
                if selected_set & positives:
                    hit_count += 1
            tp = len(selected_set & positives)
            fp = len(selected_set - positives)
            fn = len(positives - selected_set)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)
            selected_sizes.append(info["adaptive_k"])
            false_positives.append(fp)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            output.append({
                "sample": str(path),
                "n_columns": int(probs.numel()),
                "n_positive_columns": int(len(positives)),
                "adaptive_k_star": int(info["adaptive_k"]),
                "selected_fraction": float(info["adaptive_k"]) / max(float(probs.numel()), 1.0),
                "teacher_positive_columns": sorted(int(idx) for idx in positives),
                "selected_columns": selected,
                "ranked_columns": ranked_columns,
                "overlap": {
                    "true_positive": int(tp),
                    "false_positive": int(fp),
                    "false_negative": int(fn),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                },
            })

        size_values, size_counts = np.unique(selected_sizes, return_counts=True) if selected_sizes else ([], [])
        summary = {
            "checkpoint": checkpoint_path,
            "split": args.split,
            "n_samples": len(output),
            "selection_mode": args.selection_mode,
            "mass_threshold": args.mass_threshold,
            "relative_threshold": args.relative_threshold,
            "adaptive_hit_rate": hit_count / comparable if comparable else float("nan"),
            "avg_selected_size": float(np.mean(selected_sizes)) if selected_sizes else float("nan"),
            "median_selected_size": float(np.median(selected_sizes)) if selected_sizes else float("nan"),
            "min_selected_size": int(min(selected_sizes)) if selected_sizes else 0,
            "max_selected_size": int(max(selected_sizes)) if selected_sizes else 0,
            "selected_size_frequency": {str(int(size)): int(count) for size, count in zip(size_values, size_counts)},
            "avg_false_positives": float(np.mean(false_positives)) if false_positives else float("nan"),
            "max_false_positives": int(max(false_positives)) if false_positives else 0,
            "selected_set_precision": float(np.mean(precisions)) if precisions else float("nan"),
            "selected_set_recall": float(np.mean(recalls)) if recalls else float("nan"),
            "selected_set_f1": float(np.mean(f1s)) if f1s else float("nan"),
        }
        checkpoint_reports.append({"summary": summary, "samples": output})

    summary = checkpoint_reports[0]["summary"] if len(checkpoint_reports) == 1 else {
        "split": args.split,
        "n_checkpoints": len(checkpoint_reports),
        "checkpoints": [report["summary"] for report in checkpoint_reports],
    }
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        payload = {"summary": summary, "samples": checkpoint_reports[0]["samples"]} if len(checkpoint_reports) == 1 else {"summary": summary, "checkpoint_reports": checkpoint_reports}
        json.dump(payload, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved selected-column report to {out_path}")


if __name__ == "__main__":
    main()
