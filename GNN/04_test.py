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
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    return model, checkpoint


def apply_stats(samples, stats):
    normalized, _ = utilities.normalize_dataset(samples, stats=stats)
    return normalized


def ranking_metrics(logits: torch.Tensor, labels: torch.Tensor):
    positives = torch.nonzero(labels > 0.5, as_tuple=False).flatten()
    if positives.numel() == 0:
        return {"mrr": float("nan"), "first_positive_rank": float("nan"), "mean_positive_rank": float("nan"), "ndcg": float("nan")}
    order = torch.argsort(logits, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, order.numel() + 1)
    pos_ranks = ranks[positives].float()
    gains = labels[order].float()
    discounts = 1.0 / torch.log2(torch.arange(2, gains.numel() + 2).float())
    dcg = float((gains * discounts).sum())
    ideal = torch.sort(labels.float(), descending=True).values
    idcg = float((ideal * discounts).sum())
    return {
        "mrr": float(1.0 / pos_ranks.min()),
        "first_positive_rank": float(pos_ranks.min()),
        "mean_positive_rank": float(pos_ranks.mean()),
        "ndcg": dcg / max(idcg, 1e-12),
    }


def evaluate(model, samples, device: str, mass_thresholds):
    rows = []
    forward_times: list[float] = []
    for sample_id, sample in enumerate(samples):
        graph = {key: value.to(device) for key, value in sample.items()}
        _fwd_start = time.perf_counter()
        with torch.no_grad():
            logits = model(
                graph["column_features"],
                graph["constraint_features"],
                graph["edge_index_col_to_con"],
                graph["edge_attr_col_to_con"],
            )
        forward_times.append(time.perf_counter() - _fwd_start)
        labels = graph.get("labels_binary", graph["labels"]).detach().cpu()
        logits_cpu = logits.detach().cpu()
        probs = torch.sigmoid(logits_cpu)
        metrics = utilities.binary_metrics(logits_cpu, labels)
        metrics.update(utilities.topk_accuracy(logits_cpu, labels, ks=(1, 3, 5)))
        metrics.update(ranking_metrics(logits_cpu, labels))
        metrics["score_mean"] = float(probs.mean())
        metrics["score_std"] = float(probs.std(unbiased=False)) if probs.numel() > 1 else 0.0
        gold = labels > 0.5
        for mass_threshold in mass_thresholds:
            selected_list, info = utilities.adaptive_select_indices(
                probs.tolist(),
                selection_mode="cumulative_mass",
                mass_threshold=mass_threshold,
                min_keep=1,
            )
            selected_idx = torch.as_tensor(selected_list, dtype=torch.long)
            selected = torch.zeros_like(labels, dtype=torch.bool)
            selected[selected_idx] = True
            tp = torch.logical_and(selected, gold).sum().item()
            fp = torch.logical_and(selected, ~gold).sum().item()
            fn = torch.logical_and(~selected, gold).sum().item()
            row = dict(metrics)
            row["mass_threshold"] = float(mass_threshold)
            row["adaptive_k_star"] = float(info["adaptive_k"])
            row["selected_fraction"] = float(info["adaptive_k"]) / max(float(probs.numel()), 1.0)
            row["adaptive_precision"] = tp / max(tp + fp, 1)
            row["adaptive_recall"] = tp / max(tp + fn, 1)
            row["adaptive_f1"] = 2 * row["adaptive_precision"] * row["adaptive_recall"] / max(row["adaptive_precision"] + row["adaptive_recall"], 1e-9)
            row["adaptive_hit"] = float(tp > 0) if gold.any() else float("nan")
            row["adaptive_zero_selected"] = float(info["adaptive_k"] == 0)
            row["sample_id"] = sample_id
            rows.append(row)
    return rows, forward_times


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained IRP-LT BiGAT model.")
    parser.add_argument("--data-dir", default="GNN/data/irplt_teacher")
    parser.add_argument("--checkpoint", default="GNN/trained_models/irplt_teacher/bigat/pairwise_rank/best_model.pt")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--out-file", default=None)
    parser.add_argument("--mass-threshold", type=float, default=None, help="Single cumulative-mass threshold to evaluate.")
    parser.add_argument("--mass-thresholds", type=float, nargs="+", default=[0.60, 0.70, 0.80])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    samples = utilities.load_split(args.data_dir, args.split)
    if not samples:
        raise RuntimeError(f"No {args.split} samples found under {args.data_dir}")
    samples = apply_stats(samples, checkpoint["normalization"])
    mass_thresholds = [args.mass_threshold] if args.mass_threshold is not None else args.mass_thresholds
    _eval_start = time.perf_counter()
    rows, forward_times = evaluate(model, samples, args.device, mass_thresholds=mass_thresholds)
    total_eval_seconds = time.perf_counter() - _eval_start

    summaries = []
    for threshold in sorted({row["mass_threshold"] for row in rows}):
        threshold_rows = [row for row in rows if row["mass_threshold"] == threshold]
        summary = {"mass_threshold": threshold}
        for key in threshold_rows[0]:
            if key == "sample_id":
                continue
            values = [row[key] for row in threshold_rows if not np.isnan(row[key])]
            summary[key] = float(np.mean(values)) if values else float("nan")
        summaries.append(summary)

    print(f"Evaluated {len(samples)} {args.split} samples across {len(mass_thresholds)} threshold setting(s)")
    for summary in summaries:
        print(f"\nthreshold={summary['mass_threshold']:.2f}")
        for key, value in summary.items():
            if key == "mass_threshold":
                continue
            print(f"{key}: {value:.6f}")

    out_file = args.out_file
    if out_file is None:
        out_file = f"GNN/results/irplt_teacher_{args.split}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id", "mass_threshold", "accuracy", "precision", "recall", "f1",
        "top1_hit", "top3_hit", "top5_hit", "mrr", "first_positive_rank",
        "mean_positive_rank", "ndcg", "adaptive_k_star", "adaptive_precision",
        "adaptive_recall", "adaptive_f1", "adaptive_hit", "adaptive_zero_selected",
        "score_mean", "score_std", "selected_fraction",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-sample metrics to {out_path}")

    runtime_path = out_path.parent / "test_runtime_seconds.json"
    total_forward = float(sum(forward_times))
    runtime_payload = {
        "runtime_category": "offline_gnn_test",
        "description": "Model forward-pass runtime on held-out graph split. Does NOT include solver runtime; see thesis_summary/runtime_breakdown.json for end-to-end categories.",
        "n_samples": len(samples),
        "mass_thresholds": list(mass_thresholds),
        "forward_pass_seconds_total": total_forward,
        "forward_pass_seconds_mean": total_forward / max(len(forward_times), 1),
        "forward_pass_seconds_min": float(min(forward_times)) if forward_times else 0.0,
        "forward_pass_seconds_max": float(max(forward_times)) if forward_times else 0.0,
        "evaluate_function_seconds": float(total_eval_seconds),
        "device": args.device,
    }
    with open(runtime_path, "w", encoding="utf-8") as f:
        import json as _json
        _json.dump(runtime_payload, f, indent=2)
    print(f"Saved offline-GNN-test runtime to {runtime_path}")


if __name__ == "__main__":
    main()
