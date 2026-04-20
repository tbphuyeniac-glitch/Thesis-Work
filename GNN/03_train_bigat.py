from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utilities
from models.attention.model import BiGATColumnScorer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parameter_vector(model) -> torch.Tensor:
    return torch.cat([param.detach().flatten().cpu() for param in model.parameters()])


def collect_trace(model, samples, device="cpu"):
    model.eval()
    pos_scores = []
    neg_scores = []
    column_norms = []
    constraint_norms = []
    with torch.no_grad():
        for sample in samples:
            graph = {key: value.to(device) for key, value in sample.items()}
            column_embeddings, constraint_embeddings = model.encode_graph(
                graph["column_features"],
                graph["constraint_features"],
                graph["edge_index_col_to_con"],
                graph["edge_attr_col_to_con"],
            )
            logits = model.scoring_head(column_embeddings).squeeze(-1)
            probs = torch.sigmoid(logits)
            labels = graph.get("labels_binary", graph["labels"]) > 0.5
            if labels.any():
                pos_scores.append(probs[labels].detach().cpu())
            if (~labels).any():
                neg_scores.append(probs[~labels].detach().cpu())
            column_norms.append(column_embeddings.norm(dim=1).mean().detach().cpu())
            constraint_norms.append(constraint_embeddings.norm(dim=1).mean().detach().cpu())
    out = {
        "pos_score": float(torch.cat(pos_scores).mean()) if pos_scores else float("nan"),
        "neg_score": float(torch.cat(neg_scores).mean()) if neg_scores else float("nan"),
        "column_embedding_norm": float(torch.stack(column_norms).mean()),
        "constraint_embedding_norm": float(torch.stack(constraint_norms).mean()),
    }
    out["score_gap"] = out["pos_score"] - out["neg_score"]
    return out


def pairwise_ranking_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    pos_idx = torch.nonzero(labels > 0.5, as_tuple=False).flatten()
    neg_idx = torch.nonzero(labels <= 0.5, as_tuple=False).flatten()
    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        return logits.sum() * 0.0
    pos_scores = logits[pos_idx][:, None]
    neg_scores = logits[neg_idx][None, :]
    return F.softplus(-(pos_scores - neg_scores)).mean()


def score_regression_loss(logits: torch.Tensor, target_score: torch.Tensor) -> torch.Tensor:
    target = target_score.float()
    if target.numel() and target.max() > 1e-12:
        target = target / target.max().clamp_min(1e-12)
    return F.smooth_l1_loss(torch.sigmoid(logits), target)


def run_epoch(model, samples, optimizer=None, device="cpu", objective="pairwise_rank"):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_cols = 0
    metric_rows = []

    for sample in samples:
        graph = {key: value.to(device) for key, value in sample.items()}
        logits = model(
            graph["column_features"],
            graph["constraint_features"],
            graph["edge_index_col_to_con"],
            graph["edge_attr_col_to_con"],
        )
        labels = graph.get("labels_binary", graph["labels"])
        if objective == "pairwise_rank":
            loss = pairwise_ranking_loss(logits, labels)
            if float(loss.detach().cpu()) == 0.0:
                pos_count = labels.sum().clamp_min(1.0)
                neg_count = (labels.numel() - labels.sum()).clamp_min(1.0)
                pos_weight = (neg_count / pos_count).detach()
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        elif objective == "score_regression":
            loss = score_regression_loss(logits, graph.get("target_score", labels))
        else:
            pos_count = labels.sum().clamp_min(1.0)
            neg_count = (labels.numel() - labels.sum()).clamp_min(1.0)
            pos_weight = (neg_count / pos_count).detach()
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += float(loss.detach().cpu()) * labels.numel()
        total_cols += int(labels.numel())
        with torch.no_grad():
            row = {
                f"binary_{key}": value
                for key, value in utilities.binary_metrics(logits.detach().cpu(), labels.detach().cpu()).items()
            }
            row.update({
                f"ranking_{key}": value
                for key, value in utilities.topk_accuracy(logits.detach().cpu(), labels.detach().cpu(), ks=(1, 3, 5)).items()
            })
            pos = torch.nonzero(labels.detach().cpu() > 0.5, as_tuple=False).flatten()
            if pos.numel() > 0:
                order = torch.argsort(logits.detach().cpu(), descending=True)
                ranks = torch.empty_like(order)
                ranks[order] = torch.arange(1, order.numel() + 1)
                pos_ranks = ranks[pos].float()
                row["ranking_mean_positive_rank"] = float(pos_ranks.mean())
                row["ranking_mrr"] = float((1.0 / pos_ranks.min()).item())
            else:
                row["ranking_mean_positive_rank"] = float("nan")
                row["ranking_mrr"] = float("nan")
            metric_rows.append(row)

    metrics = {"loss": total_loss / max(total_cols, 1)}
    for key in metric_rows[0]:
        values = [row[key] for row in metric_rows if not np.isnan(row[key])]
        metrics[key] = float(np.mean(values)) if values else float("nan")
    return metrics


def save_loss_chart(history, out_path: Path) -> None:
    episodes = [row.get("epoch", row.get("episode")) for row in history]
    train_loss = [row["train_loss"] for row in history]
    valid_loss = [row["valid_loss"] for row in history]

    plt.figure(figsize=(8, 4.5))
    plt.plot(episodes, train_loss, marker="o", label="Train loss")
    plt.plot(episodes, valid_loss, marker="o", label="Validation loss")
    plt.xlabel("Training epoch")
    plt.ylabel("Loss")
    plt.title("BiGAT Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the IRP-LT BiGAT column scorer.")
    parser.add_argument("--data-dir", default="GNN/data/irplt_teacher")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dataset-type", choices=["teacher", "synthetic"], default="teacher")
    parser.add_argument("--seed", type=utilities.valid_seed, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--objective", choices=["binary", "pairwise_rank", "score_regression"], default="pairwise_rank")
    parser.add_argument("--resume-checkpoint", default=None, help="Optional checkpoint to fine-tune from.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.out_dir is None:
        if args.dataset_type == "teacher":
            args.out_dir = f"GNN/trained_models/irplt_teacher/bigat/{args.objective}"
        else:
            args.out_dir = f"GNN/trained_models/irplt_synthetic_debug/bigat/{args.objective}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / "log.txt"

    train_samples = utilities.load_split(args.data_dir, "train")
    valid_samples = utilities.load_split(args.data_dir, "valid")
    if not train_samples or not valid_samples:
        hint = (
            "Run GNN/build_teacher_graph_dataset.py after exporting Results/cg_teacher_dataset.csv from CG."
            if args.dataset_type == "teacher"
            else "Run GNN/02_generate_dataset.py for synthetic debug/warm-up samples."
        )
        raise RuntimeError(f"No train/valid samples found under {args.data_dir}. {hint}")

    label_sources = {}
    sample_paths = utilities.iter_sample_files(args.data_dir, "train") + utilities.iter_sample_files(args.data_dir, "valid")
    for path in sample_paths:
        metadata = utilities.load_raw_sample(path).get("metadata", {})
        source = metadata.get("label_source", "unknown")
        label_sources[source] = label_sources.get(source, 0) + 1
    if args.dataset_type == "teacher" and any(source != "teacher_rmp" for source in label_sources):
        raise RuntimeError(f"Teacher training requested, but non-teacher label sources were found: {label_sources}")

    train_samples, stats = utilities.normalize_dataset(train_samples)
    valid_samples, _ = utilities.normalize_dataset(valid_samples, stats=stats)

    model = BiGATColumnScorer(
        column_dim=train_samples[0]["column_features"].shape[1],
        constraint_dim=train_samples[0]["constraint_features"].shape[1],
        edge_dim=train_samples[0]["edge_attr_col_to_con"].shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # For pairwise_rank the primary metric is MRR (higher = better).
    # For other objectives, lower valid_loss is better.
    _maximize_primary = (args.objective == "pairwise_rank")
    best_valid = float("-inf") if _maximize_primary else float("inf")
    bad_epochs = 0
    best_path = out_dir / "best_model.pt"
    chart_path = out_dir / "training_loss_curve.png"
    history_path = out_dir / "training_history.json"
    history = []
    start_epoch = 1
    checkpoint = None
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location=args.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(resume_path, map_location=args.device)
        model_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in checkpoint["state_dict"].items()
            if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
        }
        skipped = sorted(set(checkpoint["state_dict"]) - set(compatible_state))
        model.load_state_dict(compatible_state, strict=False)
        utilities.log(f"resumed from checkpoint={resume_path}", logfile)
        if skipped:
            utilities.log(f"skipped incompatible checkpoint tensors={len(skipped)}", logfile)
        if checkpoint is not None and "optimizer_state" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                utilities.log("resumed optimizer state", logfile)
            except ValueError as exc:
                utilities.log(f"skipped incompatible optimizer state: {exc}", logfile)
        if checkpoint is not None:
            if _maximize_primary:
                best_valid = float(checkpoint.get("best_valid_mrr",
                                   checkpoint.get("valid_metrics", {}).get("ranking_mrr", float("-inf"))))
            else:
                best_valid = float(checkpoint.get("best_valid_loss",
                                   checkpoint.get("valid_metrics", {}).get("loss", float("inf"))))
            start_epoch = int(checkpoint.get("last_epoch", 0)) + 1
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            if history:
                start_epoch = max(start_epoch, int(history[-1].get("epoch", 0)) + 1)
            utilities.log(f"loaded existing training history rows={len(history)}", logfile)

    utilities.log(f"train_samples={len(train_samples)} valid_samples={len(valid_samples)}", logfile)
    utilities.log(f"dataset_type={args.dataset_type} data_dir={args.data_dir} label_sources={label_sources}", logfile)
    utilities.log(f"model=BiGAT device={args.device} hidden_dim={args.hidden_dim}", logfile)
    primary_metric = "ranking_mrr" if args.objective == "pairwise_rank" else "binary_f1"
    if args.objective == "score_regression":
        primary_metric = "valid_loss"
    utilities.log(
        f"objective={args.objective} primary_metric={primary_metric} "
        "ranking_metrics=valid_loss,mrr,mean_positive_rank,topk "
        "binary_threshold_metrics=secondary",
        logfile,
    )

    for epoch in range(start_epoch, start_epoch + args.epochs):
        random.shuffle(train_samples)
        before_params = parameter_vector(model)
        train_metrics = run_epoch(model, train_samples, optimizer=optimizer, device=args.device, objective=args.objective)
        after_params = parameter_vector(model)
        update_norm = float((after_params - before_params).norm())
        valid_metrics = run_epoch(model, valid_samples, optimizer=None, device=args.device, objective=args.objective)
        trace = collect_trace(model, valid_samples, device=args.device)
        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "valid_loss": valid_metrics["loss"],
            "primary_metric": primary_metric,
            "ranking_valid_mrr": valid_metrics.get("ranking_mrr", float("nan")),
            "ranking_valid_mean_positive_rank": valid_metrics.get("ranking_mean_positive_rank", float("nan")),
            "ranking_valid_top1": valid_metrics.get("ranking_top1_hit", float("nan")),
            "ranking_valid_top3": valid_metrics.get("ranking_top3_hit", float("nan")),
            "ranking_valid_top5": valid_metrics.get("ranking_top5_hit", float("nan")),
            "binary_valid_f1": valid_metrics.get("binary_f1", float("nan")),
            "binary_valid_precision": valid_metrics.get("binary_precision", float("nan")),
            "binary_valid_recall": valid_metrics.get("binary_recall", float("nan")),
            "valid_f1": valid_metrics.get("binary_f1", float("nan")),
            "valid_top1": valid_metrics.get("ranking_top1_hit", float("nan")),
            "valid_mrr": valid_metrics.get("ranking_mrr", float("nan")),
            "valid_mean_positive_rank": valid_metrics.get("ranking_mean_positive_rank", float("nan")),
            "pos_score": trace["pos_score"],
            "neg_score": trace["neg_score"],
            "score_gap": trace["score_gap"],
            "column_embedding_norm": trace["column_embedding_norm"],
            "constraint_embedding_norm": trace["constraint_embedding_norm"],
            "param_update_norm": update_norm,
        }
        history.append(history_row)
        utilities.log(
            f"epoch={epoch:03d} objective={args.objective} "
            f"train_loss={train_metrics['loss']:.4f} valid_loss={valid_metrics['loss']:.4f} "
            f"ranking_valid_mrr={valid_metrics.get('ranking_mrr', float('nan')):.4f} "
            f"ranking_valid_top1={valid_metrics.get('ranking_top1_hit', float('nan')):.4f} "
            f"ranking_valid_top3={valid_metrics.get('ranking_top3_hit', float('nan')):.4f} "
            f"ranking_mean_positive_rank={valid_metrics.get('ranking_mean_positive_rank', float('nan')):.4f} "
            f"binary_valid_f1={valid_metrics.get('binary_f1', float('nan')):.4f} "
            f"pos_score={trace['pos_score']:.4f} neg_score={trace['neg_score']:.4f} "
            f"score_gap={trace['score_gap']:.4f} "
            f"col_emb_norm={trace['column_embedding_norm']:.4f} "
            f"con_emb_norm={trace['constraint_embedding_norm']:.4f} "
            f"param_update_norm={update_norm:.6f}",
            logfile,
        )

        if _maximize_primary:
            current_primary = valid_metrics.get("ranking_mrr", float("nan"))
            improved = not math.isnan(current_primary) and current_primary > best_valid
        else:
            current_primary = valid_metrics["loss"]
            improved = current_primary < best_valid
        if improved:
            best_valid = current_primary
            bad_epochs = 0
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "column_dim": model.column_dim,
                    "constraint_dim": model.constraint_dim,
                    "edge_dim": model.edge_dim,
                    "hidden_dim": model.hidden_dim,
                    "dropout": args.dropout,
                },
                "normalization": stats,
                "feature_names": {
                    "column": utilities.COLUMN_FEATURE_NAMES,
                    "constraint": utilities.CONSTRAINT_FEATURE_NAMES,
                    "edge": utilities.EDGE_FEATURE_NAMES,
                },
                "valid_metrics": valid_metrics,
                "optimizer_state": optimizer.state_dict(),
                "best_valid_loss": best_valid if not _maximize_primary else valid_metrics["loss"],
                "best_valid_mrr": best_valid if _maximize_primary else valid_metrics.get("ranking_mrr", float("nan")),
                "last_epoch": epoch,
                "objective": args.objective,
                "dataset_type": args.dataset_type,
                "label_sources": label_sources,
            }, best_path)
            utilities.log("saved best BiGAT model", logfile)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                utilities.log("early stopping", logfile)
                break

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    save_loss_chart(history, chart_path)
    utilities.log(f"saved training history to {history_path}", logfile)
    utilities.log(f"saved training loss chart to {chart_path}", logfile)

    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_valid_loss": best_valid,
            "objective": args.objective,
            "dataset_type": args.dataset_type,
            "data_dir": args.data_dir,
            "resume_checkpoint": args.resume_checkpoint,
            "label_sources": label_sources,
            "last_epoch": history[-1]["epoch"] if history else 0,
            "history_rows": len(history),
            "checkpoint": str(best_path),
            "history": str(history_path),
            "loss_chart": str(chart_path),
        }, f, indent=2)


if __name__ == "__main__":
    main()
