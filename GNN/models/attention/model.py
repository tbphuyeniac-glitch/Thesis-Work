from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BipartiteAttentionLayer(nn.Module):
    """One directed attention pass on a bipartite graph.

    The layer sends messages from source nodes to destination nodes using edge
    features in the attention score and in the message representation. It is a
    lightweight PyTorch-only equivalent of the BiGAT layer described in the
    thesis design, avoiding a hard dependency on torch_geometric.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        self.src_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dst_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.attn = nn.Linear(3 * hidden_dim, 1, bias=False)
        self.out = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src_x: torch.Tensor,
        dst_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.out(torch.cat([dst_x, torch.zeros_like(dst_x)], dim=-1))

        src_idx = edge_index[0].long()
        dst_idx = edge_index[1].long()

        src_h = self.src_proj(src_x)
        dst_h = self.dst_proj(dst_x)
        edge_h = self.edge_proj(edge_attr)

        raw = self.attn(torch.cat([src_h[src_idx], dst_h[dst_idx], edge_h], dim=-1)).squeeze(-1)
        raw = F.leaky_relu(raw, negative_slope=0.2)

        aggregated = torch.zeros_like(dst_h)
        for node_id in torch.unique(dst_idx):
            mask = dst_idx == node_id
            weights = torch.softmax(raw[mask], dim=0)
            weights = self.dropout(weights)
            messages = src_h[src_idx[mask]] + edge_h[mask]
            aggregated[node_id] = torch.sum(weights.unsqueeze(-1) * messages, dim=0)

        return self.out(torch.cat([dst_x, aggregated], dim=-1))


class BiGATColumnScorer(nn.Module):
    """Scores candidate LT columns in a column-constraint bipartite graph."""

    def __init__(
        self,
        column_dim: int = 12,
        constraint_dim: int = 4,
        edge_dim: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        column_feature_mask: Optional[Iterable[float]] = None,
    ):
        super().__init__()
        self.column_dim = int(column_dim)
        self.constraint_dim = int(constraint_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)

        # Per-column-feature gating mask. When None, all features are active
        # (legacy behaviour, bit-identical to previous checkpoints). When
        # provided, must be length column_dim with values in {0, 1}; the
        # mask is registered as a non-trainable buffer so it travels with
        # `state_dict()` and `to(device)`. Used by the rc-only ablation
        # variant to disable service-level features (shortage_ratio,
        # surplus_ratio, time_urgency, ...) at the input layer without
        # changing model architecture.
        if column_feature_mask is None:
            mask_tensor = torch.ones(self.column_dim, dtype=torch.float32)
        else:
            mask_list = [float(value) for value in column_feature_mask]
            if len(mask_list) != self.column_dim:
                raise ValueError(
                    f"column_feature_mask length {len(mask_list)} != column_dim {self.column_dim}"
                )
            mask_tensor = torch.tensor(mask_list, dtype=torch.float32)
        self.register_buffer("column_feature_mask", mask_tensor, persistent=True)

        self.column_encoder = nn.Sequential(
            nn.Linear(self.column_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.constraint_encoder = nn.Sequential(
            nn.Linear(self.constraint_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.column_norm0 = nn.LayerNorm(hidden_dim)
        self.constraint_norm0 = nn.LayerNorm(hidden_dim)

        self.col_to_con_1 = BipartiteAttentionLayer(hidden_dim, self.edge_dim, dropout)
        self.con_to_col_1 = BipartiteAttentionLayer(hidden_dim, self.edge_dim, dropout)
        self.col_to_con_2 = BipartiteAttentionLayer(hidden_dim, self.edge_dim, dropout)
        self.con_to_col_2 = BipartiteAttentionLayer(hidden_dim, self.edge_dim, dropout)
        self.column_norm1 = nn.LayerNorm(hidden_dim)
        self.constraint_norm1 = nn.LayerNorm(hidden_dim)
        self.column_norm2 = nn.LayerNorm(hidden_dim)
        self.constraint_norm2 = nn.LayerNorm(hidden_dim)

        self.scoring_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        column_features: torch.Tensor,
        constraint_features: torch.Tensor,
        edge_index_col_to_con: torch.Tensor,
        edge_attr_col_to_con: torch.Tensor,
    ) -> torch.Tensor:
        h_col, _ = self.encode_graph(
            column_features,
            constraint_features,
            edge_index_col_to_con,
            edge_attr_col_to_con,
        )
        return self.scoring_head(h_col).squeeze(-1)

    def encode_graph(
        self,
        column_features: torch.Tensor,
        constraint_features: torch.Tensor,
        edge_index_col_to_con: torch.Tensor,
        edge_attr_col_to_con: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return final column and constraint embeddings before scoring."""
        # Apply per-feature gating mask. Default mask is all ones, so this is
        # a no-op for legacy checkpoints (preserves bit-identical inference).
        masked_features = column_features.float() * self.column_feature_mask.to(column_features.device)
        h_col = self.column_norm0(self.column_encoder(masked_features))
        h_con = self.constraint_norm0(self.constraint_encoder(constraint_features.float()))

        edge_index_col_to_con = edge_index_col_to_con.long()
        edge_index_con_to_col = torch.stack(
            [edge_index_col_to_con[1], edge_index_col_to_con[0]], dim=0
        )

        h_con = self.constraint_norm1(
            h_con + self.col_to_con_1(h_col, h_con, edge_index_col_to_con, edge_attr_col_to_con.float())
        )
        h_col = self.column_norm1(
            h_col + self.con_to_col_1(h_con, h_col, edge_index_con_to_col, edge_attr_col_to_con.float())
        )
        h_con = self.constraint_norm2(
            h_con + self.col_to_con_2(h_col, h_con, edge_index_col_to_con, edge_attr_col_to_con.float())
        )
        h_col = self.column_norm2(
            h_col + self.con_to_col_2(h_con, h_col, edge_index_con_to_col, edge_attr_col_to_con.float())
        )

        return h_col, h_con

    def save(self, path: str | Path, **metadata) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "column_dim": self.column_dim,
                "constraint_dim": self.constraint_dim,
                "edge_dim": self.edge_dim,
                "hidden_dim": self.hidden_dim,
                # Persisted alongside state_dict so reloads honor the same
                # ablation mask the checkpoint was trained under. Stored as
                # a list of floats (json-friendly) for portability.
                "column_feature_mask": [
                    float(v) for v in self.column_feature_mask.tolist()
                ],
            },
            "metadata": metadata,
        }, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "BiGATColumnScorer":
        checkpoint = torch.load(path, map_location=map_location)
        config = dict(checkpoint.get("config", {}))
        # Backward compat: legacy checkpoints have no mask field. cls() will
        # default to an all-ones mask matching column_dim.
        model = cls(**config)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        return model


def score_graph(model: BiGATColumnScorer, graph: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Convenience inference wrapper for one graph sample."""
    model.eval()
    with torch.no_grad():
        return model(
            graph["column_features"],
            graph["constraint_features"],
            graph["edge_index_col_to_con"],
            graph["edge_attr_col_to_con"],
        )
