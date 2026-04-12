from __future__ import annotations

"""Compatibility shim.

The thesis GNN pipeline now uses PyTorch instead of the original TensorFlow 1
SCIP-branching code. This module is kept so old imports fail gracefully and can
still load the new IRP-LT graph samples.
"""

from typing import Any, Dict

import torch

from utilities import graph_to_tensors, load_graph_sample


def load_batch_bigat(sample_files):
    raise RuntimeError(
        "load_batch_bigat is not used by the PyTorch runner. "
        "Use utilities.load_graph_sample(...) with the PyTorch BiGAT scripts."
    )


def load_graph_for_torch(path: str, device: str | torch.device | None = None) -> Dict[str, torch.Tensor]:
    sample = load_graph_sample(path)
    if device is None:
        return sample
    return {key: value.to(device) for key, value in sample.items()}


def as_torch_graph(sample: Dict[str, Any], device: str | torch.device | None = None) -> Dict[str, torch.Tensor]:
    return graph_to_tensors(sample, device=device)
