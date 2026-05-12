"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import MAX_LENGTH, _DEFAULT_MODEL


DATA_FILES = (Path("data/dataset.csv"), Path("data/test.csv"))
PROMPT_LENS = None
CURSOR = 0


def _prompt_lens():
    global PROMPT_LENS
    if PROMPT_LENS is not None:
        return PROMPT_LENS

    prompts = []
    for path in DATA_FILES:
        with path.open(newline="", encoding="utf-8") as f:
            prompts.extend(str(row.get("prompt", "")) for row in csv.DictReader(f))

    tokenizer = AutoTokenizer.from_pretrained(_DEFAULT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lens = []
    for start in range(0, len(prompts), 64):
        enc = tokenizer(
            prompts[start : start + 64],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        lens.extend(enc["attention_mask"].sum(dim=1).cpu().tolist())

    PROMPT_LENS = [int(x) for x in lens]
    return PROMPT_LENS


def _next_prompt_len():
    global CURSOR
    lens = _prompt_lens()
    if CURSOR >= len(lens):
        CURSOR += 1
        return None
    value = lens[CURSOR]
    CURSOR += 1
    return value


def _real_pos(attention_mask):
    pos = torch.nonzero(attention_mask.bool().cpu(), as_tuple=False).flatten()
    if pos.numel() == 0:
        pos = torch.tensor([0], dtype=torch.long)
    return pos


def _positions(attention_mask, prompt_len, k, scope):
    pos = _real_pos(attention_mask)
    if scope == "full" or prompt_len is None:
        selected = pos
    else:
        boundary = int(pos[0].item()) + min(prompt_len, int(pos[-1].item()) + 1)
        selected = pos[pos >= boundary]
        if selected.numel() == 0:
            selected = pos
        if scope == "response_no_last" and selected.numel() > 1:
            selected = selected[:-1]
    return selected[-k:].to(dtype=torch.long)


def _layer(hidden_states, layer_idx):
    return hidden_states[min(layer_idx, hidden_states.shape[0] - 1)]


def _last_token(hidden_states, attention_mask, layer_idx, offset=0):
    layer = _layer(hidden_states, layer_idx)
    pos = _real_pos(attention_mask)
    idx = max(0, pos.numel() - 1 - offset)
    return layer[int(pos[idx].item())]


def _pool(hidden_states, attention_mask, layer_idx, scope, pool, k, prompt_len):
    layer = _layer(hidden_states, layer_idx)
    pos = _positions(attention_mask, prompt_len, k, scope).to(layer.device)
    values = layer.index_select(0, pos)
    if pool == "mean":
        return values.mean(dim=0)
    return values.max(dim=0).values


def _branch(hidden_states, attention_mask, layer_idx, scope, pool, k, prompt_len):
    pooled = _pool(hidden_states, attention_mask, layer_idx, scope, pool, k, prompt_len)
    last = _last_token(hidden_states, attention_mask, layer_idx)
    return torch.cat([pooled, last], dim=0)


def _cos(a, b):
    return F.cosine_similarity(a[None, :], b[None, :], dim=1, eps=1e-8).reshape(1)


def _geometry(hidden_states, attention_mask, prompt_len):
    layers = tuple(x for x in (12, 15, 19) if x < hidden_states.shape[0])
    if not layers:
        layers = (hidden_states.shape[0] - 1,)

    views = {key: [] for key in [
        "last", "second_last", "response_mean_64", "response_max_16",
        "response_max_32", "response_no_last_mean_64",
    ]}
    for layer_idx in layers:
        views["last"].append(_last_token(hidden_states, attention_mask, layer_idx))
        views["second_last"].append(_last_token(hidden_states, attention_mask, layer_idx, 1))
        views["response_mean_64"].append(_pool(hidden_states, attention_mask, layer_idx, "response", "mean", 64, prompt_len))
        views["response_max_16"].append(_pool(hidden_states, attention_mask, layer_idx, "response", "max", 16, prompt_len))
        views["response_max_32"].append(_pool(hidden_states, attention_mask, layer_idx, "response", "max", 32, prompt_len))
        views["response_no_last_mean_64"].append(_pool(hidden_states, attention_mask, layer_idx, "response_no_last", "mean", 64, prompt_len))

    views = {key: torch.stack(value, dim=0) for key, value in views.items()}
    out = []
    for arr in views.values():
        out += [
            torch.linalg.vector_norm(arr, dim=1),
            arr.mean(dim=1),
            arr.std(dim=1, unbiased=False),
            arr.max(dim=1).values,
            arr.min(dim=1).values,
        ]
        for i, j in combinations(range(len(layers)), 2):
            out.append(_cos(arr[i], arr[j]))
            out.append(torch.linalg.vector_norm(arr[i] - arr[j]).reshape(1))

    for a_key, b_key in [
        ("last", "response_mean_64"),
        ("last", "response_no_last_mean_64"),
        ("second_last", "response_mean_64"),
    ]:
        for i in range(len(layers)):
            out.append(_cos(views[a_key][i], views[b_key][i]))
            out.append(torch.linalg.vector_norm(views[a_key][i] - views[b_key][i]).reshape(1))

    return torch.cat(out, dim=0).float()


def _aggregate(hidden_states, attention_mask, prompt_len):
    return torch.cat([
        _branch(hidden_states, attention_mask, 12, "response", "max", 16, prompt_len),
        _branch(hidden_states, attention_mask, 19, "response", "mean", 64, prompt_len),
        _branch(hidden_states, attention_mask, 15, "response_no_last", "mean", 64, prompt_len),
    ], dim=0).float()


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    return _aggregate(hidden_states, attention_mask, _next_prompt_len())


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    return _geometry(hidden_states, attention_mask, _next_prompt_len())


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    prompt_len = _next_prompt_len()
    agg_features = _aggregate(hidden_states, attention_mask, prompt_len)  # (feature_dim,)

    if use_geometric:
        geo_features = _geometry(hidden_states, attention_mask, prompt_len)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
