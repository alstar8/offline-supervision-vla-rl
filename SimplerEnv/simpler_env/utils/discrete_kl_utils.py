from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def _validate_edges(edges: Tensor) -> None:
    if edges.ndim != 1:
        raise ValueError("edges must be 1D of shape [K+1].")
    if not torch.all(edges[1:] > edges[:-1]):
        raise ValueError("edges must be strictly increasing.")


def _validate_probs(probs: Tensor, k: int) -> None:
    if probs.ndim != 1:
        raise ValueError("probs must be 1D of shape [K].")
    if probs.shape[0] != k:
        raise ValueError(f"probs has length {probs.shape[0]}, expected {k}.")
    s = probs.sum()
    if not torch.isfinite(s):
        raise ValueError("probs contains non-finite values.")
    if not torch.isclose(
        s,
        torch.tensor(1.0, device=probs.device, dtype=probs.dtype),
        atol=1e-4,
    ):
        raise ValueError(f"probs must sum to 1 (got {float(s)}).")


def _unique_sorted_cat(a: Tensor, b: Tensor) -> Tensor:
    x = torch.cat([a, b], dim=0)
    x = torch.unique(x)
    x, _ = torch.sort(x)
    return x


def rebin_mass_1d(probs: Tensor, edges: Tensor, new_edges: Tensor) -> Tensor:
    """
    Rebins a 1D piecewise-constant density (probs over bins, edges)
    onto a new partition given by new_edges, returning probability mass per new bin.
    """
    _validate_edges(edges)
    _validate_edges(new_edges)

    k = edges.shape[0] - 1
    _validate_probs(probs, k)

    widths = edges[1:] - edges[:-1]
    if torch.any(widths <= 0):
        raise ValueError("Invalid bin widths.")

    l = edges[:-1].unsqueeze(1)  # [K, 1]
    r = edges[1:].unsqueeze(1)   # [K, 1]
    a = new_edges[:-1].unsqueeze(0)  # [1, Knew]
    b = new_edges[1:].unsqueeze(0)   # [1, Knew]

    overlap_left = torch.maximum(l, a)
    overlap_right = torch.minimum(r, b)
    overlap = torch.clamp(overlap_right - overlap_left, min=0.0)

    mass = torch.sum(probs.unsqueeze(1) * (overlap / widths.unsqueeze(1)), dim=0)
    mass = torch.clamp(mass, min=0.0)
    mass = mass / mass.sum()
    return mass


def kl_piecewise_1d(
    probs_p: Tensor,
    edges_p: Tensor,
    probs_q: Tensor,
    edges_q: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    _validate_edges(edges_p)
    _validate_edges(edges_q)

    union_edges = _unique_sorted_cat(edges_p, edges_q)

    p_mass = rebin_mass_1d(probs_p, edges_p, union_edges)
    q_mass = rebin_mass_1d(probs_q, edges_q, union_edges)

    p_mass = torch.clamp(p_mass, min=eps)
    q_mass = torch.clamp(q_mass, min=eps)

    return torch.sum(p_mass * (torch.log(p_mass) - torch.log(q_mass)))


def kl_factorized_from_logits(
    logits_p: Tensor,
    edges_p: Tensor,
    logits_q: Tensor,
    edges_q: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """
    Exact KL between factorized distributions over continuous actions when each
    dimension is discretized into bins with potentially different edges.

    logits_*: [B, D, K]
    edges_*: [D, K+1]
    Returns: scalar KL averaged over batch.
    """
    if logits_p.ndim != 3 or logits_q.ndim != 3:
        raise ValueError("logits must be [B, D, K].")
    if edges_p.ndim != 2 or edges_q.ndim != 2:
        raise ValueError("edges must be [D, K+1].")
    if logits_p.shape != logits_q.shape:
        raise ValueError("logits_p and logits_q must have the same shape.")
    if edges_p.shape[0] != logits_p.shape[1] or edges_q.shape[0] != logits_p.shape[1]:
        raise ValueError("edges must have one entry per action dimension.")
    if edges_p.shape[1] != logits_p.shape[2] + 1 or edges_q.shape[1] != logits_p.shape[2] + 1:
        raise ValueError("edges must be K+1 where K is logits last dim.")

    probs_p = torch.softmax(logits_p, dim=-1)
    probs_q = torch.softmax(logits_q, dim=-1)

    batch_size, dims, _ = logits_p.shape
    kl_total = torch.zeros((batch_size,), device=logits_p.device, dtype=logits_p.dtype)

    for b in range(batch_size):
        kl_bd = []
        for d in range(dims):
            kl_bd.append(
                kl_piecewise_1d(
                    probs_p[b, d], edges_p[d], probs_q[b, d], edges_q[d], eps=eps
                )
            )
        kl_total[b] = torch.stack(kl_bd).sum()

    return kl_total.mean()


def build_action_edges_from_stats(
    action_stats: dict,
    n_bins: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Build continuous bin edges per action dimension based on dataset statistics.

    Returns: edges [D, K+1]
    """
    q01 = torch.tensor(action_stats["q01"], device=device, dtype=dtype)
    q99 = torch.tensor(action_stats["q99"], device=device, dtype=dtype)
    mask = action_stats.get("mask", None)
    if mask is None:
        mask_t = torch.ones_like(q01, dtype=torch.bool)
    else:
        mask_t = torch.tensor(mask, device=device, dtype=torch.bool)

    edges_norm = torch.linspace(-1.0, 1.0, n_bins + 1, device=device, dtype=dtype)
    eps = torch.finfo(dtype).eps
    edges = []
    for i in range(q01.shape[0]):
        if not bool(mask_t[i].item()):
            edges_i = edges_norm
        else:
            low = torch.minimum(q01[i], q99[i])
            high = torch.maximum(q01[i], q99[i])
            span = high - low
            if span < 1e-6:
                # degenerate stats; fall back to normalized grid
                edges_i = edges_norm
            else:
                edges_i = 0.5 * (edges_norm + 1.0) * span + low
        # ensure strictly increasing by adding a tiny monotone jitter
        jitter = torch.linspace(0, eps * 10, n_bins + 1, device=device, dtype=dtype)
        edges_i = edges_i + jitter
        edges.append(edges_i)
    return torch.stack(edges, dim=0)
