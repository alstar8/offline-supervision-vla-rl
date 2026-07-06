from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor


def logits_to_probs(logits: Tensor, dim: int = -1) -> Tensor:
    return torch.softmax(logits, dim=dim)


def _validate_edges(edges: Tensor) -> None:
    if edges.ndim != 1:
        raise ValueError('edges must be 1D of shape [K+1].')
    if not torch.all(edges[1:] > edges[:-1]):
        raise ValueError('edges must be strictly increasing.')


def _validate_probs(probs: Tensor, K: int) -> None:
    if probs.ndim != 1:
        raise ValueError('probs must be 1D of shape [K].')
    if probs.shape[0] != K:
        raise ValueError(f'probs has length {probs.shape[0]}, expected {K}.')
    s = probs.sum()
    if not torch.isfinite(s):
        raise ValueError('probs contains non-finite values.')
    if not torch.isclose(s, torch.tensor(1.0, device=probs.device, dtype=probs.dtype), atol=1e-4):
        raise ValueError(f'probs must sum to 1 (got {float(s)}).')


def _unique_sorted_cat(a: Tensor, b: Tensor) -> Tensor:
    x = torch.cat([a, b], dim=0)
    x = torch.unique(x)
    x, _ = torch.sort(x)
    return x


def rebin_mass_1d(probs: Tensor, edges: Tensor, new_edges: Tensor) -> Tensor:
    """
    Rebins a 1D piecewise-constant density represented by (probs over bins, edges)
    onto a new partition given by new_edges, returning probability mass per new bin.

    This is exact mass transfer via interval overlap.
    """
    _validate_edges(edges)
    _validate_edges(new_edges)

    K = edges.shape[0] - 1
    _validate_probs(probs, K)

    device = probs.device
    dtype = probs.dtype

    widths = edges[1:] - edges[:-1]
    if torch.any(widths <= 0):
        raise ValueError('Invalid bin widths.')

    # For each new bin [a, b), accumulate overlaps with old bins [l_i, r_i)
    a = new_edges[:-1]
    b = new_edges[1:]
    new_K = a.shape[0]
    mass = torch.zeros((new_K,), device=device, dtype=dtype)

    # O(K * new_K) draft; for speed you can two-pointer this since edges are sorted.
    l = edges[:-1]
    r = edges[1:]

    for j in range(new_K):
        left = a[j]
        right = b[j]

        overlap_left = torch.maximum(l, left)
        overlap_right = torch.minimum(r, right)
        overlap = torch.clamp(overlap_right - overlap_left, min=0.0)

        # Old bin i contributes: probs[i] * overlap / width[i]
        mass[j] = torch.sum(probs * (overlap / widths))

    # Numerical cleanup
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
    """
    KL(P || Q) where P and Q are 1D action distributions induced by (probs, edges).
    Computed exactly by integrating over the union partition of edges.

    Returns a scalar Tensor.
    """
    _validate_edges(edges_p)
    _validate_edges(edges_q)

    union_edges = _unique_sorted_cat(edges_p, edges_q)

    p_mass = rebin_mass_1d(probs_p, edges_p, union_edges)
    q_mass = rebin_mass_1d(probs_q, edges_q, union_edges)

    p_mass = torch.clamp(p_mass, min=eps)
    q_mass = torch.clamp(q_mass, min=eps)

    # For piecewise-constant densities on a shared partition, widths cancel:
    # KL = sum_j p_mass[j] * (log p_mass[j] - log q_mass[j])
    return torch.sum(p_mass * (torch.log(p_mass) - torch.log(q_mass)))


def kl_factorized_nd(
    probs_p_list: Sequence[Tensor],
    edges_p_list: Sequence[Tensor],
    probs_q_list: Sequence[Tensor],
    edges_q_list: Sequence[Tensor],
    eps: float = 1e-12,
) -> Tensor:
    """
    If your decoder factorizes over action dimensions, KL over the joint distribution
    equals sum of per-dim KLs.
    """
    if not (len(probs_p_list) == len(edges_p_list) == len(probs_q_list) == len(edges_q_list)):
        raise ValueError('All input lists must have the same length.')

    kls: List[Tensor] = []
    for probs_p, edges_p, probs_q, edges_q in zip(probs_p_list, edges_p_list, probs_q_list, edges_q_list):
        kls.append(kl_piecewise_1d(probs_p, edges_p, probs_q, edges_q, eps=eps))

    return torch.stack(kls).sum()


def sample_from_piecewise_1d(probs: Tensor, edges: Tensor, n: int, g: Optional[torch.Generator] = None) -> Tensor:
    _validate_edges(edges)
    K = edges.shape[0] - 1
    _validate_probs(probs, K)

    idx = torch.multinomial(probs, num_samples=n, replacement=True, generator=g)
    left = edges[idx]
    right = edges[idx + 1]
    u = torch.rand((n,), device=probs.device, dtype=probs.dtype, generator=g)
    return left + u * (right - left)


def logpdf_piecewise_1d(x: Tensor, probs: Tensor, edges: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Returns log p(x) where p is piecewise-constant on bins defined by edges with mass probs.
    """
    _validate_edges(edges)
    K = edges.shape[0] - 1
    _validate_probs(probs, K)

    widths = edges[1:] - edges[:-1]
    dens = torch.clamp(probs / widths, min=eps)

    # Bucketize: i such that edges[i] <= x < edges[i+1]
    idx = torch.bucketize(x, edges, right=False) - 1
    idx = torch.clamp(idx, min=0, max=K - 1)

    return torch.log(dens[idx])


def kl_joint_mc_factorized(
    probs_p_list: Sequence[Tensor],
    edges_p_list: Sequence[Tensor],
    probs_q_list: Sequence[Tensor],
    edges_q_list: Sequence[Tensor],
    n: int = 4096,
    eps: float = 1e-12,
    seed: int = 0,
) -> Tensor:
    """
    MC estimate of KL(P||Q) over joint action a = (a1..ad) assuming factorized sampling.
    Useful if you want a sample-based check. If you truly have a non-factorized joint,
    you need its joint sampler / evaluator.
    """
    if not (len(probs_p_list) == len(edges_p_list) == len(probs_q_list) == len(edges_q_list)):
        raise ValueError('All input lists must have the same length.')

    g = torch.Generator(device=probs_p_list[0].device)
    g.manual_seed(seed)

    logp = torch.zeros((n,), device=probs_p_list[0].device, dtype=probs_p_list[0].dtype)
    logq = torch.zeros_like(logp)

    for probs_p, edges_p, probs_q, edges_q in zip(probs_p_list, edges_p_list, probs_q_list, edges_q_list):
        x = sample_from_piecewise_1d(probs_p, edges_p, n=n, g=g)
        logp = logp + logpdf_piecewise_1d(x, probs_p, edges_p, eps=eps)
        logq = logq + logpdf_piecewise_1d(x, probs_q, edges_q, eps=eps)

    return (logp - logq).mean()
