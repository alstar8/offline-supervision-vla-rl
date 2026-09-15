"""Weighted action-token CE used by SFT and offline DataBC."""

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F


def parse_action_dim_loss_weights(spec: Optional[str]) -> Optional[List[float]]:
    """Parse '3,3,1.5,0,0,0,1' or '3:3:1.5:0:0:0:1' into 7 floats. Empty -> None."""
    if spec is None:
        return None
    text = str(spec).strip()
    if not text:
        return None
    sep = ":" if ":" in text and "," not in text else ","
    parts = [float(x.strip()) for x in text.split(sep) if x.strip() != ""]
    if len(parts) != 7:
        raise ValueError(
            f"action_dim_loss_weights must have 7 values (x,y,z,rx,ry,rz,gripper); got {len(parts)} from {spec!r}"
        )
    return parts


def weighted_action_token_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_visual_tokens: int,
    action_token_begin_idx: int,
    dim_weights: Sequence[float],
    ignore_index: int = -100,
) -> torch.Tensor:
    """CE over action tokens only, weighted by action dimension (x,y,z,rx,ry,rz,gripper).

    Matches OpenVLA metric slicing: logits[:, num_visual_tokens:-1] vs labels[:, 1:].
    Zero-weight dims (typically rotation) are dropped from the denominator.
    """
    shift_logits = logits[:, num_visual_tokens:-1, :].contiguous()
    shift_labels = labels[:, 1:].to(device=shift_logits.device).contiguous()
    vocab = shift_logits.size(-1)
    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, vocab).float(),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).view(shift_labels.shape)
    action_mask = shift_labels > action_token_begin_idx
    valid = action_mask & (shift_labels != ignore_index)
    weights = torch.tensor(list(dim_weights), device=token_loss.device, dtype=token_loss.dtype)
    dim_index = (valid.to(torch.long).cumsum(dim=1) - 1).clamp(min=0, max=max(len(dim_weights) - 1, 0))
    token_w = torch.where(valid, weights[dim_index], torch.zeros_like(token_loss))
    denom = token_w.sum().clamp_min(1.0)
    return (token_loss * token_w).sum() / denom
