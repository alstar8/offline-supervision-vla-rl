from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class RolloutBatch:
    prompt_ids: Tensor          # [B, T_prompt]
    gen_ids: Tensor             # [B, T_gen]
    attention_mask: Tensor      # [B, T_prompt + T_gen]

    old_logprobs: Tensor        # [B, T_gen] log p_pi_old(a_t | s_t)
    ref_logprobs: Tensor        # [B, T_gen] log p_ref(a_t | s_t)

    rewards: Tensor             # [B, T_gen] SHAPED: rm_reward - beta * (logp - logp_ref)
    advantages: Tensor          # [B, T_gen]
    returns: Tensor             # [B, T_gen]


def token_logprobs(logits: Tensor, token_ids: Tensor) -> Tensor:
    # logits: [B, T, V], token_ids: [B, T]
    logp = F.log_softmax(logits, dim=-1)
    return torch.gather(logp, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def collect_rollout_with_kl_penalty(
    policy,           # trainable causal LM
    ref_policy,       # frozen reference LM (pretrain or SFT)
    reward_model,     # (prompt_ids, gen_ids) -> scalar reward per sample
    prompt_ids: Tensor,
    prompt_attn: Tensor,
    beta_kl: float,
    gen_kwargs: Dict,
) -> RolloutBatch:
    """
    Collects one PPO rollout and *builds the KL-shaped reward* used in RLHF:

        r'_t = r_t - beta_kl * ( log p_pi(a_t|s_t) - log p_ref(a_t|s_t) )

    Notes:
    - The term (log p_pi - log p_ref) is a per-token *sample estimate* of KL.
    - Penalizing it keeps the policy close to the frozen reference distribution.
    """
    device = prompt_ids.device

    # 1) Sample from current policy
    gen = policy.generate(
        input_ids=prompt_ids,
        attention_mask=prompt_attn,
        return_dict_in_generate=True,
        **gen_kwargs,
    )
    full_ids = gen.sequences                               # [B, T_prompt + T_gen]
    gen_ids = full_ids[:, prompt_ids.size(1):]             # [B, T_gen]

    # Use a simple full attention mask for the forward pass (adjust for padding as needed)
    attention_mask = torch.ones_like(full_ids, device=device)

    # 2) Get token logprobs under (a) current policy and (b) frozen reference
    pi_logits = policy(full_ids, attention_mask=attention_mask).logits
    ref_logits = ref_policy(full_ids, attention_mask=attention_mask).logits

    # Next-token alignment: token at position t uses logits from position t-1
    start = prompt_ids.size(1)
    pi_logits_gen = pi_logits[:, start - 1:-1, :]          # [B, T_gen, V]
    ref_logits_gen = ref_logits[:, start - 1:-1, :]        # [B, T_gen, V]

    old_logprobs = token_logprobs(pi_logits_gen, gen_ids)  # [B, T_gen]
    ref_logprobs = token_logprobs(ref_logits_gen, gen_ids) # [B, T_gen]

    # 3) KL “trick”: compute the per-token logprob difference.
    # This quantity is what becomes the KL penalty inside the reward.
    #
    #   kl_token ≈ log p_pi(a_t|s_t) - log p_ref(a_t|s_t)
    #
    # If this grows, the policy is drifting away from the reference.
    kl_per_token = old_logprobs - ref_logprobs             # [B, T_gen]

    # 4) Reward model gives a scalar per completion; common pattern: put it on the final token
    rm_score = reward_model(prompt_ids, gen_ids).to(device)  # [B]
    rm_reward = torch.zeros_like(old_logprobs)
    rm_reward[:, -1] = rm_score

    # 5) The key RLHF step: *shaped reward* includes the KL penalty.
    #
    #   shaped_reward_t = rm_reward_t - beta_kl * kl_per_token_t
    #
    # This makes PPO act like: "maximize RM reward but pay a cost for deviating from ref".
    shaped_rewards = rm_reward - beta_kl * kl_per_token

    # 6) Standard PPO: compute (advantages, returns) from shaped rewards and a value baseline
    # (GAE implementation is intentionally omitted here.)
    advantages, returns = compute_gae(shaped_rewards)       # user-provided standard PPO component

    return RolloutBatch(
        prompt_ids=prompt_ids,
        gen_ids=gen_ids,
        attention_mask=attention_mask,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
        rewards=shaped_rewards,
        advantages=advantages,
        returns=returns,
    )


def ppo_update_step(
    policy,
    value_fn,        # returns per-token values for generated positions
    optimizer,
    batch: RolloutBatch,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
) -> Dict[str, float]:
    """
    Standard PPO update on the *KL-shaped reward* rollouts.
    The KL penalty is already baked into batch.rewards via rollout collection.
    """
    full_ids = torch.cat([batch.prompt_ids, batch.gen_ids], dim=1)
    logits = policy(full_ids, attention_mask=batch.attention_mask).logits

    start = batch.prompt_ids.size(1)
    logits_gen = logits[:, start - 1:-1, :]                  # [B, T_gen, V]
    new_logprobs = token_logprobs(logits_gen, batch.gen_ids)  # [B, T_gen]

    # PPO ratio
    log_ratio = new_logprobs - batch.old_logprobs
    ratio = torch.exp(log_ratio)

    # Advantage normalization is common (optional)
    adv = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

    # Clipped policy gradient loss
    pg1 = ratio * adv
    pg2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -torch.mean(torch.minimum(pg1, pg2))

    # Value loss
    values = value_fn(full_ids, attention_mask=batch.attention_mask)[:, start:]  # [B, T_gen]
    value_loss = F.mse_loss(values, batch.returns)

    # Entropy bonus (optional)
    logp = F.log_softmax(logits_gen, dim=-1)
    p = torch.exp(logp)
    entropy = -torch.mean(torch.sum(p * logp, dim=-1))
    ent_loss = -ent_coef * entropy

    loss = policy_loss + vf_coef * value_loss + ent_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Diagnostics: “approx KL” on sampled actions vs ref (not exact KL over distributions)
    with torch.no_grad():
        approx_kl = torch.mean(new_logprobs - batch.ref_logprobs).item()

    return {
        'loss': float(loss.item()),
        'policy_loss': float(policy_loss.item()),
        'value_loss': float(value_loss.item()),
        'entropy': float(entropy.item()),
        'approx_kl': float(approx_kl),
    }


# --- You provide these standard PPO bits elsewhere ---
def compute_gae(rewards: Tensor) -> Tuple[Tensor, Tensor]:
    raise NotImplementedError
