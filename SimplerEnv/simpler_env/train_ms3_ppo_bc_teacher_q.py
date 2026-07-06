import os
import pprint
import random
import gc
import signal
import math
from collections import defaultdict
import time
from pathlib import Path
from typing import Annotated, Optional, List
import torch
from torch import nn
import numpy as np
import tyro
import wandb
from dataclasses import dataclass
import yaml
from tqdm import tqdm
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video
from simpler_env.policies.openvla.openvla_train import huber_loss
from simpler_env.utils.discrete_kl_utils import rebin_mass_1d

from simpler_env.env.simpler_wrapper import SimlerWrapper
from simpler_env.utils.replay_buffer import SeparatedReplayBuffer
from simpler_env.utils.wandb_utils import init_wandb_with_online_fallback

signal.signal(signal.SIGINT, signal.default_int_handler)  # allow ctrl+c KeyboardInterrupt
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PutCarrotOnPlateInScene-v1"
    """The environment ID of the task you want to simulate. Can be one of
    PutCarrotOnPlateInScene-v1, PutSpoonOnTableClothInScene-v1, StackGreenCubeOnYellowCubeBakedTexInScene-v1, PutEggplantInBasketScene-v1"""

    """Number of environments to run. With more than 1 environment the environment will use the GPU backend 
    which runs faster enabling faster large-scale evaluations. Note that the overall behavior of the simulation
    will be slightly different between CPU and GPU backends."""

    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """Seed the model and environment. Default seed is 0"""

    name: str = "PPO-bc-teacher-q"

    # env
    num_envs: int = 64
    episode_len: int = 80
    use_same_init: bool = False
    rollouts_per_update: int = 1
    use_default_task: bool = False

    steps_max: int = 2000000
    steps_vh: int = 0  # episodes
    interval_eval: int = 10
    interval_save: int = 10
    eval_save_video: bool = True

    # buffer
    buffer_inferbatch: int = 32
    buffer_minibatch: int = 8
    buffer_gamma: float = 0.99
    buffer_lambda: float = 0.95
    store_rollouts_on_cpu: bool = False

    # vla
    vla_path: str = "openvla/openvla-7b"
    vla_unnorm_key: str = "bridge_orig"
    vla_load_path: str = ""
    vla_lora_rank: int = 32
    vla_lora_alpha: int = 0
    vla_lora_target_modules: Optional[List[str]] = None
    vla_device_map: str = ""
    vla_max_memory_gb: int = 0
    vla_gradient_checkpointing: bool = False

    vla_lr: float = 1e-4
    vla_vhlr: float = 3e-3
    vla_optim_beta1: float = 0.9
    vla_optim_beta2: float = 0.999
    vla_temperature: float = 1.0
    vla_temperature_eval: float = 0.6

    # ppo & grpo
    alg_name: str = "ppo"  # ppo, grpo
    alg_grpo_fix: bool = True
    alg_gradient_accum: int = 20
    alg_ppo_epoch: int = 1
    ppo_clip: float = 0.2
    alg_entropy_coef: float = 0.0
    freeze_actor_updates: int = 0
    kl_to_ref_enabled: bool = False
    kl_to_ref_steps: int = 500000
    kl_to_ref_coef: float = 0.008
    kl_to_ref_path: str = ""
    kl_to_ref_unnorm_key: str = ""
    bc_to_ref_enabled: bool = False
    bc_to_ref_coef: float = 0.6
    bc_to_ref_hold_steps: int = 100000
    bc_to_ref_decay_steps: int = 300000
    bc_to_ref_deterministic: bool = True
    bc_to_ref_q_adv_top_frac: float = 0.5
    bc_to_ref_q_adv_margin: float = 0.0

    # q head
    q_enabled: bool = False
    q_coef: float = 1.0
    q_huber_delta: float = 10.0
    q_detach_backbone: bool = True
    q_target_enabled: bool = False
    q_target_tau: float = 0.01
    q_adv_use_target: bool = False

    # other
    wandb: bool = True
    only_render: bool = False
    render_info: bool = False
    update_mem_check: bool = False



class OpenVLAPPOBCTeacher:
    def __init__(self, all_args, policy):
        self.args = all_args
        self.policy = policy
        self.ppo_clip = self.args.ppo_clip
        self.ppo_grad_norm = 10.0
        self.ppo_entropy_coef = self.args.alg_entropy_coef
        self.ppo_huber_delta = 10.0
        self.tpdv = self.policy.tpdv
        self.tpdv_vn = self.policy.tpdv_vn
        self.freeze_actor_updates = False
        self.global_step = 0

    @staticmethod
    def _as_tensor(value, device, dtype=None):
        if torch.is_tensor(value):
            if dtype is None:
                return value.to(device=device)
            return value.to(device=device, dtype=dtype)
        if dtype is None:
            return torch.as_tensor(value, device=device)
        return torch.as_tensor(value, device=device, dtype=dtype)

    def get_bc_to_ref_coef(self, step: Optional[int] = None) -> float:
        if not self.args.bc_to_ref_enabled:
            return 0.0
        if step is None:
            step = self.global_step
        hold_steps = max(0, int(self.args.bc_to_ref_hold_steps))
        decay_steps = max(hold_steps + 1, int(self.args.bc_to_ref_decay_steps))
        start_coef = float(self.args.bc_to_ref_coef)
        if step <= hold_steps:
            return start_coef
        if step >= decay_steps:
            return 0.0
        frac = (step - hold_steps) / float(decay_steps - hold_steps)
        return start_coef * (1.0 - frac)

    def get_bc_adv_top_frac(self) -> float:
        return float(np.clip(self.args.bc_to_ref_q_adv_top_frac, 0.0, 1.0))

    @staticmethod
    def _select_top_mask(values: torch.Tensor, keep_frac: float) -> torch.Tensor:
        values = values.reshape(-1)
        count = values.shape[0]
        if count == 0:
            return torch.zeros_like(values, dtype=torch.bool)
        keep_frac = float(np.clip(keep_frac, 0.0, 1.0))
        if keep_frac <= 0.0:
            return torch.zeros_like(values, dtype=torch.bool)
        if keep_frac >= 1.0:
            return torch.ones_like(values, dtype=torch.bool)
        k = max(1, int(math.ceil(count * keep_frac)))
        top_idx = torch.topk(values, k=k, largest=True, sorted=False).indices
        mask = torch.zeros_like(values, dtype=torch.bool)
        mask[top_idx] = True
        return mask

    @staticmethod
    def _collapse_execution_bins(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Match OpenVLA execution semantics:
        the top two token bins map to the same executed action value.
        Collapse to execution bins and return log-probabilities.
        """
        if logits.ndim != 3:
            raise ValueError("logits must be [B, D, K].")
        if logits.shape[-1] < 2:
            raise ValueError("Need at least 2 bins to collapse execution bins.")
        probs = torch.softmax(logits, dim=-1)
        probs = torch.cat([probs[..., :-2], probs[..., -2:].sum(dim=-1, keepdim=True)], dim=-1)
        return torch.log(torch.clamp(probs, min=eps))

    @staticmethod
    def _rebin_teacher_probs_to_student(
        teacher_probs: torch.Tensor,
        teacher_edges: torch.Tensor,
        student_edges: torch.Tensor,
    ) -> torch.Tensor:
        if teacher_probs.ndim != 3:
            raise ValueError("teacher_probs must be [B, D, K].")
        batch_size, dims, _ = teacher_probs.shape
        out = torch.zeros_like(teacher_probs)
        same_edges = torch.allclose(teacher_edges, student_edges, atol=1e-7, rtol=1e-6)
        if same_edges:
            return teacher_probs
        for b in range(batch_size):
            for d in range(dims):
                out[b, d] = rebin_mass_1d(
                    teacher_probs[b, d],
                    teacher_edges[d],
                    student_edges[d],
                )
        return out

    def _feed_forward_generator_with_teacher(self, buffer, teacher_actions):
        episode_length, n_rollout_threads = buffer.rewards.shape[:2]
        batch_size = episode_length * n_rollout_threads
        if buffer.buffer_minibatch < 0:
            mini_batch_size = batch_size
            num_mini_batch = 1
        else:
            assert batch_size % buffer.buffer_minibatch == 0
            mini_batch_size = buffer.buffer_minibatch
            num_mini_batch = batch_size // mini_batch_size

        if buffer.store_rollouts_on_cpu:
            rand = torch.randperm(batch_size).numpy()
            sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

            obs = buffer.obs[:-1].reshape(-1, *buffer.obs.shape[2:])
            actions = buffer.actions.reshape(-1, buffer.actions.shape[-1])
            value_preds = buffer.value_preds[:-1].reshape(-1, 1)
            returns = buffer.returns.reshape(-1, 1)
            masks = buffer.masks[:-1].reshape(-1, 1)
            action_logits = buffer.action_log_probs.reshape(-1, buffer.action_log_probs.shape[-1])
            advantages = buffer.advantages.reshape(-1, 1)
            teacher = teacher_actions.reshape(-1, teacher_actions.shape[-1])

            for indices in sampler:
                instruct_indices = indices % n_rollout_threads
                instruct_batch = [buffer.instruction[i] for i in instruct_indices]
                yield (
                    obs[indices],
                    instruct_batch,
                    actions[indices],
                    value_preds[indices],
                    returns[indices],
                    masks[indices],
                    action_logits[indices],
                    advantages[indices],
                    teacher[indices],
                )
        else:
            rand = torch.randperm(batch_size, device=buffer.device)
            sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

            obs = buffer.obs[:-1].reshape(-1, *buffer.obs.shape[2:])
            actions = buffer.actions.reshape(-1, buffer.actions.shape[-1])
            value_preds = buffer.value_preds[:-1].reshape(-1, 1)
            returns = buffer.returns.reshape(-1, 1)
            masks = buffer.masks[:-1].reshape(-1, 1)
            action_logits = buffer.action_log_probs.reshape(-1, buffer.action_log_probs.shape[-1])
            advantages = buffer.advantages.reshape(-1, 1)
            teacher = teacher_actions.reshape(-1, teacher_actions.shape[-1])

            for indices in sampler:
                instruct_indices = (indices % n_rollout_threads).to("cpu").tolist()
                instruct_batch = [buffer.instruction[i] for i in instruct_indices]
                yield (
                    obs[indices],
                    instruct_batch,
                    actions[indices],
                    value_preds[indices],
                    returns[indices],
                    masks[indices],
                    action_logits[indices],
                    advantages[indices],
                    teacher[indices],
                )

    def train_ppo_step(self, idx, total, batch):
        (
            obs_image,
            instruct,
            actions,
            value_preds,
            returns,
            masks,
            old_logprob,
            advantages,
            teacher_actions,
        ) = batch

        obs_image = self._as_tensor(obs_image, self.tpdv["device"])
        obs = dict(image=obs_image, task_description=instruct)
        actions = self._as_tensor(actions, self.tpdv["device"], dtype=torch.int32)
        teacher_actions = self._as_tensor(teacher_actions, self.tpdv["device"], dtype=torch.int32)
        value_preds = self._as_tensor(value_preds, self.tpdv["device"], dtype=self.tpdv["dtype"])
        returns = self._as_tensor(returns, self.tpdv_vn["device"], dtype=self.tpdv_vn["dtype"])
        old_logprob = self._as_tensor(old_logprob, self.tpdv["device"], dtype=self.tpdv["dtype"])
        advantages = self._as_tensor(advantages, self.tpdv["device"], dtype=self.tpdv["dtype"])
        returns_norm = returns.to(**self.tpdv)

        q_values_student = None
        q_loss = None
        if self.policy.has_q_head():
            logprob, entropy, values, q_values_student = self.policy.evaluate_actions_with_q(
                obs,
                actions,
                use_target_q=False,
                detach_backbone_for_q=self.args.q_detach_backbone,
            )
            q_error = returns_norm - q_values_student
            q_loss = huber_loss(q_error, self.args.q_huber_delta).mean()
        else:
            logprob, entropy, values = self.policy.evaluate_actions(obs, actions)
        ratio = torch.exp(logprob - old_logprob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()

        bc_coef = self.get_bc_to_ref_coef()
        bc_loss = None
        bc_adv_keep_ratio = 0.0
        bc_adv_threshold = 0.0
        bc_adv_delta_mean = 0.0
        if bc_coef > 0.0:
            student_key = self.args.vla_unnorm_key
            teacher_key = self.args.kl_to_ref_unnorm_key or self.args.vla_unnorm_key
            student_logits = self.policy.get_action_logits(obs, teacher_actions, unnorm_key=student_key)
            teacher_logits = self.policy.get_teacher_action_logits(obs, teacher_actions, unnorm_key=teacher_key)

            student_log_probs_exec = self._collapse_execution_bins(student_logits.float())
            teacher_log_probs_exec = self._collapse_execution_bins(teacher_logits.float())
            teacher_probs_exec = teacher_log_probs_exec.exp()

            effective_bins = student_log_probs_exec.shape[-1]
            student_edges = self.policy.get_action_edges(student_key, n_bins=effective_bins).float()
            teacher_edges = self.policy.get_action_edges(teacher_key, n_bins=effective_bins).float()
            teacher_probs_on_student = self._rebin_teacher_probs_to_student(
                teacher_probs_exec,
                teacher_edges,
                student_edges,
            )
            bc_loss_per_sample = -(teacher_probs_on_student * student_log_probs_exec).sum(dim=-1).mean(dim=-1)

            if self.policy.has_q_head():
                with torch.no_grad():
                    q_student = q_values_student.reshape(-1)
                    q_teacher = self.policy.get_q(
                        obs,
                        teacher_actions,
                        use_target_q=self.args.q_adv_use_target,
                        detach_backbone_for_q=True,
                    ).reshape(-1)
                    q_delta = q_teacher - q_student
                    bc_adv_delta_mean = q_delta.mean().item()
                    better_mask = q_delta > float(self.args.bc_to_ref_q_adv_margin)
                    keep_mask = torch.zeros_like(better_mask)
                    if better_mask.any():
                        keep_frac = self.get_bc_adv_top_frac()
                        better_indices = better_mask.nonzero(as_tuple=True)[0]
                        better_delta = q_delta[better_indices]
                        top_mask_local = self._select_top_mask(better_delta, keep_frac)
                        selected_indices = better_indices[top_mask_local]
                        keep_mask[selected_indices] = True
                        bc_adv_threshold = better_delta[top_mask_local].min().item() if top_mask_local.any() else 0.0
                    bc_adv_keep_ratio = keep_mask.to(torch.float32).mean().item()

                if keep_mask.any():
                    bc_loss = bc_loss_per_sample[keep_mask].mean()
                    policy_loss = policy_loss + bc_coef * bc_loss
                else:
                    bc_loss = bc_loss_per_sample.mean().detach() * 0.0
            else:
                bc_loss = bc_loss_per_sample.mean()
                policy_loss = policy_loss + bc_coef * bc_loss

        value_pred_clipped = value_preds + (values - value_preds).clamp(-self.ppo_clip, self.ppo_clip)
        error_clipped = returns_norm - value_pred_clipped
        error_original = returns_norm - values
        value_loss_clipped = huber_loss(error_clipped, self.ppo_huber_delta)
        value_loss_original = huber_loss(error_original, self.ppo_huber_delta)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        value_clip_indicator = (value_pred_clipped - value_preds).abs() > self.ppo_clip
        value_clip_ratio = value_clip_indicator.to(**self.tpdv).mean()
        value_loss = value_loss.mean()
        entropy_loss = entropy.mean()

        if self.freeze_actor_updates:
            loss = value_loss
            if q_loss is not None:
                loss = loss + float(self.args.q_coef) * q_loss
            policy_loss = policy_loss.detach() * 0.0
            entropy_loss = entropy_loss.detach() * 0.0
            bc_loss = bc_loss.detach() * 0.0 if bc_loss is not None else None
            q_loss = q_loss.detach() * 0.0 if q_loss is not None else None
        else:
            loss = policy_loss + value_loss - self.ppo_entropy_coef * entropy_loss
            if q_loss is not None:
                loss = loss + float(self.args.q_coef) * q_loss
        loss /= self.args.alg_gradient_accum
        loss.backward()

        if idx % self.args.alg_gradient_accum == (self.args.alg_gradient_accum - 1) or idx == (total - 1):
            grad_params = self.policy.params_vla + self.policy.params_vh
            if self.policy.has_q_head():
                grad_params = grad_params + self.policy.params_q
            grad_norm = nn.utils.clip_grad_norm_(grad_params, self.ppo_grad_norm)
            self.policy.vh_optimizer.step()
            if self.policy.has_q_head() and self.policy.q_optimizer is not None:
                self.policy.q_optimizer.step()
                if self.policy.has_target_q_head():
                    self.policy.sync_target_q_head(tau=self.args.q_target_tau)
            if not self.freeze_actor_updates:
                self.policy.vla_optimizer.step()
            self.policy.vh_optimizer.zero_grad()
            if self.policy.has_q_head() and self.policy.q_optimizer is not None:
                self.policy.q_optimizer.zero_grad()
            self.policy.vla_optimizer.zero_grad()
        else:
            grad_norm = None

        info = dict(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy_loss=entropy_loss.item(),
            ratio=ratio.mean().item(),
            ratio_median=ratio.median().item(),
            ratio_2=(logprob - old_logprob).mean().exp().item(),
            value_clip_ratio=value_clip_ratio.item(),
            value_old_mean=value_preds.mean().item(),
            values_mean=values.mean().item(),
            returns_mean=returns.mean().item(),
            returns_norm_mean=returns_norm.mean().item(),
            logprob_mean=logprob.mean().item(),
            logprob_old_mean=old_logprob.mean().item(),
            bc_to_ref_coef=float(bc_coef),
            bc_to_ref_active=1.0 if bc_coef > 0.0 else 0.0,
            bc_to_ref_loss=bc_loss.item() if bc_loss is not None else 0.0,
            bc_to_ref_adv_keep_ratio=bc_adv_keep_ratio,
            bc_to_ref_adv_threshold=bc_adv_threshold,
            bc_to_ref_adv_delta_mean=bc_adv_delta_mean,
            q_enabled=1.0 if self.policy.has_q_head() else 0.0,
            q_loss=q_loss.item() if q_loss is not None else 0.0,
            q_coef=float(self.args.q_coef) if self.policy.has_q_head() else 0.0,
        )
        if grad_norm is not None:
            info["grad_norm"] = grad_norm.item()
        return info

    def train_grpo(self, buffer):
        from simpler_env.policies.openvla.openvla_train import OpenVLAPPO
        return OpenVLAPPO(self.args, self.policy).train_grpo(buffer)

    def train_ppo(self, buffer, global_step: int, teacher_actions):
        train_info = defaultdict(lambda: [])
        self.global_step = global_step
        if teacher_actions is None:
            teacher_actions = buffer.actions
        buffer.compute_returns_ppo()
        minibatch_count = buffer.get_minibatch_count()

        for _ in range(self.args.alg_ppo_epoch):
            data_generator = self._feed_forward_generator_with_teacher(buffer, teacher_actions)
            for idx, batch in tqdm(enumerate(data_generator), total=minibatch_count, desc="train"):
                info = self.train_ppo_step(idx, minibatch_count, batch)
                for key, value in info.items():
                    train_info[key].append(value)

        return {key: np.mean(value) for key, value in train_info.items()}


class Runner:
    """Orchestrate PPO/GRPO training and evaluation for SimplerEnv tasks.

    Handles seeding, logging setup, policy/algorithm wiring, environment creation,
    replay buffer management, and optional memory checks used before training.
    """
    def __init__(self, all_args: Args):
        self.args = all_args

        # alg_name
        assert self.args.alg_name in ["ppo", "grpo"]
        if self.args.bc_to_ref_enabled and not self.args.kl_to_ref_enabled:
            self.args.kl_to_ref_enabled = True

        # set seed
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        # set wandb
        init_wandb_with_online_fallback(
            config=all_args.__dict__,
            project="RLVLA",
            name=self.args.name,
            use_wandb=self.args.wandb,
        )
        self.save_dir = Path(wandb.run.dir)
        self.glob_dir = Path(wandb.run.dir) / ".." / "glob"
        self.glob_dir.mkdir(parents=True, exist_ok=True)

        yaml.dump(all_args.__dict__, open(self.glob_dir / "config.yaml", "w"))
        self._log_args()

        # policy
        from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy
        device_id = 0
        device_id_other = 1 if torch.cuda.device_count() > 1 else 0
        self.device = torch.device("cuda:" + str(device_id))
        self.policy = OpenVLAPolicy(all_args, device_id_other)
        self.alg = OpenVLAPPOBCTeacher(all_args, self.policy)
        if self.policy.has_target_q_head():
            self.policy.sync_target_q_head(tau=1.0)
        self.teacher_key = self.args.kl_to_ref_unnorm_key or self.args.vla_unnorm_key
        self.student_key = self.args.vla_unnorm_key
        self.bc_to_ref_enabled = self.args.bc_to_ref_enabled and self.policy.has_teacher()
        self.teacher_actions = None
        if self.bc_to_ref_enabled:
            student_len = self.policy.vla.get_action_dim(self.student_key)
            teacher_len = self.policy.vla.get_action_dim(self.teacher_key)
            if student_len != teacher_len:
                raise RuntimeError(
                    f"Student/teacher action dims differ ({student_len} vs {teacher_len}); "
                    "cannot do BC-to-teacher with mismatched lengths."
                )
            self.teacher_action_dim = teacher_len
            print(
                "BC-to-teacher enabled | "
                f"teacher_key={self.teacher_key} | "
                f"student_key={self.student_key} | "
                f"coef_start={self.args.bc_to_ref_coef} | "
                f"hold_steps={self.args.bc_to_ref_hold_steps} | "
                f"decay_steps={self.args.bc_to_ref_decay_steps}"
            )
        else:
            self.teacher_action_dim = 0
            print("BC-to-teacher disabled.")
        if self.policy.has_q_head():
            print(
                "Q-head enabled | "
                f"q_coef={self.args.q_coef} | "
                f"detach_backbone={self.args.q_detach_backbone} | "
                f"target_enabled={self.args.q_target_enabled} | "
                f"target_tau={self.args.q_target_tau} | "
                f"adv_top_frac={self.args.bc_to_ref_q_adv_top_frac} | "
                f"adv_margin={self.args.bc_to_ref_q_adv_margin}"
            )
        else:
            print("Q-head disabled.")

        # env
        unnorm_state = self.policy.vla.get_action_stats(self.args.vla_unnorm_key)
        self.env = SimlerWrapper(self.args, unnorm_state)

        # buffer
        self.buffer = SeparatedReplayBuffer(
            all_args,
            obs_dim=(480, 640, 3),
            act_dim=7,
            device=self.device,
        )
        minibatch_count = self.buffer.get_minibatch_count()
        print(f"Buffer minibatch count: {minibatch_count}")

        self._check_update_memory()

    def _log_args(self):
        print("Training args (full):")
        print(yaml.safe_dump(self.args.__dict__, sort_keys=True).strip())

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _format_env_metrics(env_metrics: dict) -> str:
        if not env_metrics:
            return "env: none"
        parts = [f"{k}={env_metrics[k]:.4f}" for k in sorted(env_metrics.keys())]
        return "env: " + " | ".join(parts)

    def _check_update_memory(self):
        if not self.args.update_mem_check:
            return

        batch_size = self.args.buffer_minibatch
        if batch_size <= 0:
            batch_size = (
                self.args.episode_len
                * self.args.num_envs
                * self.args.rollouts_per_update
            )

        action_dim = self.policy.vla.get_action_dim(self.args.vla_unnorm_key)
        device = self.policy.tpdv["device"]
        dtype = self.policy.tpdv["dtype"]
        vn_dtype = self.policy.tpdv_vn["dtype"]

        obs_image = torch.randint(
            0, 256, (batch_size, 480, 640, 3), dtype=torch.uint8
        )
        instruction = ["memory check"] * batch_size
        actions = torch.randint(
            32000 - 256, 32000, (batch_size, action_dim), dtype=torch.int32
        )
        value_preds = torch.zeros((batch_size, 1), dtype=torch.float32)
        returns = torch.zeros((batch_size, 1), dtype=torch.float32)
        advantages = torch.zeros((batch_size, 1), dtype=torch.float32)
        old_logprob = torch.zeros((batch_size, action_dim), dtype=torch.float32)

        try:
            obs = dict(image=obs_image.to(device), task_description=instruction)
            actions = actions.to(device)
            value_preds = value_preds.to(device=device, dtype=dtype)
            returns = returns.to(device=device, dtype=vn_dtype)
            advantages = advantages.to(device=device, dtype=dtype)
            old_logprob = old_logprob.to(device=device, dtype=dtype)
            returns_norm = returns.to(device=device, dtype=dtype)

            if self.policy.has_q_head():
                logprob, entropy, values, q_values = self.policy.evaluate_actions_with_q(
                    obs,
                    actions,
                    use_target_q=False,
                    detach_backbone_for_q=self.args.q_detach_backbone,
                )
            else:
                logprob, entropy, values = self.policy.evaluate_actions(obs, actions)

            ratio = torch.exp(logprob - old_logprob)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.alg.ppo_clip, 1 + self.alg.ppo_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()

            value_pred_clipped = value_preds + (values - value_preds).clamp(
                -self.alg.ppo_clip, self.alg.ppo_clip
            )
            error_clipped = returns_norm - value_pred_clipped
            error_original = returns_norm - values
            value_loss_clipped = huber_loss(error_clipped, self.alg.ppo_huber_delta)
            value_loss_original = huber_loss(error_original, self.alg.ppo_huber_delta)
            value_loss = torch.max(value_loss_original, value_loss_clipped).mean()

            entropy_loss = entropy.mean()

            loss = policy_loss + value_loss - self.alg.ppo_entropy_coef * entropy_loss
            if self.policy.has_q_head():
                q_error = returns_norm - q_values
                q_loss = huber_loss(q_error, self.args.q_huber_delta).mean()
                loss = loss + float(self.args.q_coef) * q_loss
            loss.backward()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "Update memory check failed (OOM during PPO update). "
                    "Try reducing --buffer_minibatch, --num_envs, or --rollouts_per_update."
                ) from exc
            raise
        else:
            print("Update memory check: OK")
        finally:
            self.policy.vh_optimizer.zero_grad()
            if self.policy.has_q_head() and self.policy.q_optimizer is not None:
                self.policy.q_optimizer.zero_grad()
            self.policy.vla_optimizer.zero_grad()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _get_action(self, obs, deterministic=False):
        total_batch = obs["image"].shape[0]

        values = []
        actions = []
        logprobs = []

        for i in range(0, total_batch, self.args.buffer_inferbatch):
            obs_batch = {k: v[i:i + self.args.buffer_inferbatch] for k, v in obs.items()}
            value, action, logprob = self.policy.get_action(obs_batch, deterministic)
            values.append(value)
            actions.append(action)
            logprobs.append(logprob)

        values = torch.cat(values, dim=0).to(device=self.device)
        actions = torch.cat(actions, dim=0).to(device=self.device)
        logprobs = torch.cat(logprobs, dim=0).to(device=self.device)

        return values, actions, logprobs

    def _obs_to_tensor(self, obs_image):
        if torch.is_tensor(obs_image):
            return obs_image.to(self.device)
        return torch.tensor(obs_image, device=self.device)

    def _get_teacher_action(self, obs, deterministic=True):
        if not self.policy.has_teacher():
            raise RuntimeError("Teacher adapter is not loaded.")

        temperature = self.args.vla_temperature_eval if deterministic else self.args.vla_temperature
        do_sample = (temperature != 0.0) and not deterministic
        effective_temperature = temperature if do_sample else 1.0
        current_adapter = self.policy._get_active_adapter_name()
        was_training = self.policy.vla.training
        if current_adapter != self.policy.teacher_adapter_name:
            self.policy._set_active_adapter(self.policy.teacher_adapter_name)
        if was_training:
            self.policy.vla.eval()

        try:
            actions = []
            total_batch = obs["image"].shape[0]
            for i in range(0, total_batch, self.args.buffer_inferbatch):
                obs_batch = {k: v[i:i + self.args.buffer_inferbatch] for k, v in obs.items()}
                features = self.policy._preprocess_obs(obs_batch)
                _, action, _ = self.policy.vla.predict_action_batch(
                    **features,
                    unnorm_key=self.teacher_key,
                    do_sample=do_sample,
                    temperature=effective_temperature,
                )
                actions.append(action)
            teacher_action = torch.cat(actions, dim=0).to(device=self.device, dtype=torch.int32)
        finally:
            if was_training:
                self.policy.vla.train()
            if current_adapter != self.policy.teacher_adapter_name:
                self.policy._set_active_adapter(current_adapter)

        return teacher_action

    def _allocate_teacher_actions(self, need_teacher_actions: bool):
        if not need_teacher_actions:
            self.teacher_actions = None
            return
        shape = (self.buffer.total_steps, self.args.num_envs, self.teacher_action_dim)
        if self.args.store_rollouts_on_cpu:
            self.teacher_actions = np.zeros(shape, dtype=np.int32)
        else:
            self.teacher_actions = torch.zeros(shape, dtype=torch.int32, device=self.device)

    def _store_teacher_action(self, teacher_action):
        if teacher_action is None or self.teacher_actions is None:
            return
        if self.args.store_rollouts_on_cpu:
            self.teacher_actions[self.buffer.step] = teacher_action.to(torch.int32).cpu().numpy()
        else:
            self.teacher_actions[self.buffer.step].copy_(teacher_action.to(device=self.device, dtype=torch.int32))

    def collect(self, need_teacher_action: bool = False):
        self.policy.prep_rollout()

        obs_image = self._obs_to_tensor(self.buffer.obs[self.buffer.step])
        obs = dict(image=obs_image, task_description=self.buffer.instruction)
        value, action, logprob = self._get_action(obs)
        teacher_action = None
        if need_teacher_action:
            teacher_action = self._get_teacher_action(
                obs,
                deterministic=self.args.bc_to_ref_deterministic,
            )

        return value, action, logprob, teacher_action

    def insert(self, data):
        obs_img, actions, logprob, value_preds, rewards, done = data
        masks = 1.0 - done.to(torch.float32)

        if self.args.store_rollouts_on_cpu:
            obs_img = obs_img.cpu().numpy()
            actions = actions.to(torch.int32).cpu().numpy()
            logprob = logprob.to(torch.float32).cpu().numpy()
            value_preds = value_preds.to(torch.float32).cpu().numpy()
            rewards = rewards.cpu().numpy()
            masks = masks.cpu().numpy()

        self.buffer.insert(obs_img, actions, logprob, value_preds, rewards, masks)

    def compute_endup(self):
        self.policy.prep_rollout()

        obs_image = self._obs_to_tensor(self.buffer.obs[-1])
        obs = dict(image=obs_image, task_description=self.buffer.instruction)
        with torch.no_grad():
            next_value, _, _ = self._get_action(obs)
        if self.args.store_rollouts_on_cpu:
            next_value = next_value.to(torch.float32).cpu().numpy()

        self.buffer.endup(next_value)

    def train(self, steps: int):
        self.policy.prep_training()

        if self.args.alg_name == "ppo":
            train_info = self.alg.train_ppo(self.buffer, steps, teacher_actions=self.teacher_actions)
        elif self.args.alg_name == "grpo":
            train_info = self.alg.train_grpo(self.buffer)
        else:
            raise ValueError(f"Unknown alg_name: {self.args.alg_name}")

        info = {f"train/{k}": v for k, v in train_info.items()}
        if torch.is_tensor(self.buffer.rewards):
            info["buffer/reward_mean"] = self.buffer.rewards.mean().item()
            info["buffer/mask_mean"] = (1.0 - self.buffer.masks).mean().item()
        else:
            info["buffer/reward_mean"] = float(np.mean(self.buffer.rewards))
            info["buffer/mask_mean"] = float(np.mean(1.0 - self.buffer.masks))

        return info

    @torch.no_grad()
    def eval(self, obj_set: str) -> dict:
        self.policy.prep_rollout()
        env_infos = defaultdict(lambda: [])
        datas = None
        if self.args.eval_save_video:
            datas = [{
                "image": [],  # obs_t: [0, T-1]
                "info": [],  # info after executing a_t: [1, T]
            } for _ in range(self.args.num_envs)]

        obs_img, instruction, info = self.env.reset(obj_set=obj_set)
        if datas is not None:
            for i in range(self.args.num_envs):
                datas[i]["image"].append(obs_img[i].cpu().numpy())

        for _ in range(self.args.episode_len):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob = self._get_action(obs, deterministic=True)

            obs_img, reward, done, env_info = self.env.step(action)

            # info
            print({k: round(v.to(torch.float32).mean().tolist(), 4) for k, v in env_info.items() if k != "episode"})
            if "episode" in env_info.keys():
                for k, v in env_info["episode"].items():
                    env_infos[f"{k}"] += v
            if datas is not None:
                for i in range(self.args.num_envs):
                    datas[i]["image"].append(obs_img[i].cpu().numpy())
                    datas[i]["info"].append({k: v[i].tolist() for k, v in env_info.items() if k != "episode"})

        # infos
        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        env_stats = env_stats.copy()

        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        print(f"")

        if datas is not None:
            exp_dir = Path(self.glob_dir) / f"eval_vis_{obj_set}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            for i in range(self.args.num_envs):
                images = datas[i]["image"]
                infos = datas[i]["info"]
                assert len(images) == len(infos) + 1
                success = int(infos[-1]["success"]) if infos else 0
                images_to_video(
                    images, str(exp_dir), f"video_{i}-s_{success}",
                    fps=10, verbose=False
                )

        return env_stats

    @torch.no_grad()
    def render(self, epoch: int, obj_set: str) -> dict:
        self.policy.prep_rollout()

        # init logger
        env_infos = defaultdict(lambda: [])
        datas = [{
            "image": [],  # obs_t: [0, T-1]
            "instruction": "",
            "action": [],  # a_t: [0, T-1]
            "info": [],  # info after executing a_t: [1, T]
        } for idx in range(self.args.num_envs)]

        obs_img, instruction, info = self.env.reset(obj_set)
        print("instruction[:3]:", instruction[:3])

        # data dump: instruction
        for idx in range(self.args.num_envs):
            datas[idx]["instruction"] = instruction[idx]

        for _ in range(self.args.episode_len):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob = self._get_action(obs, deterministic=True)

            obs_img_new, reward, done, env_info = self.env.step(action)

            # info
            print({k: round(v.to(torch.float32).mean().tolist(), 4) for k, v in env_info.items() if k != "episode"})
            if "episode" in env_info.keys():
                for k, v in env_info["episode"].items():
                    env_infos[f"{k}"] += v

            for i in range(self.args.num_envs):
                post_action = self.env._process_action(action)
                log_image = obs_img[i].cpu().numpy()
                log_action = post_action[i].cpu().numpy().tolist()
                log_info = {k: v[i].tolist() for k, v in env_info.items() if k != "episode"}
                datas[i]["image"].append(log_image)
                datas[i]["action"].append(log_action)
                datas[i]["info"].append(log_info)

            # update obs_img
            obs_img = obs_img_new

        # data dump: last image
        for i in range(self.args.num_envs):
            log_image = obs_img[i].cpu().numpy()
            datas[i]["image"].append(log_image)

        # save video
        exp_dir = Path(self.glob_dir) / f"vis_{epoch}_{obj_set}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self.args.num_envs):
            images = datas[i]["image"]
            infos = datas[i]["info"]
            assert len(images) == len(infos) + 1

            if self.args.render_info:
                for j in range(len(infos)):
                    images[j + 1] = visualization.put_info_on_image(
                        images[j + 1], infos[j],
                        extras=[f"Ins: {instruction[i]}"]
                    )

            success = int(infos[-1]["success"])
            images_to_video(images, str(exp_dir), f"video_{i}-s_{success}",
                            fps=10, verbose=False)

        # infos
        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        env_stats_ret = env_stats.copy()

        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        print(f"")

        # save stats
        last_info = {
            idx: {k: env_infos[k][idx] for k in env_infos.keys()}
            for idx in range(self.args.num_envs)
        }

        save_stats = {}
        save_stats["env_name"] = self.args.env_id
        save_stats["ep_len"] = self.args.episode_len
        save_stats["epoch"] = epoch
        save_stats["stats"] = {k: v.item() for k, v in env_stats.items()}
        save_stats["instruction"] = {idx: ins for idx, ins in enumerate(instruction)}
        save_stats["last_info"] = last_info

        yaml.dump(save_stats, open(exp_dir / "stats.yaml", "w"))

        return env_stats_ret

    def run(self):
        max_episodes = self.args.steps_max // (self.args.episode_len * self.args.rollouts_per_update) // self.args.num_envs
        train_start = time.time()

        for episode in range(max_episodes):
            env_infos = defaultdict(lambda: [])
            ep_time = time.time()
            prompts_seen = set()
            steps_before_rollout = (
                episode * self.args.episode_len * self.args.num_envs * self.args.rollouts_per_update
            )
            bc_coef_rollout = self.alg.get_bc_to_ref_coef(steps_before_rollout)
            need_teacher_actions = self.bc_to_ref_enabled and (bc_coef_rollout > 0.0)
            self._allocate_teacher_actions(need_teacher_actions)

            for rollout_idx in range(self.args.rollouts_per_update):
                obs_img, instruction, info = self.env.reset(obj_set="train", same_init=self.args.use_same_init)
                prompts_seen.update(instruction)
                step_offset = rollout_idx * self.args.episode_len
                obs_warmup = obs_img.cpu().numpy() if self.args.store_rollouts_on_cpu else obs_img
                self.buffer.warmup(obs_warmup, instruction, step_offset=step_offset)

                for _ in tqdm(range(self.args.episode_len), desc="rollout"):
                    value, action, logprob, teacher_action = self.collect(
                        need_teacher_action=need_teacher_actions
                    )
                    obs_img, reward, done, env_info = self.env.step(action)
                    self._store_teacher_action(teacher_action)

                    data = (obs_img, action, logprob, value, reward, done)
                    self.insert(data)

                    # info
                    if "episode" in env_info.keys():
                        for k, v in env_info["episode"].items():
                            env_infos[f"{k}"] += v

            # steps
            steps = (episode + 1) * self.args.episode_len * self.args.num_envs * self.args.rollouts_per_update
            env_metrics = {f"env/{k}": np.mean(v) for k, v in env_infos.items()}
            print(self._format_env_metrics(env_metrics))
            if prompts_seen:
                print("Rollout prompts:")
                for prompt in sorted(prompts_seen):
                    print(f"- {prompt}")

            # train and process infos
            self.compute_endup()
            del value, action, logprob, obs_img, reward, done
            gc.collect()
            torch.cuda.empty_cache()

            # train
            self.alg.freeze_actor_updates = episode < self.args.freeze_actor_updates
            if self.alg.freeze_actor_updates:
                print(
                    "Actor updates: frozen "
                    f"({min(episode + 1, self.args.freeze_actor_updates)}/{self.args.freeze_actor_updates})"
                )
            infos = self.train(steps)
            infos.update(env_metrics)

            # log
            wandb.log(infos, step=steps)

            elapsed_time = time.time() - ep_time
            total_elapsed = time.time() - train_start
            remaining_steps = max(0, self.args.steps_max - steps)
            steps_per_sec = steps / total_elapsed if total_elapsed > 0 else 0.0
            eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0.0
            print("-" * 60)
            print(
                f"{self.args.name}: ep {episode:0>4d} | steps {steps} | "
                f"e {elapsed_time:.2f}s | "
                f"total {self._format_seconds(total_elapsed)} | "
                f"eta {self._format_seconds(eta)}"
            )
            reward_mean = infos.get("buffer/reward_mean")
            returns_mean = infos.get("returns_mean")
            reward_text = f"{reward_mean:.6f}" if reward_mean is not None else "n/a"
            returns_text = f"{returns_mean:.6f}" if returns_mean is not None else "n/a"
            print(f"reward_mean={reward_text} | returns_mean={returns_text}")

            # eval
            if episode % self.args.interval_eval == self.args.interval_eval - 1 or episode == max_episodes - 1:
                print(f"Evaluating at {steps}")
                sval_stats = self.eval(obj_set="train")
                sval_stats = {f"eval/{k}": v for k, v in sval_stats.items()}
                wandb.log(sval_stats, step=steps)

                sval_stats = self.eval(obj_set="test")
                sval_stats = {f"eval/{k}_ood": v for k, v in sval_stats.items()}
                wandb.log(sval_stats, step=steps)

            # save
            if episode % self.args.interval_save == self.args.interval_save - 1 or episode == max_episodes - 1:
                print(f"Saving model at {steps}")
                save_path = self.glob_dir / f"steps_{episode:0>4d}"
                self.policy.save(save_path)

                self.render(epoch=episode, obj_set="train")
                self.render(epoch=episode, obj_set="test")


def main():
    args = tyro.cli(Args)
    runner = Runner(args)

    if args.only_render:
        ll = [
            "PutOnPlateInScene25VisionImage-v1",
            "PutOnPlateInScene25VisionTexture03-v1",
            "PutOnPlateInScene25VisionTexture05-v1",
            "PutOnPlateInScene25VisionWhole03-v1",
            "PutOnPlateInScene25VisionWhole05-v1",

            "PutOnPlateInScene25Instruct-v1",
            "PutOnPlateInScene25Plate-v1",
            "PutOnPlateInScene25Position-v1",
            "PutOnPlateInScene25EEPose-v1",
            "PutOnPlateInScene25PositionChange-v1",
            "PutOnPlateInScene25PositionChangeTo-v1"
        ]
        if args.env_id not in ll:
            runner.render(epoch=0, obj_set="train")
        runner.render(epoch=0, obj_set="test")
    else:
        runner.run()


if __name__ == "__main__":
    main()
