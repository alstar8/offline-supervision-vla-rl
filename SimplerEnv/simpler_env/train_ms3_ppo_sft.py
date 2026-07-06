import os
import pprint
import random
import gc
import signal
from collections import defaultdict
import time
from pathlib import Path
from typing import Annotated, Optional, List, Dict, Any
import torch
from torch import nn
import numpy as np
import tyro
import wandb
from dataclasses import dataclass
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers.modeling_outputs import CausalLMOutputWithPast
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video
from simpler_env.policies.openvla.openvla_train import huber_loss
from simpler_env.utils.discrete_kl_utils import kl_factorized_from_logits

from simpler_env.env.simpler_wrapper import SimlerWrapper
from simpler_env.utils.replay_buffer import SeparatedReplayBuffer
from simpler_env.utils.wandb_utils import init_wandb_with_online_fallback

signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c
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

    name: str = "PPO-sft"

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
    bc_to_ref_hold_steps: int = 150000
    bc_to_ref_decay_steps: int = 500000

    # offline sft dataset
    sft_data_root_dir: Path = Path("datasets/open-x-embodiment")
    sft_dataset_name: str = "bridge_orig"
    sft_batch_size: int = 8
    sft_shuffle_buffer_size: int = 100000
    sft_image_aug: bool = True

    # other
    wandb: bool = True
    only_render: bool = False
    render_info: bool = False
    update_mem_check: bool = False



class OfflineSFTBatchProvider:
    def __init__(self, args: Args, policy):
        self.args = args
        self.policy = policy
        self.enabled = bool(args.bc_to_ref_enabled)
        self.loader: Optional[DataLoader] = None
        self._iter = None

        if not self.enabled:
            return

        from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
        from prismatic.util.data_utils import PaddedCollatorForActionPrediction
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset

        action_tokenizer = ActionTokenizer(policy.processor.tokenizer)

        def _image_transform_for_rlds(img):
            # RLDSBatchTransform provides PIL images; PrismaticImageProcessor expects [B, C, H, W] tensors.
            if not torch.is_tensor(img):
                img = torch.from_numpy(np.asarray(img)).to(torch.uint8)
            if img.ndim != 3:
                raise ValueError(f"Expected image with 3 dims [H, W, C], got shape {tuple(img.shape)}")
            img = img.permute(2, 0, 1).unsqueeze(0)
            img = policy.processor.image_processor.apply_transform(img)
            return img.squeeze(0)

        batch_transform = RLDSBatchTransform(
            action_tokenizer,
            policy.processor.tokenizer,
            image_transform=_image_transform_for_rlds,
            prompt_builder_fn=PurePromptBuilder if "v01" not in args.vla_path else VicunaV15ChatPromptBuilder,
        )

        image_sizes = tuple(policy.vla.base_model.config.image_sizes)
        unnorm_stats = policy.vla.base_model.norm_stats.get(args.vla_unnorm_key, None)
        dataset = RLDSDataset(
            Path(args.sft_data_root_dir),
            args.sft_dataset_name,
            batch_transform,
            resize_resolution=image_sizes,
            shuffle_buffer_size=args.sft_shuffle_buffer_size,
            image_aug=args.sft_image_aug,
            train=True,
            unnorm_stats=unnorm_stats,
        )
        collator = PaddedCollatorForActionPrediction(
            policy.processor.tokenizer.model_max_length,
            policy.processor.tokenizer.pad_token_id,
            padding_side="right",
        )
        self.loader = DataLoader(
            dataset,
            batch_size=args.sft_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
        )
        self._iter = iter(self.loader)
        print(
            "Offline SFT BC enabled | "
            f"dataset={args.sft_dataset_name} | "
            f"batch_size={args.sft_batch_size} | "
            f"coef_start={args.bc_to_ref_coef} | "
            f"hold_steps={args.bc_to_ref_hold_steps} | "
            f"decay_steps={args.bc_to_ref_decay_steps}"
        )

    def next_batch(self) -> Dict[str, Any]:
        if not self.enabled or self._iter is None:
            raise RuntimeError("Offline SFT BC is not enabled.")
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)


class OpenVLAPPOSFT:
    def __init__(self, all_args: Args, policy, sft_batch_provider: Optional[OfflineSFTBatchProvider]):
        self.args = all_args
        self.policy = policy
        self.sft_batch_provider = sft_batch_provider
        self.ppo_clip = self.args.ppo_clip
        self.ppo_grad_norm = 10.0
        self.ppo_entropy_coef = self.args.alg_entropy_coef
        self.ppo_huber_delta = 10.0
        self.tpdv = self.policy.tpdv
        self.tpdv_vn = self.policy.tpdv_vn
        self.freeze_actor_updates = False
        self.global_step = 0
        self.grad_align_every_minibatches = 10

    def _should_apply_kl_to_ref(self) -> bool:
        return (
            self.args.kl_to_ref_enabled
            and self.policy.has_teacher()
            and self.global_step < self.args.kl_to_ref_steps
        )

    @staticmethod
    def _as_tensor(value, device, dtype=None):
        if torch.is_tensor(value):
            if dtype is None:
                return value.to(device=device)
            return value.to(device=device, dtype=dtype)
        if dtype is None:
            return torch.as_tensor(value, device=device)
        return torch.as_tensor(value, device=device, dtype=dtype)

    @staticmethod
    def _collapse_execution_bins(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError("logits must be [B, D, K].")
        if logits.shape[-1] < 2:
            raise ValueError("Need at least 2 bins to collapse execution bins.")

        probs = torch.softmax(logits, dim=-1)
        probs = torch.cat([probs[..., :-2], probs[..., -2:].sum(dim=-1, keepdim=True)], dim=-1)
        return torch.log(torch.clamp(probs, min=eps))

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

    @staticmethod
    def _flatten_grads(grads):
        flat = [g.reshape(-1) for g in grads if g is not None]
        if not flat:
            return None
        return torch.cat(flat, dim=0)

    def _measure_ppo_bc_grad_alignment(self, ppo_loss: torch.Tensor, bc_loss: torch.Tensor):
        params = self.policy.params_vla
        ppo_grads = torch.autograd.grad(
            ppo_loss,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        bc_grads = torch.autograd.grad(
            bc_loss,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        ppo_vec = self._flatten_grads(ppo_grads)
        bc_vec = self._flatten_grads(bc_grads)
        if ppo_vec is None or bc_vec is None:
            return None
        eps = 1e-12
        ppo_norm = torch.linalg.norm(ppo_vec)
        bc_norm = torch.linalg.norm(bc_vec)
        if ppo_norm.item() <= eps or bc_norm.item() <= eps:
            return None
        cosine = torch.dot(ppo_vec, bc_vec) / (ppo_norm * bc_norm + eps)
        cosine_value = float(cosine.detach().item())
        return dict(
            grad_align_cosine=cosine_value,
            grad_align_conflict=1.0 if cosine_value < 0.0 else 0.0,
            grad_align_conflict_magnitude=max(0.0, -cosine_value),
        )

    def _compute_sft_bc_loss(self) -> torch.Tensor:
        if self.sft_batch_provider is None or not self.sft_batch_provider.enabled:
            raise RuntimeError("SFT BC provider is not configured.")
        batch = self.sft_batch_provider.next_batch()
        output: CausalLMOutputWithPast = self.policy.vla(
            input_ids=batch["input_ids"].to(self.tpdv["device"]),
            attention_mask=batch["attention_mask"].to(self.tpdv["device"]),
            pixel_values=batch["pixel_values"].to(dtype=self.tpdv["dtype"]).to(self.tpdv["device"]),
            labels=batch["labels"].to(self.tpdv["device"]),
            return_dict=True,
        )
        return output.loss.to(dtype=self.tpdv["dtype"])

    def train_ppo_step(self, idx, total, batch):
        obs_image, instruct, actions, value_preds, returns, masks, old_logprob, advantages = batch

        obs_image = self._as_tensor(obs_image, self.tpdv["device"])
        obs = dict(image=obs_image, task_description=instruct)
        actions = self._as_tensor(actions, self.tpdv["device"], dtype=torch.int32)
        value_preds = self._as_tensor(value_preds, self.tpdv["device"], dtype=self.tpdv["dtype"])
        returns = self._as_tensor(returns, self.tpdv_vn["device"], dtype=self.tpdv_vn["dtype"])
        old_logprob = self._as_tensor(old_logprob, self.tpdv["device"], dtype=self.tpdv["dtype"])
        advantages = self._as_tensor(advantages, self.tpdv["device"], dtype=self.tpdv["dtype"])
        returns_norm = returns.to(**self.tpdv)

        logprob, entropy, values = self.policy.evaluate_actions(obs, actions)

        ratio = torch.exp(logprob - old_logprob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
        ppo_policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()
        policy_loss = ppo_policy_loss

        kl_to_ref = None
        kl_to_ref_weighted = None
        if self._should_apply_kl_to_ref():
            student_key = self.args.vla_unnorm_key
            teacher_key = self.args.kl_to_ref_unnorm_key or self.args.vla_unnorm_key
            student_len = self.policy.vla.get_action_dim(student_key)
            teacher_len = self.policy.vla.get_action_dim(teacher_key)
            if student_len != teacher_len:
                raise RuntimeError(
                    f"Student/teacher action dims differ ({student_len} vs {teacher_len}); "
                    "cannot compute KL across different action lengths."
                )
            student_logits = self.policy.get_action_logits(obs, actions, unnorm_key=student_key)
            teacher_logits = self.policy.get_teacher_action_logits(obs, actions, unnorm_key=teacher_key)

            student_logits_exec = self._collapse_execution_bins(student_logits.float())
            teacher_logits_exec = self._collapse_execution_bins(teacher_logits.float())
            effective_bins = student_logits_exec.shape[-1]

            student_edges = self.policy.get_action_edges(student_key, n_bins=effective_bins)
            teacher_edges = self.policy.get_action_edges(teacher_key, n_bins=effective_bins)

            kl_to_ref = kl_factorized_from_logits(
                student_logits_exec,
                student_edges.float(),
                teacher_logits_exec,
                teacher_edges.float(),
            )
            kl_to_ref_weighted = self.args.kl_to_ref_coef * kl_to_ref
            policy_loss = policy_loss + kl_to_ref_weighted

        bc_coef = self.get_bc_to_ref_coef()
        bc_loss = None
        bc_loss_weighted = None
        if bc_coef > 0.0:
            bc_loss = self._compute_sft_bc_loss()
            bc_loss_weighted = bc_coef * bc_loss
            policy_loss = policy_loss + bc_loss_weighted

        grad_align_info = None
        should_measure_grad_align = (
            bc_loss is not None
            and not self.freeze_actor_updates
            and ((idx + 1) % self.grad_align_every_minibatches == 0)
        )
        if should_measure_grad_align:
            grad_align_info = self._measure_ppo_bc_grad_alignment(ppo_policy_loss, bc_loss)

        # Keep detached snapshots for logging, even if actor updates are frozen.
        ppo_policy_loss_raw_log = ppo_policy_loss.detach()
        ppo_policy_loss_weighted_log = ppo_policy_loss.detach()
        policy_loss_total_log = policy_loss.detach()
        bc_loss_raw_log = bc_loss.detach() if bc_loss is not None else None
        bc_loss_weighted_log = bc_loss_weighted.detach() if bc_loss_weighted is not None else None
        kl_to_ref_raw_log = kl_to_ref.detach() if kl_to_ref is not None else None
        kl_to_ref_weighted_log = kl_to_ref_weighted.detach() if kl_to_ref_weighted is not None else None

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
            policy_loss = policy_loss.detach() * 0.0
            entropy_loss = entropy_loss.detach() * 0.0
            bc_loss = bc_loss.detach() * 0.0 if bc_loss is not None else None
        else:
            loss = policy_loss + value_loss - self.ppo_entropy_coef * entropy_loss
        loss /= self.args.alg_gradient_accum
        loss.backward()

        if idx % self.args.alg_gradient_accum == (self.args.alg_gradient_accum - 1) or idx == (total - 1):
            grad_norm = nn.utils.clip_grad_norm_(self.policy.params_vla + self.policy.params_vh, self.ppo_grad_norm)
            self.policy.vh_optimizer.step()
            if not self.freeze_actor_updates:
                self.policy.vla_optimizer.step()
            self.policy.vh_optimizer.zero_grad()
            self.policy.vla_optimizer.zero_grad()
        else:
            grad_norm = None

        info = dict(
            loss=loss.item(),
            policy_loss=policy_loss_total_log.item(),
            ppo_policy_loss_raw=ppo_policy_loss_raw_log.item(),
            ppo_policy_loss_weighted=ppo_policy_loss_weighted_log.item(),
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
            kl_to_ref_active=1.0 if self._should_apply_kl_to_ref() else 0.0,
            kl_to_ref_coef=float(self.args.kl_to_ref_coef),
            kl_to_ref=kl_to_ref_raw_log.item() if kl_to_ref_raw_log is not None else 0.0,
            kl_to_ref_loss_raw=kl_to_ref_raw_log.item() if kl_to_ref_raw_log is not None else 0.0,
            kl_to_ref_loss_weighted=kl_to_ref_weighted_log.item() if kl_to_ref_weighted_log is not None else 0.0,
            bc_to_ref_coef=float(bc_coef),
            bc_to_ref_active=1.0 if bc_coef > 0.0 else 0.0,
            bc_to_ref_loss=bc_loss_raw_log.item() if bc_loss_raw_log is not None else 0.0,
            bc_to_ref_loss_raw=bc_loss_raw_log.item() if bc_loss_raw_log is not None else 0.0,
            bc_to_ref_loss_weighted=bc_loss_weighted_log.item() if bc_loss_weighted_log is not None else 0.0,
        )
        if grad_norm is not None:
            info["grad_norm"] = grad_norm.item()
        if grad_align_info is not None:
            info.update(grad_align_info)

        return info

    def train_grpo(self, buffer):
        from simpler_env.policies.openvla.openvla_train import OpenVLAPPO
        return OpenVLAPPO(self.args, self.policy).train_grpo(buffer)

    def train_ppo(self, buffer, global_step: int):
        train_info = defaultdict(lambda: [])
        self.global_step = global_step
        align_count = 0
        align_conflict_sum = 0.0
        align_conflict_magnitude_sum = 0.0
        align_cosine_values = []

        buffer.compute_returns_ppo()
        minibatch_count = buffer.get_minibatch_count()

        for _ in range(self.args.alg_ppo_epoch):
            data_generator = buffer.feed_forward_generator()
            for idx, batch in tqdm(enumerate(data_generator), total=minibatch_count, desc="train"):
                info = self.train_ppo_step(idx, minibatch_count, batch)
                if "grad_align_cosine" in info:
                    align_count += 1
                    align_conflict_sum += info["grad_align_conflict"]
                    align_conflict_magnitude_sum += info["grad_align_conflict_magnitude"]
                    align_cosine_values.append(info["grad_align_cosine"])
                for key, value in info.items():
                    train_info[key].append(value)

        out = {key: np.mean(value) for key, value in train_info.items()}
        out["grad_align_measurements"] = float(align_count)
        if align_count > 0:
            out["grad_align_conflict_fraction"] = align_conflict_sum / align_count
            out["grad_align_conflict_magnitude"] = align_conflict_magnitude_sum / align_count
            out["grad_align_cosine_mean"] = float(np.mean(align_cosine_values))
            out["grad_align_cosine_std"] = float(np.std(align_cosine_values))
        else:
            out["grad_align_conflict_fraction"] = 0.0
            out["grad_align_conflict_magnitude"] = 0.0
            out["grad_align_cosine_mean"] = 0.0
            out["grad_align_cosine_std"] = 0.0
        return out


class Runner:
    """Orchestrate PPO/GRPO training and evaluation for SimplerEnv tasks.

    Handles seeding, logging setup, policy/algorithm wiring, environment creation,
    replay buffer management, and optional memory checks used before training.
    """
    def __init__(self, all_args: Args):
        self.args = all_args
        self.args_for_logging = self._serialize_args_for_logging(all_args.__dict__)

        # alg_name
        assert self.args.alg_name in ["ppo", "grpo"]

        # set seed
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        # set wandb
        init_wandb_with_online_fallback(
            config=self.args_for_logging,
            project="RLVLA",
            name=self.args.name,
            use_wandb=self.args.wandb,
        )
        self.save_dir = Path(wandb.run.dir)
        self.glob_dir = Path(wandb.run.dir) / ".." / "glob"
        self.glob_dir.mkdir(parents=True, exist_ok=True)

        yaml.dump(self.args_for_logging, open(self.glob_dir / "config.yaml", "w"))
        self._log_args()

        # policy
        from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy
        device_id = 0
        device_id_other = 1 if torch.cuda.device_count() > 1 else 0
        self.device = torch.device("cuda:" + str(device_id))
        self.policy = OpenVLAPolicy(all_args, device_id_other)
        self.sft_batch_provider = OfflineSFTBatchProvider(all_args, self.policy)
        self.alg = OpenVLAPPOSFT(all_args, self.policy, self.sft_batch_provider)

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
        print(yaml.safe_dump(self.args_for_logging, sort_keys=True).strip())

    @staticmethod
    def _serialize_args_for_logging(args_dict: dict) -> dict:
        out = {}
        for key, value in args_dict.items():
            if isinstance(value, Path):
                out[key] = str(value)
            else:
                out[key] = value
        return out

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

    def collect(self):
        self.policy.prep_rollout()

        obs_image = self._obs_to_tensor(self.buffer.obs[self.buffer.step])
        obs = dict(image=obs_image, task_description=self.buffer.instruction)
        value, action, logprob = self._get_action(obs)

        return value, action, logprob

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
            train_info = self.alg.train_ppo(self.buffer, steps)
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

            for rollout_idx in range(self.args.rollouts_per_update):
                obs_img, instruction, info = self.env.reset(obj_set="train", same_init=self.args.use_same_init)
                prompts_seen.update(instruction)
                step_offset = rollout_idx * self.args.episode_len
                obs_warmup = obs_img.cpu().numpy() if self.args.store_rollouts_on_cpu else obs_img
                self.buffer.warmup(obs_warmup, instruction, step_offset=step_offset)

                for _ in tqdm(range(self.args.episode_len), desc="rollout"):
                    value, action, logprob = self.collect()
                    obs_img, reward, done, env_info = self.env.step(action)

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
