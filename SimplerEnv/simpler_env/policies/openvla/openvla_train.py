import json
import os
import inspect
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer
from tqdm import tqdm
from transformers import AutoTokenizer, BatchFeature
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPredictionWithValueHead
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from simpler_env.utils.discrete_kl_utils import (
    build_action_edges_from_stats,
    kl_factorized_from_logits,
)

def huber_loss(e, d):
    a = (abs(e) <= d).to(torch.float32)
    b = (abs(e) > d).to(torch.float32)
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


class OpenVLAPolicy:
    def __init__(self, all_args, device_id: int):
        self.args = all_args
        self.device_id = device_id
        self.tpdv = dict(device=torch.device("cuda:" + str(device_id)), dtype=torch.bfloat16)
        self.tpdv_vn = dict(device=torch.device("cuda:" + str(device_id)), dtype=torch.float32)
        self.action_scale = 1.0
        self.student_adapter_name = None
        self.teacher_adapter_name = None
        self.q_enabled = bool(getattr(self.args, "q_enabled", False))
        self.q_target_enabled = bool(getattr(self.args, "q_target_enabled", False)) and self.q_enabled
        self.q_detach_backbone = bool(getattr(self.args, "q_detach_backbone", True))
        os.environ.setdefault("HF_HUB_RESUME_DOWNLOAD", "1")

        # openvla: register
        self.image_processor = PrismaticImageProcessor.from_pretrained(self.args.vla_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.vla_path, trust_remote_code=True, padding_side="left")
        self.processor = PrismaticProcessor.from_pretrained(
            self.args.vla_path,
            image_processor=self.image_processor,
            tokenizer=self.tokenizer,
            trust_remote_code=True
        )
        # self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        device_map = "cuda:" + str(self.device_id)
        max_memory = None
        if self.args.vla_device_map:
            device_map = self.args.vla_device_map
            if device_map == "auto" and self.args.vla_max_memory_gb > 0:
                max_memory = {self.device_id: f"{self.args.vla_max_memory_gb}GB", "cpu": "64GB"}

        self.vla = self._load_vla_model(
            device_map=device_map,
            max_memory=max_memory,
        )
        if self.args.vla_gradient_checkpointing:
            if hasattr(self.vla, "enable_input_require_grads"):
                self.vla.enable_input_require_grads()
            if hasattr(self.vla, "gradient_checkpointing_enable"):
                self.vla.gradient_checkpointing_enable()

        # openvla: lora
        if not self.args.vla_load_path:
            if self.args.vla_lora_target_modules is None:
                target_modules = [
                    "proj", "qkv", "fc1", "fc2",  # vision
                    "q", "kv", "fc3",  # project
                    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head",  # llm
                ]
            else:
                target_modules = self.args.vla_lora_target_modules

            lora_alpha = self.args.vla_lora_alpha or min(self.args.vla_lora_rank, 16)
            lora_config = LoraConfig(
                r=self.args.vla_lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.0,
                target_modules=target_modules,
                init_lora_weights="gaussian"
            )
            self.vla = get_peft_model(self.vla, lora_config)
        else:
            self.vla = PeftModel.from_pretrained(self.vla, self.args.vla_load_path, is_trainable=True)
            print(f"VLA load: {self.args.vla_load_path}")

            if self.args.vla_unnorm_key not in self.vla.base_model.norm_stats:
                path = Path(self.args.vla_load_path) / "dataset_statistics.json"
                ds = json.load(open(path, "r"))
                self.vla.base_model.norm_stats[self.args.vla_unnorm_key] = ds[self.args.vla_unnorm_key]

        # set value head trainable
        for name, param in self.vla.named_parameters():
            if "value_head" in name:
                param.requires_grad = True

        self.vla.print_trainable_parameters()

        self.student_adapter_name = self._get_active_adapter_name()

        if self.args.kl_to_ref_enabled:
            if not self.args.kl_to_ref_path:
                raise ValueError("kl_to_ref_enabled requires --kl_to_ref_path to be set.")
            self._load_teacher_adapter(self.args.kl_to_ref_path)
            print(
                "Teacher adapter loaded (kl_to_ref_enabled) | "
                f"teacher_path={self.args.kl_to_ref_path} | "
                f"student_key={self.args.vla_unnorm_key} | "
                f"teacher_key={self.args.kl_to_ref_unnorm_key} | "
                f"steps={self.args.kl_to_ref_steps} | "
                f"coef={self.args.kl_to_ref_coef} | "
                "note=KL loss is applied only in trainer codepaths that implement KL-to-ref."
            )
        else:
            print("Teacher adapter disabled (kl_to_ref_enabled=False).")

        # ensure student trainable, teacher frozen
        self._set_trainability(student_trainable=True)

        # openvla: optimizer
        self.params_vh = None
        self.params_q = None
        self.params_vla = None
        self.vh_optimizer = None
        self.q_optimizer = None
        self.vla_optimizer = None
        self._setup_optimizer()

        if self.args.vla_load_path:
            training_state_path = Path(self.args.vla_load_path) / "training_state.pt"
            if training_state_path.exists():
                training_state = torch.load(training_state_path, map_location=self.tpdv["device"])

                if "vh" in training_state:
                    self.vla.value_head.load_state_dict(training_state['vh'], assign=True)
                else:
                    print("Warning: value_head state not found in training_state")
                if self.has_q_head():
                    if "qh" in training_state:
                        self.vla.q_head.load_state_dict(training_state["qh"], assign=True)
                    else:
                        print("Warning: q_head state not found in training_state while q_enabled=True")
                    if self.has_target_q_head() and "target_qh" in training_state:
                        self.vla.target_q_head.load_state_dict(training_state["target_qh"], assign=True)

                self._setup_optimizer()
                self.vh_optimizer.load_state_dict(training_state['vh_optimizer'])
                if self.has_q_head() and "qh_optimizer" in training_state and self.q_optimizer is not None:
                    self.q_optimizer.load_state_dict(training_state["qh_optimizer"])
                self.vla_optimizer.load_state_dict(training_state['vla_optimizer'])

                print(f"Optimizer load: {self.args.vla_load_path}")
            else:
                print(f"Warning: training_state not found in {training_state_path}")

    @staticmethod
    def _supports_q_init_kwargs() -> bool:
        init_sig = inspect.signature(OpenVLAForActionPredictionWithValueHead.__init__)
        params = init_sig.parameters
        return (
            "q_enabled" in params
            and "q_target_enabled" in params
            and "q_detach_backbone" in params
        )

    def _load_vla_model(self, device_map, max_memory):
        base_kwargs = dict(
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map,
            max_memory=max_memory,
            vh_mode="a0",
        )
        if self._supports_q_init_kwargs():
            base_kwargs.update(
                q_enabled=self.q_enabled,
                q_target_enabled=self.q_target_enabled,
                q_detach_backbone=self.q_detach_backbone,
            )
        else:
            if self.q_enabled or self.q_target_enabled:
                print(
                    "WARNING: Installed OpenVLA class does not accept q_* init kwargs. "
                    "Proceeding without Q-head support in this runtime."
                )
                self.q_enabled = False
                self.q_target_enabled = False
        return OpenVLAForActionPredictionWithValueHead.from_pretrained(
            self.args.vla_path,
            **base_kwargs,
        )

    def _setup_optimizer(self):
        self.params_vh = [p for n, p in self.vla.named_parameters() if "value_head" in n and p.requires_grad]
        self.params_q = [p for n, p in self.vla.named_parameters() if "q_head" in n and p.requires_grad]
        self.params_vla = [
            p
            for n, p in self.vla.named_parameters()
            if "value_head" not in n and "q_head" not in n and "target_q_head" not in n and p.requires_grad
        ]

        if len(self.params_vh) == 0 or len(self.params_vla) == 0:
            trainable = [(n, tuple(p.shape)) for n, p in self.vla.named_parameters() if p.requires_grad]
            print("Trainable params debug:")
            for n, s in trainable[:30]:
                print(f"  {n} {s}")
            print(f"Total trainable count: {len(trainable)}")
            if len(self.params_vh) == 0:
                print("Warning: no value_head params found trainable.")
            if len(self.params_vla) == 0:
                print("Warning: no policy params (non-value_head) found trainable.")
        betas = (self.args.vla_optim_beta1, self.args.vla_optim_beta2)
        self.vh_optimizer = AdamW(self.params_vh, lr=self.args.vla_vhlr, betas=betas)
        if self.has_q_head():
            self.q_optimizer = AdamW(self.params_q, lr=self.args.vla_vhlr, betas=betas)
        else:
            self.q_optimizer = None
        self.vla_optimizer = AdamW(self.params_vla, lr=self.args.vla_lr, betas=betas)

    def _set_trainability(self, student_trainable: bool):
        """
        Ensure only student LoRA + value head are trainable; teacher frozen.
        """
        target_adapter = self.student_adapter_name or "default"

        for name, p in self.vla.named_parameters():
            # default: freeze
            p.requires_grad = False

            # always train value head
            if "value_head" in name:
                p.requires_grad = True
                continue
            if "target_q_head" in name:
                p.requires_grad = False
                continue
            if "q_head" in name:
                p.requires_grad = self.q_enabled
                continue

            if not student_trainable:
                continue

            # train only student adapter params (lora_A/lora_B or modules_to_save) by name match
            if f".{target_adapter}." in name:
                if "lora_A" in name or "lora_B" in name or "modules_to_save" in name:
                    p.requires_grad = True

    def _get_active_adapter_name(self) -> str:
        if hasattr(self.vla, "active_adapter"):
            active = self.vla.active_adapter
            if isinstance(active, list):
                return active[0]
            return active
        if hasattr(self.vla, "active_adapters"):
            active = self.vla.active_adapters
            if isinstance(active, list) and active:
                return active[0]
            return active
        return "default"

    def _set_active_adapter(self, name: str):
        if hasattr(self.vla, "set_adapter"):
            self.vla.set_adapter(name)
            return
        if hasattr(self.vla, "set_active_adapters"):
            self.vla.set_active_adapters(name)
            return
        raise RuntimeError("Peft model does not support adapter switching.")

    def _load_teacher_adapter(self, path: str):
        adapter_name = "teacher"
        if not isinstance(self.vla, PeftModel):
            raise RuntimeError("Teacher adapter requires a PeftModel.")
        if not hasattr(self.vla, "load_adapter"):
            raise RuntimeError("Peft model does not support loading adapters.")
        self.vla.load_adapter(path, adapter_name=adapter_name, is_trainable=False)
        self.teacher_adapter_name = adapter_name
        if self.args.kl_to_ref_unnorm_key:
            if self.args.kl_to_ref_unnorm_key not in self.vla.base_model.norm_stats:
                stats_path = Path(path) / "dataset_statistics.json"
                if stats_path.exists():
                    ds = json.load(open(stats_path, "r"))
                    if self.args.kl_to_ref_unnorm_key in ds:
                        self.vla.base_model.norm_stats[self.args.kl_to_ref_unnorm_key] = ds[
                            self.args.kl_to_ref_unnorm_key
                        ]
            if self.args.kl_to_ref_unnorm_key not in self.vla.base_model.norm_stats:
                raise ValueError(
                    f"Teacher unnorm key '{self.args.kl_to_ref_unnorm_key}' not found in norm_stats "
                    f"and could not be loaded from {Path(path) / 'dataset_statistics.json'}."
                )
        else:
            self.args.kl_to_ref_unnorm_key = self.args.vla_unnorm_key
        if self.student_adapter_name:
            self._set_active_adapter(self.student_adapter_name)

    def has_teacher(self) -> bool:
        return self.teacher_adapter_name is not None

    def has_q_head(self) -> bool:
        return bool(getattr(self.vla, "has_q_head", lambda: False)())

    def has_target_q_head(self) -> bool:
        return bool(getattr(self.vla, "has_target_q_head", lambda: False)())

    @torch.no_grad()
    def sync_target_q_head(self, tau: float = 1.0) -> None:
        if self.has_target_q_head():
            self.vla.sync_target_q_head(tau=tau)

    @torch.no_grad()
    def get_teacher_logprob(self, x: dict, action: torch.Tensor) -> torch.Tensor:
        if not self.teacher_adapter_name:
            raise RuntimeError("Teacher adapter is not loaded.")
        current_adapter = self._get_active_adapter_name()
        if current_adapter != self.teacher_adapter_name:
            self._set_active_adapter(self.teacher_adapter_name)
        try:
            logprob, _, _ = self.evaluate_actions(x, action)
        finally:
            if current_adapter != self.teacher_adapter_name:
                self._set_active_adapter(current_adapter)
        return logprob

    def _get_action_logits(self, x: dict, action: torch.Tensor, unnorm_key: str) -> torch.Tensor:
        features = self._preprocess_obs(x, action)
        action_len = self.vla.get_action_dim(unnorm_key)
        outputs = self.vla.forward(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            pixel_values=features["pixel_values"],
            labels=features["labels"],
            output_hidden_states=False,
            return_dict=True,
        )
        logits_tensor = outputs.logits[:, -action_len - 2 : -2]
        logits_tensor = logits_tensor[:, :, 32000 - 256 : 32000]
        return logits_tensor

    def get_action_logits(self, x: dict, action: torch.Tensor, unnorm_key: str) -> torch.Tensor:
        return self._get_action_logits(x, action, unnorm_key)

    @torch.no_grad()
    def get_teacher_action_logits(self, x: dict, action: torch.Tensor, unnorm_key: str) -> torch.Tensor:
        if not self.teacher_adapter_name:
            raise RuntimeError("Teacher adapter is not loaded.")
        current_adapter = self._get_active_adapter_name()
        was_training = self.vla.training
        if current_adapter != self.teacher_adapter_name:
            self._set_active_adapter(self.teacher_adapter_name)
        if was_training:
            self.vla.eval()
        try:
            logits = self._get_action_logits(x, action, unnorm_key)
        finally:
            if was_training:
                self.vla.train()
            if current_adapter != self.teacher_adapter_name:
                self._set_active_adapter(current_adapter)
        return logits

    def get_action_edges(self, unnorm_key: str, n_bins: Optional[int] = None) -> torch.Tensor:
        action_stats = self.vla.base_model.norm_stats[unnorm_key]["action"]
        if n_bins is None:
            n_bins = int(self.vla.config.n_action_bins)
        # Use float32 to avoid the large jitter introduced by bfloat16 eps when
        # constructing monotonically increasing edges for KL rebinding.
        return build_action_edges_from_stats(
            action_stats,
            n_bins=n_bins,
            device=self.tpdv["device"],
            dtype=torch.float32,
        )

    def _preprocess_obs(self, x: dict, action: torch.Tensor = None) -> BatchFeature:
        images = x["image"]
        task_description = x["task_description"]

        assert isinstance(images, torch.Tensor)
        assert len(images.shape) == 4
        assert images.shape[3] == 3
        assert images.dtype == torch.uint8

        assert isinstance(task_description, list)
        assert isinstance(task_description[0], str)
        assert images.shape[0] == len(task_description)

        images = images.permute(0, 3, 1, 2)  # [B, C, H, W]
        images = images.to(**self.tpdv)

        # prompt
        if action is None:
            task_prompt = [f"In: What action should the robot take to {t.lower()}?\nOut: "
                           for t in task_description]
        else:
            assert isinstance(action, torch.Tensor)
            # action = action.cpu().numpy() # [B, dim]
            action_str = self.tokenizer.batch_decode(action)

            task_prompt = [f"In: What action should the robot take to {t.lower()}?\nOut: {a}</s>"
                           for t, a in zip(task_description, action_str)]

        inputs = self.processor(task_prompt, images, padding=True)
        inputs = inputs.to(**self.tpdv)

        if action is not None:
            inputs["labels"] = inputs["input_ids"].clone()

        return inputs

    def get_action(self, x: dict, deterministic) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temperature = self.args.vla_temperature_eval if deterministic else self.args.vla_temperature
        # deterministic=True => greedy decode; deterministic=False => sample using selected temperature.
        do_sample = not deterministic
        effective_temperature = temperature if do_sample else 1.0
        features = self._preprocess_obs(x)

        values, action, logprobs = self.vla.predict_action_batch(
            **features,
            unnorm_key=self.args.vla_unnorm_key,
            do_sample=do_sample,
            temperature=effective_temperature,
        )

        assert len(values.shape) == 2 and values.shape[1] == 1
        assert len(action.shape) == 2 and action.shape[0] == values.shape[0]
        assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1

        return values, action, logprobs

    def get_action_temp(self, x: dict, do_sample, temperature, num_beams) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._preprocess_obs(x)

        _, action, logprobs = self.vla.predict_action_batch(
            **features,
            unnorm_key=self.args.vla_unnorm_key,
            do_sample=do_sample,
            temperature=temperature,
            num_beams=num_beams,
        )

        assert len(action.shape) == 2
        assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1

        return action, logprobs

    def get_value(self, x: dict) -> torch.Tensor:
        features = self._preprocess_obs(x)

        value = self.vla.get_value(**features)

        assert len(value.shape) == 2 and value.shape[1] == 1

        return value

    def get_hidden(self, x: dict) -> torch.Tensor:
        features = self._preprocess_obs(x)

        hs = self.vla.get_hidden(**features)

        return hs

    def evaluate_actions(self, x: dict, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._preprocess_obs(x, action)

        logprobs, entropy, values = self.vla.evaluate_action(
            **features,
            unnorm_key=self.args.vla_unnorm_key
        )

        assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1
        assert len(entropy.shape) == 2 and entropy.shape[1] == 1
        assert len(values.shape) == 2 and values.shape[1] == 1

        return logprobs, entropy, values

    def evaluate_actions_with_q(
        self,
        x: dict,
        action: torch.Tensor,
        use_target_q: bool = False,
        detach_backbone_for_q: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.has_q_head():
            raise RuntimeError("Q head is not enabled.")
        features = self._preprocess_obs(x, action)
        logprobs, entropy, values, q_values = self.vla.evaluate_action_with_q(
            **features,
            unnorm_key=self.args.vla_unnorm_key,
            use_target_q=use_target_q,
            detach_backbone_for_q=detach_backbone_for_q,
        )
        return logprobs, entropy, values, q_values

    def get_q(
        self,
        x: dict,
        action: torch.Tensor,
        use_target_q: bool = False,
        detach_backbone_for_q: Optional[bool] = None,
    ) -> torch.Tensor:
        _, _, _, q_values = self.evaluate_actions_with_q(
            x,
            action,
            use_target_q=use_target_q,
            detach_backbone_for_q=detach_backbone_for_q,
        )
        return q_values

    def prep_rollout(self):
        self.vla.eval()

    def prep_training(self):
        self.vla.train()

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)

        self.vla.save_pretrained(str(path))
        training_state = {
            "vh": self.vla.value_head.state_dict(),
            "vh_optimizer": self.vh_optimizer.state_dict(),
            "vla_optimizer": self.vla_optimizer.state_dict(),
        }
        if self.has_q_head():
            training_state["qh"] = self.vla.q_head.state_dict()
            if self.q_optimizer is not None:
                training_state["qh_optimizer"] = self.q_optimizer.state_dict()
            if self.has_target_q_head():
                training_state["target_qh"] = self.vla.target_q_head.state_dict()
        torch.save(training_state, path / "training_state.pt")

        json.dump(self.vla.base_model.norm_stats, open(path / "dataset_statistics.json", "w"))

    def load(self, path: Path, setup_optimizer: bool = True):
        del self.vla
        torch.cuda.empty_cache()

        self.vla = self._load_vla_model(
            device_map="cuda:" + str(self.device_id),
            max_memory=None,
        )
        self.vla = PeftModel.from_pretrained(self.vla, path, is_trainable=True)
        self.vla.print_trainable_parameters()

        if self.args.vla_unnorm_key not in self.vla.base_model.norm_stats:
            ds = json.load(open(path / "dataset_statistics.json", "r"))
            self.vla.base_model.norm_stats[self.args.vla_unnorm_key] = ds[self.args.vla_unnorm_key]

        training_state_path = path / "training_state.pt"
        if not training_state_path.exists():
            if setup_optimizer:
                raise FileNotFoundError(
                    f"Missing training_state.pt for training-capable load: {training_state_path}"
                )
            print(f"Warning: training_state not found in {training_state_path}; proceeding with eval-only load.")
            return

        training_state = torch.load(training_state_path, map_location=self.tpdv["device"])

        if "vh" in training_state:
            self.vla.value_head.load_state_dict(training_state['vh'], assign=True)
        else:
            print("Warning: value_head state not found in training_state")
        if self.has_q_head() and "qh" in training_state:
            self.vla.q_head.load_state_dict(training_state["qh"], assign=True)
        if self.has_target_q_head() and "target_qh" in training_state:
            self.vla.target_q_head.load_state_dict(training_state["target_qh"], assign=True)

        if setup_optimizer:
            self._setup_optimizer()
            self.vh_optimizer.load_state_dict(training_state['vh_optimizer'])
            if self.has_q_head() and "qh_optimizer" in training_state and self.q_optimizer is not None:
                self.q_optimizer.load_state_dict(training_state["qh_optimizer"])
            self.vla_optimizer.load_state_dict(training_state['vla_optimizer'])

class OpenVLAPPO:
    def __init__(self, all_args, policy: OpenVLAPolicy):
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
        """
        Match SimplerEnv/OpenVLA action decoding semantics:
        the top two token bins map to the same executed action value.
        Collapse them before KL-to-ref so KL is applied in executed-action space.
        """
        if logits.ndim != 3:
            raise ValueError("logits must be [B, D, K].")
        if logits.shape[-1] < 2:
            raise ValueError("Need at least 2 bins to collapse execution bins.")

        probs = torch.softmax(logits, dim=-1)
        probs = torch.cat([probs[..., :-2], probs[..., -2:].sum(dim=-1, keepdim=True)], dim=-1)
        return torch.log(torch.clamp(probs, min=eps))

    def train_ppo_step(self, idx, total, batch):
        obs_image, instruct, actions, value_preds, returns, masks, old_logprob, advantages = batch

        obs_image = self._as_tensor(obs_image, self.tpdv["device"])
        obs = dict(image=obs_image, task_description=instruct)  # uint8
        actions = self._as_tensor(actions, self.tpdv["device"], dtype=torch.int32)
        value_preds = self._as_tensor(value_preds, self.tpdv["device"], dtype=self.tpdv["dtype"])
        returns = self._as_tensor(returns, self.tpdv_vn["device"], dtype=self.tpdv_vn["dtype"])  # float32
        # masks = torch.tensor(masks).to(**self.tpdv)
        old_logprob = self._as_tensor(old_logprob, self.tpdv["device"], dtype=self.tpdv["dtype"])
        advantages = self._as_tensor(advantages, self.tpdv["device"], dtype=self.tpdv["dtype"])
        returns_norm = returns.to(**self.tpdv)

        # Policy loss
        logprob, entropy, values = self.policy.evaluate_actions(obs, actions)

        ratio = torch.exp(logprob - old_logprob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()
        kl_to_ref = None
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
            policy_loss = policy_loss + self.args.kl_to_ref_coef * kl_to_ref

        # Value loss
        value_pred_clipped = value_preds + (values - value_preds).clamp(-self.ppo_clip, self.ppo_clip)
        error_clipped = returns_norm - value_pred_clipped
        error_original = returns_norm - values
        value_loss_clipped = huber_loss(error_clipped, self.ppo_huber_delta)
        value_loss_original = huber_loss(error_original, self.ppo_huber_delta)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        value_clip_indicator = (value_pred_clipped - value_preds).abs() > self.ppo_clip
        value_clip_ratio = value_clip_indicator.to(**self.tpdv).mean()

        value_loss = value_loss.mean()

        # Entropy loss
        entropy_loss = entropy.mean()

        # Total loss
        if self.freeze_actor_updates:
            loss = value_loss
            policy_loss = policy_loss.detach() * 0.0
            entropy_loss = entropy_loss.detach() * 0.0
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
            kl_to_ref_active=1.0 if self._should_apply_kl_to_ref() else 0.0,
            kl_to_ref_coef=float(self.args.kl_to_ref_coef),
        )
        if kl_to_ref is not None:
            info["kl_to_ref"] = kl_to_ref.item()
        else:
            info["kl_to_ref"] = 0.0
        if grad_norm is not None:
            info["grad_norm"] = grad_norm.item()

        return info

    def train_grpo_step(self, idx, total, batch):
        obs_image, instruct, actions, value_preds, returns, masks, old_logprob, advantages = batch

        obs_image = self._as_tensor(obs_image, self.tpdv["device"])
        obs = dict(image=obs_image, task_description=instruct)  # uint8
        actions = self._as_tensor(actions, self.tpdv["device"], dtype=torch.int32)
        # value_preds = torch.tensor(value_preds).to(**self.tpdv)
        # returns = torch.tensor(returns).to(**self.tpdv_vn) # float32
        # masks = torch.tensor(masks).to(**self.tpdv)
        old_logprob = self._as_tensor(old_logprob, self.tpdv["device"], dtype=self.tpdv["dtype"])
        advantages = self._as_tensor(advantages, self.tpdv["device"], dtype=self.tpdv["dtype"])

        # Policy loss
        logprob, entropy, values = self.policy.evaluate_actions(obs, actions)

        ratio = torch.exp(logprob - old_logprob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()

        # Entropy loss
        entropy_loss = entropy.mean()

        # Total loss
        loss = policy_loss - self.ppo_entropy_coef * entropy_loss
        loss /= self.args.alg_gradient_accum
        if self.freeze_actor_updates:
            loss = loss.detach() * 0.0
        else:
            loss.backward()

        if idx % self.args.alg_gradient_accum == (self.args.alg_gradient_accum - 1) or idx == (total - 1):
            if self.freeze_actor_updates:
                grad_norm = None
            else:
                grad_norm = nn.utils.clip_grad_norm_(self.policy.params_vla, self.ppo_grad_norm)
                self.policy.vla_optimizer.step()
                self.policy.vla_optimizer.zero_grad()
        else:
            grad_norm = None

        info = dict(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            entropy_loss=entropy_loss.item(),
            ratio=ratio.mean().item(),
            ratio_median=ratio.median().item(),
            ratio_2=(logprob - old_logprob).mean().exp().item(),

            logprob_mean=logprob.mean().item(),
            logprob_old_mean=old_logprob.mean().item(),
        )
        if grad_norm is not None:
            info["grad_norm"] = grad_norm.item()

        return info

    def train_ppo(self, buffer, global_step: int):
        train_info = defaultdict(lambda: [])
        self.global_step = global_step

        # buffer
        buffer.compute_returns_ppo()
        minibatch_count = buffer.get_minibatch_count()

        for _ in range(self.args.alg_ppo_epoch):
            data_generator = buffer.feed_forward_generator()

            for idx, batch in tqdm(enumerate(data_generator), total=minibatch_count, desc="train"):
                info = self.train_ppo_step(idx, minibatch_count, batch)
                for key, value in info.items():
                    train_info[key].append(value)

        final_info = {}
        for key, value in train_info.items():
            final_info[key] = np.mean(value)

        return final_info

    def train_grpo(self, buffer):
        train_info = defaultdict(lambda: [])

        # buffer
        buffer.compute_returns_grpo()
        minibatch_count = buffer.get_minibatch_count()

        for _ in range(self.args.alg_ppo_epoch):
            data_generator = buffer.feed_forward_generator()

            for idx, batch in tqdm(enumerate(data_generator), total=minibatch_count, desc="train"):
                info = self.train_grpo_step(idx, minibatch_count, batch)
                for key, value in info.items():
                    train_info[key].append(value)

        final_info = {}
        for key, value in train_info.items():
            final_info[key] = np.mean(value)

        return final_info
