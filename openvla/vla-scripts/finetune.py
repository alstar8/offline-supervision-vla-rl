"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
import copy
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import (
    ActionTokenizer,
    parse_action_dim_loss_weights,
    weighted_action_token_ce_loss,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig, OpenVLAV2Config
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction, OpenVLAV2ForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


_AIRI_CUBES_Q01 = [-0.04, -0.04, -0.04, -0.25, -0.25, -0.25, -1.0]
_AIRI_CUBES_Q99 = [0.04, 0.04, 0.04, 0.25, 0.25, 0.25, 1.0]

FIXED_UNNORM_STATS = {
    "airi_cubes_delta_v1": {
        "action": {
            "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "std": [0.0231, 0.0231, 0.0231, 0.1443, 0.1443, 0.1443, 1.0],
            "min": _AIRI_CUBES_Q01,
            "max": _AIRI_CUBES_Q99,
            "q01": _AIRI_CUBES_Q01,
            "q99": _AIRI_CUBES_Q99,
            "mask": [True, True, True, True, True, True, True],
        },
    },
}

ACTION_DIM_NAMES = ("x", "y", "z", "rot_x", "rot_y", "rot_z", "gripper")

# Dims that can actually change the executed action. On this dataset rotation is
# identically zero, so q01 == q99 and any rotation token unnormalizes back to 0 —
# rotation accuracy cannot correlate with closed-loop success. Averaging over
# these four gives a headline that stays comparable when dim weights change.
CONTROL_DIM_NAMES = ("x", "y", "z", "gripper")


def trained_dim_names(dim_weights: Optional[Sequence[float]]) -> Tuple[str, ...]:
    """Action dims that actually receive gradient.

    A dim with weight 0 is dropped from the loss, so its tokens are whatever the
    base checkpoint happens to emit. Averaging those into the headline accuracy
    makes the metric move independently of learning progress (and, on a dataset
    where the dim is constant, drift downwards), so headline metrics are reported
    over the trained dims only.
    """
    if dim_weights is None:
        return ACTION_DIM_NAMES
    return tuple(name for name, weight in zip(ACTION_DIM_NAMES, dim_weights) if weight > 0)


def _mean_over_dims(per_dim: Dict[str, float], prefix: str, dim_names: Sequence[str]) -> Optional[float]:
    """Mean of `per_dim[f'{prefix}{dim}']` over the dims that are present."""
    values = [per_dim[f"{prefix}{name}"] for name in dim_names if f"{prefix}{name}" in per_dim]
    return sum(values) / len(values) if values else None


# PEFT 0.11.1 matches a string `target_modules` with `re.fullmatch`.
# Keep projector / proprio_projector / lm_head out of LoRA so they can be
# fully trained via `modules_to_save` without the wrap/save conflict that
# `all-linear` hits on projector fc1/fc2.
LORA_TARGET_LLM = r".*language_model\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
LORA_TARGET_VISION_LLM = (
    r".*(vision_backbone\..*(qkv|proj|q|kv|fc1|fc2)"
    r"|language_model\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))$"
)


def resolve_lora_target_modules(lora_target: str) -> str:
    if lora_target == "llm":
        return LORA_TARGET_LLM
    if lora_target == "vision_llm":
        return LORA_TARGET_VISION_LLM
    if lora_target == "all-linear":
        return "all-linear"
    raise ValueError(
        f"Unknown lora_target={lora_target!r}; expected 'all-linear', 'llm', or 'vision_llm'."
    )


def resolve_unnorm_stats(norm_stats: dict, unnorm_key: Optional[str]) -> Optional[dict]:
    if not unnorm_key:
        return None

    if unnorm_key in FIXED_UNNORM_STATS:
        norm_stats[unnorm_key] = copy.deepcopy(FIXED_UNNORM_STATS[unnorm_key])

    if unnorm_key not in norm_stats:
        available_keys = ", ".join(sorted(norm_stats.keys()))
        fixed_keys = ", ".join(sorted(FIXED_UNNORM_STATS.keys()))
        raise KeyError(
            f"Unknown unnorm_key `{unnorm_key}`. Base model keys: [{available_keys}]. "
            f"Fixed in-repo keys: [{fixed_keys}]."
        )

    return norm_stats[unnorm_key]


def add_dataset_statistics_alias(dataset_statistics: dict, dataset_name: str, unnorm_key: Optional[str]) -> None:
    if not unnorm_key or unnorm_key in dataset_statistics or dataset_name not in dataset_statistics:
        return
    dataset_statistics[unnorm_key] = copy.deepcopy(dataset_statistics[dataset_name])


def dataset_statistics_message(dataset_name: str, unnorm_key: Optional[str]) -> str:
    if not unnorm_key:
        return f"Using computed dataset statistics under key `{dataset_name}`."
    if unnorm_key in FIXED_UNNORM_STATS:
        return (
            f"Using fixed in-repo unnorm key `{unnorm_key}` for action statistics; "
            f"saved statistics under `{dataset_name}` and `{unnorm_key}`."
        )
    return f"Using model-provided unnorm key `{unnorm_key}` for dataset statistics."


def action_token_metrics(
    action_tokenizer: ActionTokenizer,
    action_preds,
    action_gt,
    mask,
) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
    correct_preds = (action_preds == action_gt) & mask
    action_accuracy = correct_preds.sum().float() / mask.sum().float()

    pred_ids = action_preds[mask].detach().cpu().numpy()
    gt_ids = action_gt[mask].detach().cpu().numpy()
    continuous_actions_pred = torch.tensor(action_tokenizer.decode_token_ids_to_actions(pred_ids), dtype=torch.float32)
    continuous_actions_gt = torch.tensor(action_tokenizer.decode_token_ids_to_actions(gt_ids), dtype=torch.float32)
    action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

    per_dim_l1 = {}
    per_dim_acc = {}
    dim_count = len(ACTION_DIM_NAMES)
    if continuous_actions_pred.numel() % dim_count == 0:
        pred_by_dim = continuous_actions_pred.reshape(-1, dim_count)
        gt_by_dim = continuous_actions_gt.reshape(-1, dim_count)
        correct_by_dim = correct_preds[mask].detach().cpu().reshape(-1, dim_count)
        for dim_idx, dim_name in enumerate(ACTION_DIM_NAMES):
            per_dim_l1[f"l1_loss/{dim_name}"] = torch.nn.functional.l1_loss(
                pred_by_dim[:, dim_idx],
                gt_by_dim[:, dim_idx],
            ).item()
            per_dim_acc[f"action_accuracy/{dim_name}"] = correct_by_dim[:, dim_idx].float().mean().item()

    return action_accuracy, action_l1_loss, per_dim_l1, per_dim_acc


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    # adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    eval_steps: int = 50                                          # Interval for checkpoint saving
    save_steps: str = "0"
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    # save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance
    lora_target: str = "all-linear"                                 # "all-linear" (legacy), "llm", or "vision_llm"
    train_projector: bool = False                                   # Fully train vision (+ proprio) projectors
    train_action_head: bool = False                                 # Fully train the LM head (action-token output layer)

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on
    unnorm_key: Optional[str] = None

    # OpenVLA_V2 (dual camera + proprio) — only used when vla_model_variant == "v2"
    vla_model_variant: str = "v1"                                   # "v1" (single composited image) or "v2" (separate scene+wrist + proprio)
    proprio_dim: int = 7                                            # Proprioceptive state dimension (v2)
    num_images_in_input: int = 2                                    # Number of separate camera streams (v2)
    # Decode native camera sizes and let Prismatic apply_transform match RL eval.
    skip_image_resize: bool = False
    # Per-dim CE weights x,y,z,rx,ry,rz,gripper. Empty = HuggingFace mean CE.
    action_dim_loss_weights: str = ""


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    # save steps
    save_step_list = [int(x) for x in cfg.save_steps.split(",") if x.strip() != ""]


    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")
    assert cfg.use_lora

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = f"steps_{cfg.max_steps}"
    if not cfg.image_aug:
        exp_id += "-no_aug"

    # Start =>> Build Directories
    run_dir = cfg.run_root_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    is_v2 = cfg.vla_model_variant == "v2"
    if is_v2:
        AutoConfig.register("openvla_v2", OpenVLAV2Config)
        AutoImageProcessor.register(OpenVLAV2Config, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAV2Config, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAV2Config, OpenVLAV2ForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    if is_v2:
        # Hub processing_prismatic still uses torchvision ToTensor. The in-repo
        # processor expects uint8 NCHW (used by OpenVLA-V2 SFT + DataBC).
        hub_ip = processor.image_processor
        processor.image_processor = PrismaticImageProcessor(
            use_fused_vision_backbone=bool(getattr(hub_ip, "use_fused_vision_backbone", False)),
            image_resize_strategy=getattr(hub_ip, "image_resize_strategy", None) or "resize-naive",
            input_sizes=list(getattr(hub_ip, "input_sizes", None) or [(3, 224, 224)]),
            interpolations=list(getattr(hub_ip, "interpolations", None) or ["bicubic"]),
            means=list(getattr(hub_ip, "means", None) or [(0.5, 0.5, 0.5)]),
            stds=list(getattr(hub_ip, "stds", None) or [(0.5, 0.5, 0.5)]),
        )
        # Build the V2 config from the base checkpoint's config; new modules (proprio projector)
        # are randomly initialized and fully trained via LoRA `modules_to_save`.
        base_config = AutoConfig.from_pretrained(cfg.vla_path, trust_remote_code=True)
        config_dict = base_config.to_dict()
        config_dict.pop("model_type", None)
        v2_config = OpenVLAV2Config(
            **config_dict,
            use_proprio=True,
            proprio_dim=cfg.proprio_dim,
            num_images_in_input=cfg.num_images_in_input,
        )
        vla = OpenVLAV2ForActionPrediction.from_pretrained(
            cfg.vla_path,
            config=v2_config,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    # New V2 modules (e.g. `proprio_projector`) are absent from the base checkpoint, so with
    # `low_cpu_mem_usage=True` they stay on the meta device and crash `.to(device_id)`. Re-init
    # any leaf module still on meta as real (CPU) tensors before device placement.
    def _materialize_meta_leaves(module: nn.Module) -> None:
        for child in module.children():
            _materialize_meta_leaves(child)
        direct_params = list(module.parameters(recurse=False))
        if direct_params and any(p.is_meta for p in direct_params):
            module.to_empty(device="cpu")
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    _materialize_meta_leaves(vla)

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig`
    if cfg.use_lora:
        target_modules = resolve_lora_target_modules(cfg.lora_target)
        modules_to_save: list[str] = []
        if cfg.train_projector:
            # Whole containers are safe when LoRA does not wrap projector Linears
            # (`llm` / `vision_llm`). "projector" suffix-matches both `projector`
            # and `proprio_projector` — all projectors are fully trained.
            modules_to_save += ["projector"]
        elif is_v2:
            # The proprio projector is new (not pretrained) — train it fully and save it with the
            # adapter. Target the leaf Linears (not the `proprio_projector` container): with PEFT
            # 0.11.1 + `target_modules="all-linear"`, wrapping the container first would break the
            # fc1/fc2 paths that all-linear also matches, crashing `_get_submodules`.
            modules_to_save += ["proprio_projector.fc1", "proprio_projector.fc2"]
        if cfg.train_action_head:
            # The LM head produces the action-token logits — the action-generating output layer.
            modules_to_save += ["language_model.lm_head"]
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            init_lora_weights="gaussian",
            modules_to_save=modules_to_save or None,
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
        num_images_in_input=cfg.num_images_in_input if is_v2 else 1,
        use_proprio=is_v2,
    )
    unnorm_stats = resolve_unnorm_stats(vla.module.base_model.norm_stats, cfg.unnorm_key)
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        train=True,
        unnorm_stats=unnorm_stats,
        num_images_in_input=cfg.num_images_in_input if is_v2 else 1,
        load_proprio=is_v2,
        skip_image_resize=cfg.skip_image_resize,
    )
    add_dataset_statistics_alias(vla_dataset.dataset_statistics, cfg.dataset_name, cfg.unnorm_key)

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)
        print(dataset_statistics_message(cfg.dataset_name, cfg.unnorm_key))

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )


    # eval
    vla_dataset_eval = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        train=False,
        unnorm_stats=unnorm_stats,
        num_images_in_input=cfg.num_images_in_input if is_v2 else 1,
        load_proprio=is_v2,
        skip_image_resize=cfg.skip_image_resize,
    )
    dataloader_eval = DataLoader(
        vla_dataset_eval,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        name = f"{cfg.dataset_name}-{exp_id}"
        wandb.init(project=cfg.wandb_project, name=name)

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_per_dim_l1 = {name: deque(maxlen=cfg.grad_accumulation_steps) for name in ACTION_DIM_NAMES}
    recent_per_dim_acc = {name: deque(maxlen=cfg.grad_accumulation_steps) for name in ACTION_DIM_NAMES}

    dim_weights = parse_action_dim_loss_weights(cfg.action_dim_loss_weights)
    trained_dims = trained_dim_names(dim_weights)
    if distributed_state.is_main_process:
        print(f"Headline metrics averaged over trained dims only: {trained_dims}", flush=True)
    last_eval_metrics = {}

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                    proprio=batch["proprio"].to(device_id) if "proprio" in batch else None,
                )
                if dim_weights is None:
                    loss = output.loss
                else:
                    loss = weighted_action_token_ce_loss(
                        output.logits,
                        batch["labels"],
                        num_visual_tokens=vla.module.num_visual_tokens,
                        action_token_begin_idx=action_tokenizer.action_token_begin_idx,
                        dim_weights=dim_weights,
                    )

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, vla.module.num_visual_tokens : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            action_accuracy, action_l1_loss, per_dim_l1, per_dim_acc = action_token_metrics(
                action_tokenizer,
                action_preds,
                action_gt,
                mask,
            )

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())
            for dim_name in ACTION_DIM_NAMES:
                l1_key = f"l1_loss/{dim_name}"
                acc_key = f"action_accuracy/{dim_name}"
                if l1_key in per_dim_l1:
                    recent_per_dim_l1[dim_name].append(per_dim_l1[l1_key])
                if acc_key in per_dim_acc:
                    recent_per_dim_acc[dim_name].append(per_dim_acc[acc_key])

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)
            smoothened_per_dim = {}
            for dim_name in ACTION_DIM_NAMES:
                if recent_per_dim_l1[dim_name]:
                    smoothened_per_dim[f"train_l1_loss/{dim_name}"] = (
                        sum(recent_per_dim_l1[dim_name]) / len(recent_per_dim_l1[dim_name])
                    )
                if recent_per_dim_acc[dim_name]:
                    smoothened_per_dim[f"train_action_accuracy/{dim_name}"] = (
                        sum(recent_per_dim_acc[dim_name]) / len(recent_per_dim_acc[dim_name])
                    )

            trained_acc = _mean_over_dims(smoothened_per_dim, "train_action_accuracy/", trained_dims)
            core_acc = _mean_over_dims(smoothened_per_dim, "train_action_accuracy/", CONTROL_DIM_NAMES)

            # Push Metrics to W&B (every 10 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "l1_loss": smoothened_l1_loss,
                        **({"action_accuracy_trained": trained_acc} if trained_acc is not None else {}),
                        **({"action_accuracy_core": core_acc} if core_acc is not None else {}),
                        **smoothened_per_dim,
                    },
                    step=gradient_step_idx,
                )

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()


            if gradient_step_idx % cfg.eval_steps == 0 or gradient_step_idx == cfg.max_steps:
                eval_losses, eval_action_accuracies, eval_l1_losses = [], [], []
                eval_per_dim_l1 = {name: [] for name in ACTION_DIM_NAMES}
                eval_per_dim_acc = {name: [] for name in ACTION_DIM_NAMES}
                for eval_idx, eval_batch in enumerate(dataloader_eval):
                    with torch.no_grad():
                        output_eval: CausalLMOutputWithPast = vla(
                            input_ids=eval_batch["input_ids"].to(device_id),
                            attention_mask=eval_batch["attention_mask"].to(device_id),
                            pixel_values=eval_batch["pixel_values"].to(torch.bfloat16).to(device_id),
                            labels=eval_batch["labels"],
                            proprio=eval_batch["proprio"].to(device_id) if "proprio" in eval_batch else None,
                        )
                        if dim_weights is None:
                            loss = output_eval.loss
                        else:
                            loss = weighted_action_token_ce_loss(
                                output_eval.logits,
                                eval_batch["labels"],
                                num_visual_tokens=vla.module.num_visual_tokens,
                                action_token_begin_idx=action_tokenizer.action_token_begin_idx,
                                dim_weights=dim_weights,
                            )

                        # Compute Accuracy and L1 Loss for Logging
                        action_logits = output_eval.logits[:,
                                        vla.module.num_visual_tokens: -1]
                        action_preds = action_logits.argmax(dim=2)
                        action_gt = eval_batch["labels"][:, 1:].to(action_preds.device)
                        mask = action_gt > action_tokenizer.action_token_begin_idx

                        action_accuracy, action_l1_loss, per_dim_l1, per_dim_acc = action_token_metrics(
                            action_tokenizer,
                            action_preds,
                            action_gt,
                            mask,
                        )

                    eval_losses.append(loss.item())
                    eval_action_accuracies.append(action_accuracy.item())
                    eval_l1_losses.append(action_l1_loss.item())
                    for dim_name in ACTION_DIM_NAMES:
                        l1_key = f"l1_loss/{dim_name}"
                        acc_key = f"action_accuracy/{dim_name}"
                        if l1_key in per_dim_l1:
                            eval_per_dim_l1[dim_name].append(per_dim_l1[l1_key])
                        if acc_key in per_dim_acc:
                            eval_per_dim_acc[dim_name].append(per_dim_acc[acc_key])

                eval_loss = sum(eval_losses) / len(eval_losses)
                eval_action_accuracy = sum(eval_action_accuracies) / len(eval_action_accuracies)
                eval_l1_loss = sum(eval_l1_losses) / len(eval_l1_losses)
                eval_per_dim = {}
                for dim_name in ACTION_DIM_NAMES:
                    if eval_per_dim_l1[dim_name]:
                        eval_per_dim[f"eval_l1_loss/{dim_name}"] = (
                            sum(eval_per_dim_l1[dim_name]) / len(eval_per_dim_l1[dim_name])
                        )
                    if eval_per_dim_acc[dim_name]:
                        eval_per_dim[f"eval_action_accuracy/{dim_name}"] = (
                            sum(eval_per_dim_acc[dim_name]) / len(eval_per_dim_acc[dim_name])
                        )

                eval_acc_trained = _mean_over_dims(eval_per_dim, "eval_action_accuracy/", trained_dims)
                eval_l1_trained = _mean_over_dims(eval_per_dim, "eval_l1_loss/", trained_dims)
                eval_acc_core = _mean_over_dims(eval_per_dim, "eval_action_accuracy/", CONTROL_DIM_NAMES)
                eval_l1_core = _mean_over_dims(eval_per_dim, "eval_l1_loss/", CONTROL_DIM_NAMES)

                if distributed_state.is_main_process:
                    headline = {
                        **({"eval_action_accuracy_trained": eval_acc_trained} if eval_acc_trained is not None else {}),
                        **({"eval_l1_loss_trained": eval_l1_trained} if eval_l1_trained is not None else {}),
                        **({"eval_action_accuracy_core": eval_acc_core} if eval_acc_core is not None else {}),
                        **({"eval_l1_loss_core": eval_l1_core} if eval_l1_core is not None else {}),
                    }
                    last_eval_metrics = {
                        "eval_loss": eval_loss,
                        "eval_action_accuracy": eval_action_accuracy,
                        "eval_l1_loss": eval_l1_loss,
                        **headline,
                        **eval_per_dim,
                        "trained_dims": list(trained_dims),
                        "step": int(gradient_step_idx),
                    }
                    wandb.log(
                        {
                            "eval_loss": eval_loss,
                            "eval_action_accuracy": eval_action_accuracy,
                            "eval_l1_loss": eval_l1_loss,
                            **headline,
                            **eval_per_dim,
                        },
                        step=gradient_step_idx,
                    )

                dist.barrier()


            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx in save_step_list:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    lora_save_dir = run_dir / f"lora_{gradient_step_idx:0>6d}"

                    # Save Processor & Weights
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(lora_save_dir)

                    save_dataset_statistics(vla_dataset.dataset_statistics, lora_save_dir)
                    eval_path = lora_save_dir / "eval_metrics.json"
                    eval_path.write_text(json.dumps(last_eval_metrics, indent=2) + "\n")
                    print(dataset_statistics_message(cfg.dataset_name, cfg.unnorm_key))
                    if last_eval_metrics:
                        per_dim_str = " ".join(
                            f"{name}={last_eval_metrics.get(f'eval_action_accuracy/{name}'):.4f}"
                            for name in trained_dims
                            if last_eval_metrics.get(f"eval_action_accuracy/{name}") is not None
                        )
                        print(
                            "Saved eval metrics | "
                            f"acc_core={last_eval_metrics.get('eval_action_accuracy_core')} "
                            f"| {per_dim_str} "
                            f"| step={last_eval_metrics.get('step')}",
                            flush=True,
                        )

                # Wait for processor and adapter weights to be saved by main process
                dist.barrier()

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
