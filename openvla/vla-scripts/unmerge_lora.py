import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import draccus
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForVision2Seq

from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPredictionWithValueHead

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


DEFAULT_TARGET_MODULES: List[str] = [
    "proj",
    "qkv",
    "fc1",
    "fc2",  # vision
    "q",
    "kv",
    "fc3",  # project
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",  # llm
]


@dataclass
class UnmergeConfig:
    # fmt: off
    base_path: str = "openvla/openvla-7b"                          # Base OpenVLA model (HF hub or local path)
    merged_path: str = ""                                         # Merged (base + LoRA) model path
    output_path: str = ""                                         # Output LoRA adapter directory
    lora_rank: int = 32
    lora_alpha: int = 0                                           # If 0, use min(rank, 16) like training defaults
    lora_dropout: float = 0.0
    target_modules: Optional[str] = None                          # Comma-separated list, defaults to OpenVLA targets
    model_class: str = "vision2seq"                               # vision2seq | action_value
    vh_mode: str = "a0"                                           # Only used for action_value
    torch_dtype: str = "bfloat16"                                # bfloat16 | float16 | float32
    device_map: str = "cpu"                                      # e.g., "cpu" or "auto"
    print_shapes: bool = False                                   # Print layer weight/LoRA shapes on skips
    # fmt: on


def _dtype_from_str(name: str) -> torch.dtype:
    name = name.lower()
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16" or name == "fp16":
        return torch.float16
    if name == "float32" or name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown torch_dtype: {name}")


def _parse_target_modules(raw: Optional[str]) -> Union[str, List[str]]:
    if raw is None or raw.strip() == "":
        return list(DEFAULT_TARGET_MODULES)
    raw = raw.strip()
    if raw == "all-linear":
        return "all-linear"
    return [t.strip() for t in raw.split(",") if t.strip()]


def _load_model(cfg: UnmergeConfig, path: str):
    torch_dtype = _dtype_from_str(cfg.torch_dtype)
    if cfg.model_class == "action_value":
        return OpenVLAForActionPredictionWithValueHead.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=cfg.device_map,
            vh_mode=cfg.vh_mode,
        )
    if cfg.model_class == "vision2seq":
        return AutoModelForVision2Seq.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=cfg.device_map,
        )
    raise ValueError(f"Unknown model_class: {cfg.model_class}")


def _svd_factorize(
    delta: torch.Tensor, rank: int, a_shape: torch.Size, b_shape: torch.Size
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute rank-r factorization delta ~= B @ A
    out_dim = b_shape[0]
    in_dim = a_shape[1]
    delta_2d = delta.reshape(delta.shape[0], -1)
    if delta_2d.shape[0] != out_dim or delta_2d.shape[1] != in_dim:
        if delta.numel() == out_dim * in_dim:
            delta_2d = delta.reshape(out_dim, in_dim)
        elif delta.transpose(0, 1).numel() == out_dim * in_dim:
            delta_2d = delta.transpose(0, 1).reshape(out_dim, in_dim)
        else:
            raise ValueError(
                f"delta shape {tuple(delta.shape)} can't map to ({out_dim}, {in_dim}); "
                f"likely a Conv2d/other non-linear weight"
            )

    u, s, vh = torch.linalg.svd(delta_2d, full_matrices=False)
    r_eff = min(rank, s.shape[0])
    if r_eff < rank:
        print(f"[warn] rank {rank} > min(dim), using r={r_eff}")
    s_root = torch.sqrt(s[:r_eff])
    b = u[:, :r_eff] * s_root.unsqueeze(0)
    a = s_root.unsqueeze(1) * vh[:r_eff, :]
    return a, b


def _get_submodule_any(model: torch.nn.Module, name: str) -> Optional[torch.nn.Module]:
    candidates = [
        name,
        name.replace("base_model.model.", "", 1),
        name.replace("base_model.", "", 1),
        name.replace("model.", "", 1),
    ]
    for candidate in candidates:
        try:
            return model.get_submodule(candidate)
        except AttributeError:
            continue
    return None


@draccus.wrap()
def unmerge(cfg: UnmergeConfig) -> None:
    if not cfg.merged_path:
        raise ValueError("merged_path must be set")
    if not cfg.output_path:
        raise ValueError("output_path must be set")

    target_modules = _parse_target_modules(cfg.target_modules)
    lora_alpha = cfg.lora_alpha or min(cfg.lora_rank, 16)
    scaling = lora_alpha / float(cfg.lora_rank)
    if scaling == 0.0:
        raise ValueError("lora_alpha must be non-zero")

    print(f"Loading base model: {cfg.base_path}")
    base_model = _load_model(cfg, cfg.base_path)
    print(f"Loading merged model: {cfg.merged_path}")
    merged_model = _load_model(cfg, cfg.merged_path)

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    lora_model = get_peft_model(base_model, lora_config)

    adapter_name = "default"
    matched = 0
    total = 0
    for name, module in lora_model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
            continue

        total += 1
        merged_module = _get_submodule_any(merged_model, name)
        if merged_module is None:
            print(f"[skip] {name}: not found in merged model")
            continue
        if not hasattr(merged_module, "weight") or not hasattr(module, "weight"):
            msg = f"[skip] {name}: missing weight"
            if cfg.print_shapes:
                msg += f" (merged_has={hasattr(merged_module, 'weight')}, base_has={hasattr(module, 'weight')})"
            print(msg)
            continue
        if merged_module.weight.shape != module.weight.shape:
            msg = f"[skip] {name}: shape mismatch {merged_module.weight.shape} vs {module.weight.shape}"
            if cfg.print_shapes:
                lora_A = module.lora_A[adapter_name].weight
                lora_B = module.lora_B[adapter_name].weight
                msg += f" (lora_A={tuple(lora_A.shape)}, lora_B={tuple(lora_B.shape)})"
            print(msg)
            continue

        with torch.no_grad():
            delta = (merged_module.weight.data - module.weight.data).float()
            delta_scaled = delta / scaling
            lora_A = module.lora_A[adapter_name].weight
            lora_B = module.lora_B[adapter_name].weight
            try:
                a, b = _svd_factorize(delta_scaled, cfg.lora_rank, lora_A.shape, lora_B.shape)
            except ValueError as exc:
                msg = f"[skip] {name}: {exc}"
                if cfg.print_shapes:
                    msg += f" (delta={tuple(delta.shape)}, lora_A={tuple(lora_A.shape)}, lora_B={tuple(lora_B.shape)})"
                print(msg)
                continue

            lora_A.copy_(a.to(dtype=lora_A.dtype))
            lora_B.copy_(b.to(dtype=lora_B.dtype))

            recon = (b @ a) * scaling
            rel_err = (recon - delta).norm() / (delta.norm() + 1e-8)
            print(f"[ok] {name}: rel_err={rel_err.item():.6f}")
            matched += 1

    print(f"Processed {matched}/{total} LoRA layers")

    output_path = Path(cfg.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    lora_model.save_pretrained(str(output_path))

    # Copy dataset statistics to adapter (used for unnormalization in policy code)
    merged_stats = Path(cfg.merged_path) / "dataset_statistics.json"
    if merged_stats.exists():
        (output_path / "dataset_statistics.json").write_text(merged_stats.read_text())

    # If merged config has extra norm_stats, keep a record alongside adapter
    merged_config = Path(cfg.merged_path) / "config.json"
    if merged_config.exists():
        config = json.loads(merged_config.read_text())
        (output_path / "merged_config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    unmerge()
