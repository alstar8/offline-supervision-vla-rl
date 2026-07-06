import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.optim import AdamW

from baku_model import BakuConfig, BakuPolicy
from baku_model.utils import TruncatedNormal


class TextEmbedder:
    def __init__(self, model_name: str, device: str):
        self.model = SentenceTransformer(model_name, device=device)
        self.cache: dict[str, torch.Tensor] = {}

    def encode(self, texts: list[str], target_device: torch.device) -> torch.Tensor:
        missing = [text for text in dict.fromkeys(texts) if text not in self.cache]
        if missing:
            encoded = self.model.encode(missing, convert_to_numpy=True, show_progress_bar=False)
            for text, emb in zip(missing, encoded):
                self.cache[text] = torch.tensor(emb, dtype=torch.float32)
        return torch.stack([self.cache[text] for text in texts], dim=0).to(target_device)


class CriticHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BakuTrainPolicy:
    def __init__(self, all_args, device_id: int):
        self.args = all_args
        self.device_id = device_id
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.tpdv = dict(device=device, dtype=torch.float32)
        self.tpdv_vn = dict(device=device, dtype=torch.float32)
        self.action_dim = 7
        self.dataset_statistics = self._load_dataset_statistics()
        self.unnorm_key = self.args.baku_unnorm_key
        if self.unnorm_key not in self.dataset_statistics:
            raise KeyError(f"Dataset statistics key `{self.unnorm_key}` not found.")

        ckpt_cfg = self._load_checkpoint_config()
        if ckpt_cfg is None:
            model_cfg = BakuConfig(
                image_size=self.args.baku_image_size,
                action_dim=self.action_dim,
                hidden_dim=self.args.baku_hidden_dim,
                encoder_type=self.args.baku_encoder_type,
                policy_type=self.args.baku_policy_type,
                policy_head=self.args.baku_policy_head,
                action_chunk_size=self.args.baku_action_chunk_size,
                use_language=True,
                language_dim=384,
                language_proj_dim=self.args.baku_language_proj_dim,
                film=self.args.baku_film,
                dropout=self.args.baku_dropout,
                history_len=1,
                train_encoder=True,
                max_seq_len=self.args.baku_max_seq_len,
                gpt_layers=self.args.baku_gpt_layers,
                gpt_heads=self.args.baku_gpt_heads,
            )
        else:
            model_cfg = BakuConfig(**ckpt_cfg)
        self.model_config = model_cfg

        self.model = BakuPolicy(self.model_config).to(device)
        self.value_head = CriticHead(self.model.repr_dim, self.model_config.hidden_dim).to(device)
        self.text_embedder = TextEmbedder(
            self.args.text_encoder_name,
            self.args.text_encoder_device if torch.cuda.is_available() or self.args.text_encoder_device == "cpu" else "cpu",
        )

        if self.args.baku_load_path:
            self._load_weights(Path(self.args.baku_load_path), setup_optimizer=False)

        self.params_vla = [p for p in self.model.parameters() if p.requires_grad]
        self.params_vh = [p for p in self.value_head.parameters() if p.requires_grad]
        self.vla_optimizer = AdamW(self.params_vla, lr=self.args.baku_lr, betas=(self.args.baku_optim_beta1, self.args.baku_optim_beta2))
        self.vh_optimizer = AdamW(self.params_vh, lr=self.args.baku_vhlr, betas=(self.args.baku_optim_beta1, self.args.baku_optim_beta2))

        if self.args.baku_load_path:
            training_state_path = Path(self.args.baku_load_path) / "training_state.pt"
            if training_state_path.exists():
                training_state = torch.load(training_state_path, map_location=device)
                if "vh_optimizer" in training_state:
                    self.vh_optimizer.load_state_dict(training_state["vh_optimizer"])
                if "vla_optimizer" in training_state:
                    self.vla_optimizer.load_state_dict(training_state["vla_optimizer"])

    def _load_dataset_statistics(self) -> dict:
        if self.args.baku_load_path:
            stats_path = Path(self.args.baku_load_path) / "dataset_statistics.json"
            if stats_path.exists():
                with stats_path.open("r") as f:
                    return json.load(f)
        if not self.args.baku_stats_path:
            raise ValueError("Set --baku_stats_path or --baku_load_path so continuous actions can be unnormalized.")
        with Path(self.args.baku_stats_path).open("r") as f:
            return json.load(f)

    def _load_checkpoint_config(self) -> Optional[dict]:
        if not self.args.baku_load_path:
            return None
        checkpoint_path = Path(self.args.baku_load_path) / "checkpoint.pt"
        if not checkpoint_path.exists():
            return None
        payload = torch.load(checkpoint_path, map_location="cpu")
        return payload["model_config"]

    def _load_weights(self, path: Path, setup_optimizer: bool):
        checkpoint_path = path / "checkpoint.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint.pt not found in {path}")
        payload = torch.load(checkpoint_path, map_location=self.tpdv["device"])
        self.model.load_state_dict(payload["model"], strict=True)
        if "value_head" in payload:
            self.value_head.load_state_dict(payload["value_head"], strict=True)
        if setup_optimizer:
            training_state_path = path / "training_state.pt"
            if training_state_path.exists():
                training_state = torch.load(training_state_path, map_location=self.tpdv["device"])
                self.vh_optimizer.load_state_dict(training_state["vh_optimizer"])
                self.vla_optimizer.load_state_dict(training_state["vla_optimizer"])

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.permute(0, 3, 1, 2).to(self.tpdv["device"], dtype=torch.float32) / 255.0
        images = F.interpolate(
            images,
            size=(self.model_config.image_size, self.model_config.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return images.unsqueeze(1)

    def _encode_obs(self, x: dict) -> tuple[torch.Tensor, torch.Tensor]:
        images = self._preprocess_images(x["image"])
        instruction = [text.lower() for text in x["task_description"]]
        language = self.text_embedder.encode(instruction, self.tpdv["device"])
        obs, num_prompt_feats = self.model.encode(images, language_embeds=language)
        pooled = obs.mean(dim=1)
        return obs, pooled, num_prompt_feats

    def _execution_dist(self, dist) -> TruncatedNormal:
        mean = dist.mean[..., : self.action_dim]
        stddev = dist.stddev[..., : self.action_dim]
        return TruncatedNormal(mean, stddev)

    def _dist_and_value(self, x: dict):
        obs, pooled, num_prompt_feats = self._encode_obs(x)
        full_dist = self.model.actor(obs, num_prompt_feats)
        exec_dist = self._execution_dist(full_dist)
        values = self.value_head(pooled)
        return exec_dist, values

    def get_action(self, x: dict, deterministic) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            dist, values = self._dist_and_value(x)
            action = dist.mean[:, -1] if deterministic else dist.sample()[:, -1]
            action = torch.clamp(action, min=-1.0, max=1.0)
            logprob = dist.log_prob(action.unsqueeze(1)).sum(dim=-1)
            return values, action, logprob

    def evaluate_actions(self, x: dict, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, values = self._dist_and_value(x)
        if action.ndim == 2:
            action = action.unsqueeze(1)
        logprob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return logprob, entropy, values

    def prep_rollout(self):
        self.model.eval()
        self.value_head.eval()

    def prep_training(self):
        self.model.train()
        self.value_head.train()

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_config": self.model_config.__dict__,
            "model": self.model.state_dict(),
            "value_head": self.value_head.state_dict(),
        }
        torch.save(payload, path / "checkpoint.pt")
        training_state = {
            "vh_optimizer": self.vh_optimizer.state_dict(),
            "vla_optimizer": self.vla_optimizer.state_dict(),
        }
        torch.save(training_state, path / "training_state.pt")
        with (path / "dataset_statistics.json").open("w") as f:
            json.dump(self.dataset_statistics, f)

    def load(self, path: Path, setup_optimizer: bool = True):
        self._load_weights(Path(path), setup_optimizer=setup_optimizer)

