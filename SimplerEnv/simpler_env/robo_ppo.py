import gc
import math
import pprint
import random
import signal
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Optional

import numpy as np
import torch
import tyro
import wandb
import yaml
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video
from torch import nn
from tqdm import tqdm

from simpler_env.env.simpler_wrapper import SimlerContinuousWrapper
from simpler_env.policies.baku.robo_train import BakuTrainPolicy
from simpler_env.utils.replay_buffer import ContinuousSeparatedReplayBuffer
from simpler_env.utils.wandb_utils import init_wandb_with_online_fallback

signal.signal(signal.SIGINT, signal.default_int_handler)


def huber_loss(e, d):
    a = (abs(e) <= d).to(torch.float32)
    b = (abs(e) > d).to(torch.float32)
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PutOnPlateInScene25OpenReal2Sim-v1"
    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    name: str = "robo-ppo"

    num_envs: int = 64
    episode_len: int = 80
    use_same_init: bool = False
    rollouts_per_update: int = 1
    use_default_task: bool = False

    steps_max: int = 2000000
    interval_eval: int = 10
    interval_save: int = 10
    eval_save_video: bool = True
    eval_deterministic: bool = True
    eval_num_episodes: int = 0
    only_render: bool = False
    render_obj_set: Literal["auto", "train", "test", "both"] = "auto"
    render_info: bool = False

    buffer_inferbatch: int = 32
    buffer_minibatch: int = 512
    buffer_gamma: float = 0.99
    buffer_lambda: float = 0.95
    store_rollouts_on_cpu: bool = False

    alg_name: str = "ppo"
    alg_grpo_fix: bool = True
    alg_gradient_accum: int = 1
    alg_ppo_epoch: int = 1
    ppo_clip: float = 0.2
    alg_entropy_coef: float = 0.0
    freeze_actor_updates: int = 0

    baku_load_path: str = ""
    baku_stats_path: str = ""
    baku_unnorm_key: str = "sft"
    baku_image_size: int = 224
    baku_hidden_dim: int = 512
    baku_encoder_type: str = "resnet34"
    baku_policy_type: str = "gpt"
    baku_policy_head: str = "gaussian"
    baku_action_chunk_size: int = 1
    baku_language_proj_dim: int = 512
    baku_film: bool = True
    baku_dropout: float = 0.1
    baku_max_seq_len: int = 65
    baku_gpt_layers: int = 12
    baku_gpt_heads: int = 8
    baku_lr: float = 1e-4
    baku_vhlr: float = 3e-4
    baku_optim_beta1: float = 0.9
    baku_optim_beta2: float = 0.999

    text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_encoder_device: str = "cuda"

    wandb: bool = True
    wandb_project: str = "BAKURL"
    update_mem_check: bool = False


class ContinuousPPO:
    def __init__(self, all_args, policy: BakuTrainPolicy):
        self.args = all_args
        self.policy = policy
        self.ppo_clip = self.args.ppo_clip
        self.ppo_grad_norm = 10.0
        self.ppo_entropy_coef = self.args.alg_entropy_coef
        self.ppo_huber_delta = 10.0
        self.tpdv = self.policy.tpdv
        self.tpdv_vn = self.policy.tpdv_vn
        self.freeze_actor_updates = False

    @staticmethod
    def _as_tensor(value, device, dtype=None):
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype) if dtype is not None else value.to(device=device)
        return torch.as_tensor(value, device=device, dtype=dtype)

    def train_ppo_step(self, idx, total, batch):
        obs_image, instruct, actions, value_preds, returns, masks, old_logprob, advantages = batch

        obs = dict(
            image=self._as_tensor(obs_image, self.tpdv["device"]),
            task_description=instruct,
        )
        actions = self._as_tensor(actions, self.tpdv["device"], dtype=torch.float32)
        value_preds = self._as_tensor(value_preds, self.tpdv["device"], dtype=self.tpdv["dtype"])
        returns = self._as_tensor(returns, self.tpdv_vn["device"], dtype=self.tpdv_vn["dtype"])
        old_logprob = self._as_tensor(old_logprob, self.tpdv["device"], dtype=self.tpdv["dtype"])
        advantages = self._as_tensor(advantages, self.tpdv["device"], dtype=self.tpdv["dtype"])
        returns_norm = returns.to(**self.tpdv)

        logprob, entropy, values = self.policy.evaluate_actions(obs, actions)

        ratio = torch.exp(logprob - old_logprob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_pred_clipped = value_preds + (values - value_preds).clamp(-self.ppo_clip, self.ppo_clip)
        error_clipped = returns_norm - value_pred_clipped
        error_original = returns_norm - values
        value_loss_clipped = huber_loss(error_clipped, self.ppo_huber_delta)
        value_loss_original = huber_loss(error_original, self.ppo_huber_delta)
        value_loss = torch.max(value_loss_original, value_loss_clipped).mean()

        value_clip_indicator = (value_pred_clipped - value_preds).abs() > self.ppo_clip
        value_clip_ratio = value_clip_indicator.to(**self.tpdv).mean()

        entropy_loss = entropy.mean()
        if self.freeze_actor_updates:
            loss = value_loss
            policy_loss = policy_loss.detach() * 0.0
            entropy_loss = entropy_loss.detach() * 0.0
        else:
            loss = policy_loss + value_loss - self.ppo_entropy_coef * entropy_loss
        loss = loss / self.args.alg_gradient_accum
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
        )
        if grad_norm is not None:
            info["grad_norm"] = grad_norm.item()
        return info

    def train_ppo(self, buffer):
        train_info = defaultdict(list)
        buffer.compute_returns_ppo()
        minibatch_count = buffer.get_minibatch_count()

        for _ in range(self.args.alg_ppo_epoch):
            data_generator = buffer.feed_forward_generator()
            for idx, batch in tqdm(enumerate(data_generator), total=minibatch_count, desc="train"):
                info = self.train_ppo_step(idx, minibatch_count, batch)
                for key, value in info.items():
                    train_info[key].append(value)

        return {key: float(np.mean(value)) for key, value in train_info.items()}


class Runner:
    def __init__(self, all_args: Args):
        self.args = all_args
        if self.args.alg_name != "ppo":
            raise ValueError("robo_ppo.py currently supports only --alg_name ppo.")

        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        init_wandb_with_online_fallback(
            config=all_args.__dict__,
            project=self.args.wandb_project,
            name=self.args.name,
            use_wandb=self.args.wandb,
        )
        self.save_dir = Path(wandb.run.dir)
        self.glob_dir = Path(wandb.run.dir) / ".." / "glob"
        self.glob_dir.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(self._yamlable(all_args.__dict__), open(self.glob_dir / "config.yaml", "w"))
        self._log_args()

        device_id = 0
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.policy = BakuTrainPolicy(all_args, device_id)
        self.alg = ContinuousPPO(all_args, self.policy)
        unnorm_state = self.policy.dataset_statistics[self.args.baku_unnorm_key]["action"]
        self.env = SimlerContinuousWrapper(self.args, unnorm_state)
        self.buffer = ContinuousSeparatedReplayBuffer(
            all_args,
            obs_dim=(480, 640, 3),
            act_dim=7,
            device=self.device,
        )
        print(f"Buffer minibatch count: {self.buffer.get_minibatch_count()}")
        self._check_update_memory()

    @staticmethod
    def _yamlable(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: Runner._yamlable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Runner._yamlable(v) for v in value]
        return value

    def _log_args(self):
        print("Training args (full):")
        print(yaml.safe_dump(self._yamlable(self.args.__dict__), sort_keys=True).strip())

    def _check_update_memory(self):
        if not self.args.update_mem_check:
            return
        batch_size = self.args.buffer_minibatch if self.args.buffer_minibatch > 0 else self.args.episode_len * self.args.num_envs
        obs = dict(
            image=torch.randint(0, 256, (batch_size, 480, 640, 3), dtype=torch.uint8, device=self.device),
            task_description=["memory check"] * batch_size,
        )
        actions = torch.zeros((batch_size, 7), dtype=torch.float32, device=self.device)
        value_preds = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        returns = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        advantages = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        old_logprob = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        try:
            logprob, entropy, values = self.policy.evaluate_actions(obs, actions)
            ratio = torch.exp(logprob - old_logprob)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.alg.ppo_clip, 1 + self.alg.ppo_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_pred_clipped = value_preds + (values - value_preds).clamp(-self.alg.ppo_clip, self.alg.ppo_clip)
            error_clipped = returns - value_pred_clipped
            error_original = returns - values
            value_loss = torch.max(huber_loss(error_original, self.alg.ppo_huber_delta), huber_loss(error_clipped, self.alg.ppo_huber_delta)).mean()
            loss = policy_loss + value_loss - self.alg.ppo_entropy_coef * entropy.mean()
            loss.backward()
        finally:
            self.policy.vh_optimizer.zero_grad()
            self.policy.vla_optimizer.zero_grad()
            torch.cuda.empty_cache()
        print("Update memory check: OK")

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        seconds = max(0, int(seconds))
        return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

    @staticmethod
    def _format_env_metrics(env_metrics: dict) -> str:
        if not env_metrics:
            return "env: none"
        return "env: " + " | ".join(f"{k}={env_metrics[k]:.4f}" for k in sorted(env_metrics.keys()))

    def _get_action(self, obs, deterministic=False):
        total_batch = obs["image"].shape[0]
        values = []
        actions = []
        logprobs = []
        for i in range(0, total_batch, self.args.buffer_inferbatch):
            obs_batch = {k: v[i : i + self.args.buffer_inferbatch] for k, v in obs.items()}
            value, action, logprob = self.policy.get_action(obs_batch, deterministic)
            values.append(value)
            actions.append(action)
            logprobs.append(logprob)
        return (
            torch.cat(values, dim=0).to(self.device),
            torch.cat(actions, dim=0).to(self.device),
            torch.cat(logprobs, dim=0).to(self.device),
        )

    def _obs_to_tensor(self, obs_image):
        if torch.is_tensor(obs_image):
            return obs_image.to(self.device)
        return torch.tensor(obs_image, device=self.device)

    def collect(self):
        self.policy.prep_rollout()
        obs = dict(image=self._obs_to_tensor(self.buffer.obs[self.buffer.step]), task_description=self.buffer.instruction)
        return self._get_action(obs)

    def insert(self, data):
        obs_img, actions, logprob, value_preds, rewards, done = data
        masks = 1.0 - done.to(torch.float32)
        if self.args.store_rollouts_on_cpu:
            obs_img = obs_img.cpu().numpy()
            actions = actions.to(torch.float32).cpu().numpy()
            logprob = logprob.to(torch.float32).cpu().numpy()
            value_preds = value_preds.to(torch.float32).cpu().numpy()
            rewards = rewards.cpu().numpy()
            masks = masks.cpu().numpy()
        self.buffer.insert(obs_img, actions, logprob, value_preds, rewards, masks)

    def compute_endup(self):
        self.policy.prep_rollout()
        obs = dict(image=self._obs_to_tensor(self.buffer.obs[-1]), task_description=self.buffer.instruction)
        with torch.no_grad():
            next_value, _, _ = self._get_action(obs)
        if self.args.store_rollouts_on_cpu:
            next_value = next_value.to(torch.float32).cpu().numpy()
        self.buffer.endup(next_value)

    def train(self):
        self.policy.prep_training()
        train_info = self.alg.train_ppo(self.buffer)
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
        target_episodes = self.args.eval_num_episodes if self.args.eval_num_episodes > 0 else self.args.num_envs
        eval_batches = max(1, math.ceil(target_episodes / self.args.num_envs))
        env_infos = defaultdict(list)
        datas = None
        if self.args.eval_save_video and eval_batches == 1:
            datas = [{"image": [], "info": []} for _ in range(self.args.num_envs)]

        pbar = tqdm(total=eval_batches * self.args.episode_len, desc=f"eval[{obj_set}]", unit="step", leave=False)
        for eval_batch in range(eval_batches):
            obs_img, instruction, _info = self.env.reset(obj_set=obj_set)
            if datas is not None:
                for i in range(self.args.num_envs):
                    datas[i]["image"].append(obs_img[i].cpu().numpy())

            for _ in range(self.args.episode_len):
                obs = dict(image=obs_img, task_description=instruction)
                value, action, logprob = self._get_action(obs, deterministic=self.args.eval_deterministic)
                obs_img, reward, done, env_info = self.env.step(action)
                pbar.update(1)
                if "episode" in env_info:
                    for k, v in env_info["episode"].items():
                        env_infos[k] += v
                if datas is not None:
                    for i in range(self.args.num_envs):
                        datas[i]["image"].append(obs_img[i].cpu().numpy())
                        datas[i]["info"].append({k: v[i].tolist() for k, v in env_info.items() if k != "episode"})
            print(f"[eval] batch {eval_batch + 1}/{eval_batches} complete for obj_set={obj_set}")
        pbar.close()

        env_stats = {k: np.mean(v[:target_episodes]) for k, v in env_infos.items()}
        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        if datas is not None:
            exp_dir = Path(self.glob_dir) / f"eval_vis_{obj_set}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            for i in range(self.args.num_envs):
                infos = datas[i]["info"]
                success = int(infos[-1]["success"]) if infos else 0
                images_to_video(datas[i]["image"], str(exp_dir), f"video_{i}-s_{success}", fps=10, verbose=False)
        return env_stats

    @torch.no_grad()
    def render(self, epoch: int, obj_set: str) -> dict:
        self.policy.prep_rollout()
        env_infos = defaultdict(list)
        datas = [{"image": [], "instruction": "", "action": [], "info": []} for _ in range(self.args.num_envs)]

        obs_img, instruction, _info = self.env.reset(obj_set)
        for idx in range(self.args.num_envs):
            datas[idx]["instruction"] = instruction[idx]

        for _ in range(self.args.episode_len):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob = self._get_action(obs, deterministic=self.args.eval_deterministic)
            obs_img_new, reward, done, env_info = self.env.step(action)
            if "episode" in env_info:
                for k, v in env_info["episode"].items():
                    env_infos[k] += v

            post_action = self.env._process_action(action)
            for i in range(self.args.num_envs):
                datas[i]["image"].append(obs_img[i].cpu().numpy())
                datas[i]["action"].append(post_action[i].cpu().numpy().tolist())
                datas[i]["info"].append({k: v[i].tolist() for k, v in env_info.items() if k != "episode"})
            obs_img = obs_img_new

        for i in range(self.args.num_envs):
            datas[i]["image"].append(obs_img[i].cpu().numpy())

        exp_dir = Path(self.glob_dir) / f"vis_{epoch}_{obj_set}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        for i in range(self.args.num_envs):
            images = datas[i]["image"]
            infos = datas[i]["info"]
            if self.args.render_info:
                for j in range(len(infos)):
                    images[j + 1] = visualization.put_info_on_image(images[j + 1], infos[j], extras=[f"Ins: {instruction[i]}"])
            success = int(infos[-1]["success"]) if infos else 0
            images_to_video(images, str(exp_dir), f"video_{i}-s_{success}", fps=10, verbose=False)

        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        yaml.safe_dump(
            {
                "env_name": self.args.env_id,
                "ep_len": self.args.episode_len,
                "epoch": epoch,
                "stats": {k: float(v) for k, v in env_stats.items()},
                "instruction": {idx: ins for idx, ins in enumerate(instruction)},
            },
            open(exp_dir / "stats.yaml", "w"),
        )
        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        return env_stats

    def run(self):
        max_episodes = self.args.steps_max // (self.args.episode_len * self.args.rollouts_per_update) // self.args.num_envs
        train_start = time.time()
        for episode in range(max_episodes):
            env_infos = defaultdict(list)
            ep_time = time.time()
            prompts_seen = set()

            for rollout_idx in range(self.args.rollouts_per_update):
                obs_img, instruction, _info = self.env.reset(obj_set="train", same_init=self.args.use_same_init)
                prompts_seen.update(instruction)
                step_offset = rollout_idx * self.args.episode_len
                self.buffer.warmup(obs_img.cpu().numpy() if self.args.store_rollouts_on_cpu else obs_img, instruction, step_offset=step_offset)

                for _ in tqdm(range(self.args.episode_len), desc="rollout"):
                    value, action, logprob = self.collect()
                    obs_img, reward, done, env_info = self.env.step(action)
                    self.insert((obs_img, action, logprob, value, reward, done))
                    if "episode" in env_info:
                        for k, v in env_info["episode"].items():
                            env_infos[k] += v

            steps = (episode + 1) * self.args.episode_len * self.args.num_envs * self.args.rollouts_per_update
            env_metrics = {f"env/{k}": np.mean(v) for k, v in env_infos.items()}
            print(self._format_env_metrics(env_metrics))
            if prompts_seen:
                print("Rollout prompts:")
                for prompt in sorted(prompts_seen):
                    print(f"- {prompt}")

            self.compute_endup()
            gc.collect()
            torch.cuda.empty_cache()

            self.alg.freeze_actor_updates = episode < self.args.freeze_actor_updates
            infos = self.train()
            infos.update(env_metrics)
            wandb.log(infos, step=steps)

            elapsed_time = time.time() - ep_time
            total_elapsed = time.time() - train_start
            remaining_steps = max(0, self.args.steps_max - steps)
            steps_per_sec = steps / total_elapsed if total_elapsed > 0 else 0.0
            eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0.0
            print("-" * 60)
            print(
                f"{self.args.name}: ep {episode:0>4d} | steps {steps} | "
                f"e {elapsed_time:.2f}s | total {self._format_seconds(total_elapsed)} | eta {self._format_seconds(eta)}"
            )

            if episode % self.args.interval_eval == self.args.interval_eval - 1 or episode == max_episodes - 1:
                print(f"Evaluating at {steps}")
                wandb.log({f"eval/{k}": v for k, v in self.eval(obj_set='train').items()}, step=steps)
                wandb.log({f"eval/{k}_ood": v for k, v in self.eval(obj_set='test').items()}, step=steps)

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
        if args.render_obj_set == "auto":
            runner.render(epoch=0, obj_set="train")
            runner.render(epoch=0, obj_set="test")
        elif args.render_obj_set == "train":
            runner.render(epoch=0, obj_set="train")
        elif args.render_obj_set == "test":
            runner.render(epoch=0, obj_set="test")
        elif args.render_obj_set == "both":
            runner.render(epoch=0, obj_set="train")
            runner.render(epoch=0, obj_set="test")
        else:
            raise ValueError(f"Unsupported render_obj_set: {args.render_obj_set}")
    else:
        runner.run()


if __name__ == "__main__":
    main()
