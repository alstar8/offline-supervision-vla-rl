import os
import pprint
import random
import gc
import signal
from collections import defaultdict
import time
from pathlib import Path
from typing import Annotated, Optional, List
import torch
import numpy as np
import tyro
import wandb
from dataclasses import dataclass
import yaml
from tqdm import tqdm
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video
from simpler_env.policies.openvla.openvla_train import huber_loss

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

    name: str = "PPO-kl-reward"

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
    kl_to_ref_coef: float = 3e-5
    kl_to_ref_path: str = ""
    kl_to_ref_unnorm_key: str = ""

    # other
    wandb: bool = True
    only_render: bool = False
    render_info: bool = False
    update_mem_check: bool = False



class Runner:
    """Orchestrate PPO/GRPO training and evaluation for SimplerEnv tasks.

    Handles seeding, logging setup, policy/algorithm wiring, environment creation,
    replay buffer management, and optional memory checks used before training.
    """
    def __init__(self, all_args: Args):
        self.args = all_args

        # alg_name
        assert self.args.alg_name in ["ppo", "grpo"]

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
        from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy, OpenVLAPPO
        device_id = 0
        device_id_other = 1 if torch.cuda.device_count() > 1 else 0
        self.device = torch.device("cuda:" + str(device_id))
        self.policy = OpenVLAPolicy(all_args, device_id_other)
        self.kl_reward_enabled = self.args.kl_to_ref_enabled and self.policy.has_teacher()

        self.alg = OpenVLAPPO(all_args, self.policy)
        # This script uses KL as reward shaping only, not as a PPO loss term.
        self.alg._should_apply_kl_to_ref = lambda: False

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

    def _should_apply_kl_reward(self, steps: int) -> bool:
        return self.kl_reward_enabled and steps < self.args.kl_to_ref_steps

    def collect(self, need_teacher_logprob: bool = False):
        self.policy.prep_rollout()

        obs_image = self._obs_to_tensor(self.buffer.obs[self.buffer.step])
        obs = dict(image=obs_image, task_description=self.buffer.instruction)
        value, action, logprob = self._get_action(obs)
        teacher_logprob = None
        if need_teacher_logprob and self.policy.has_teacher():
            teacher_logprob = self.policy.get_teacher_logprob(obs, action)
        return value, action, logprob, teacher_logprob

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
        episode_steps = self.args.episode_len * self.args.rollouts_per_update * self.args.num_envs
        max_episodes = self.args.steps_max // episode_steps
        train_start = time.time()

        for episode in range(max_episodes):
            env_infos = defaultdict(lambda: [])
            ep_time = time.time()
            prompts_seen = set()
            rollout_steps = episode * episode_steps
            reward_env_means = []
            reward_shaped_means = []
            kl_sample_means = []
            kl_penalty_means = []

            for rollout_idx in range(self.args.rollouts_per_update):
                obs_img, instruction, info = self.env.reset(obj_set="train", same_init=self.args.use_same_init)
                prompts_seen.update(instruction)
                step_offset = rollout_idx * self.args.episode_len
                obs_warmup = obs_img.cpu().numpy() if self.args.store_rollouts_on_cpu else obs_img
                self.buffer.warmup(obs_warmup, instruction, step_offset=step_offset)

                for _ in tqdm(range(self.args.episode_len), desc="rollout"):
                    use_kl_reward = self._should_apply_kl_reward(rollout_steps)
                    value, action, logprob, teacher_logprob = self.collect(need_teacher_logprob=use_kl_reward)
                    obs_img, reward, done, env_info = self.env.step(action)
                    reward_env = reward.to(torch.float32)
                    if reward_env.ndim == 1:
                        reward_env = reward_env.unsqueeze(-1)
                    reward_shaped = reward_env
                    if teacher_logprob is not None:
                        kl_sample = (logprob - teacher_logprob).to(torch.float32)
                        if kl_sample.ndim == 1:
                            kl_sample = kl_sample.unsqueeze(-1)
                        kl_penalty = self.args.kl_to_ref_coef * kl_sample
                        reward_shaped = reward_env - kl_penalty
                        kl_sample_means.append(kl_sample.mean().item())
                        kl_penalty_means.append(kl_penalty.mean().item())
                    reward_env_means.append(reward_env.mean().item())
                    reward_shaped_means.append(reward_shaped.mean().item())

                    data = (obs_img, action, logprob, value, reward_shaped, done)
                    self.insert(data)
                    rollout_steps += self.args.num_envs

                    # info
                    if "episode" in env_info.keys():
                        for k, v in env_info["episode"].items():
                            env_infos[f"{k}"] += v

            # steps
            steps = (episode + 1) * self.args.episode_len * self.args.num_envs * self.args.rollouts_per_update
            env_metrics = {f"env/{k}": np.mean(v) for k, v in env_infos.items()}
            rollout_metrics = {
                "rollout/reward_env_mean": float(np.mean(reward_env_means)) if reward_env_means else 0.0,
                "rollout/reward_shaped_mean": float(np.mean(reward_shaped_means)) if reward_shaped_means else 0.0,
                "rollout/kl_reward_active": float(1.0 if kl_sample_means else 0.0),
            }
            if kl_sample_means:
                rollout_metrics["rollout/kl_sample_mean"] = float(np.mean(kl_sample_means))
                rollout_metrics["rollout/kl_penalty_mean"] = float(np.mean(kl_penalty_means))
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
            infos.update(rollout_metrics)

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
