import torch
import numpy as np
from typing import Optional

class SeparatedReplayBuffer(object):
    def __init__(self, all_args, obs_dim, act_dim, device: Optional[torch.device] = None):
        self.ep_len = all_args.episode_len
        self.rollouts_per_update = getattr(all_args, "rollouts_per_update", 1)
        self.total_steps = self.ep_len * self.rollouts_per_update
        self.num_env = all_args.num_envs
        self.gamma = all_args.buffer_gamma
        self.gae_lambda = all_args.buffer_lambda
        self.buffer_minibatch = all_args.buffer_minibatch
        self.alg_grpo_fix = all_args.alg_grpo_fix
        self.store_rollouts_on_cpu = getattr(all_args, "store_rollouts_on_cpu", True)
        self.device = device

        if self.store_rollouts_on_cpu:
            self.obs = np.zeros((self.total_steps + 1, self.num_env, *obs_dim), dtype=np.uint8)
            self.instruction = [""] * self.num_env
            self.value_preds = np.zeros((self.total_steps + 1, self.num_env, 1), dtype=np.float32)
            self.returns = np.zeros((self.total_steps, self.num_env, 1), dtype=np.float32)
            self.actions = np.zeros((self.total_steps, self.num_env, act_dim), dtype=np.int32)
            self.action_log_probs = np.zeros((self.total_steps, self.num_env, act_dim), dtype=np.float32)
            self.rewards = np.zeros((self.total_steps, self.num_env, 1), dtype=np.float32)
            self.masks = np.ones((self.total_steps + 1, self.num_env, 1), dtype=np.float32)
            self.advantages = np.zeros((self.total_steps, self.num_env, 1), dtype=np.float32)
        else:
            if self.device is None:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.obs = torch.zeros((self.total_steps + 1, self.num_env, *obs_dim), dtype=torch.uint8, device=self.device)
            self.instruction = [""] * self.num_env
            self.value_preds = torch.zeros((self.total_steps + 1, self.num_env, 1), dtype=torch.float32, device=self.device)
            self.returns = torch.zeros((self.total_steps, self.num_env, 1), dtype=torch.float32, device=self.device)
            self.actions = torch.zeros((self.total_steps, self.num_env, act_dim), dtype=torch.int32, device=self.device)
            self.action_log_probs = torch.zeros((self.total_steps, self.num_env, act_dim), dtype=torch.float32, device=self.device)
            self.rewards = torch.zeros((self.total_steps, self.num_env, 1), dtype=torch.float32, device=self.device)
            self.masks = torch.ones((self.total_steps + 1, self.num_env, 1), dtype=torch.float32, device=self.device)
            self.advantages = torch.zeros((self.total_steps, self.num_env, 1), dtype=torch.float32, device=self.device)

        self.step = 0

    def _to_tensor(self, value, dtype):
        if torch.is_tensor(value):
            return value.to(device=self.device, dtype=dtype)
        return torch.as_tensor(value, device=self.device, dtype=dtype)

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        if self.step >= self.total_steps:
            raise ValueError(f"Buffer overflow: step={self.step}, total_steps={self.total_steps}")
        if self.store_rollouts_on_cpu:
            self.obs[self.step + 1] = obs.copy()
            self.actions[self.step] = actions.copy()
            self.action_log_probs[self.step] = action_log_probs.copy()
            self.value_preds[self.step] = value_preds.copy()
            self.rewards[self.step] = rewards.copy()
            self.masks[self.step + 1] = masks.copy()
        else:
            self.obs[self.step + 1].copy_(self._to_tensor(obs, torch.uint8))
            self.actions[self.step].copy_(self._to_tensor(actions, torch.int32))
            self.action_log_probs[self.step].copy_(self._to_tensor(action_log_probs, torch.float32))
            self.value_preds[self.step].copy_(self._to_tensor(value_preds, torch.float32))
            self.rewards[self.step].copy_(self._to_tensor(rewards, torch.float32))
            self.masks[self.step + 1].copy_(self._to_tensor(masks, torch.float32))

        self.step += 1

    def warmup(self, obs, instruction, step_offset: int = 0):
        if self.store_rollouts_on_cpu:
            if torch.is_tensor(obs):
                obs = obs.cpu().numpy()
            self.obs[step_offset] = obs
            self.instruction = instruction
            self.masks[step_offset] = 1.0
        else:
            self.obs[step_offset].copy_(self._to_tensor(obs, torch.uint8))
            self.instruction = instruction
            self.masks[step_offset] = 1.0

        self.step = step_offset

    def endup(self, next_value):
        if self.store_rollouts_on_cpu:
            self.value_preds[-1] = next_value
        else:
            self.value_preds[-1].copy_(self._to_tensor(next_value, torch.float32))

    def compute_returns_ppo(self):
        if self.store_rollouts_on_cpu:
            gae = 0
            for step in reversed(range(self.rewards.shape[0])):
                vt1 = self.value_preds[step + 1]
                vt = self.value_preds[step]

                delta = self.rewards[step] + self.gamma * vt1 * self.masks[step + 1] - vt
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + vt

            # calc adv
            advantages = self.returns - self.value_preds[:-1]
            mean_advantages = advantages.mean()
            std_advantages = advantages.std()
            self.advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        else:
            gae = torch.zeros_like(self.rewards[0])
            for step in reversed(range(self.rewards.shape[0])):
                vt1 = self.value_preds[step + 1]
                vt = self.value_preds[step]

                delta = self.rewards[step] + self.gamma * vt1 * self.masks[step + 1] - vt
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + vt

            # calc adv
            advantages = self.returns - self.value_preds[:-1]
            mean_advantages = advantages.mean()
            std_advantages = advantages.std()
            self.advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

    def compute_returns_grpo(self):
        if self.store_rollouts_on_cpu:
            if self.alg_grpo_fix:
                rewards_valid = self.rewards[self.rewards != 0]
                rewards_norm = self.rewards.copy()
                rewards_norm[rewards_norm != 0] -= rewards_valid.mean()
                rewards_norm[rewards_norm != 0] /= (rewards_valid.std() + 1e-5)
            else:
                rewards_norm = (self.rewards - self.rewards.mean()) / (self.rewards.std() + 1e-5)

            returns = 0
            for step in reversed(range(self.rewards.shape[0])):
                returns = rewards_norm[step] + self.masks[step + 1] * returns
                self.returns[step] = returns

            # calc adv
            self.advantages = self.returns.copy()
        else:
            if self.alg_grpo_fix:
                rewards_valid = self.rewards[self.rewards != 0]
                rewards_norm = self.rewards.clone()
                rewards_norm[rewards_norm != 0] -= rewards_valid.mean()
                rewards_norm[rewards_norm != 0] /= (rewards_valid.std() + 1e-5)
            else:
                rewards_norm = (self.rewards - self.rewards.mean()) / (self.rewards.std() + 1e-5)

            returns = torch.zeros_like(self.rewards[0])
            for step in reversed(range(self.rewards.shape[0])):
                returns = rewards_norm[step] + self.masks[step + 1] * returns
                self.returns[step] = returns

            # calc adv
            self.advantages = self.returns.clone()

    def get_minibatch_count(self):
        episode_length, n_rollout_threads = self.rewards.shape[:2]
        batch_size = episode_length * n_rollout_threads

        if self.buffer_minibatch < 0:
            num_mini_batch = 1
        else:
            assert batch_size % self.buffer_minibatch == 0
            num_mini_batch = batch_size // self.buffer_minibatch

        return num_mini_batch

    def feed_forward_generator(self):
        episode_length, n_rollout_threads = self.rewards.shape[:2]
        batch_size = episode_length * n_rollout_threads

        if self.buffer_minibatch < 0:
            num_mini_batch = 1
        else:
            assert batch_size % self.buffer_minibatch == 0
            num_mini_batch = batch_size // self.buffer_minibatch

        if self.store_rollouts_on_cpu:
            rand = torch.randperm(batch_size).numpy()
            sampler = [rand[i * self.buffer_minibatch:(i + 1) * self.buffer_minibatch] for i in range(num_mini_batch)]

            obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
            actions = self.actions.reshape(-1, self.actions.shape[-1])
            value_preds = self.value_preds[:-1].reshape(-1, 1)
            returns = self.returns.reshape(-1, 1)
            masks = self.masks[:-1].reshape(-1, 1)
            action_logits = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
            advantages = self.advantages.reshape(-1, 1)

            for indices in sampler:
                # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
                obs_batch = obs[indices]
                actions_batch = actions[indices]
                value_preds_batch = value_preds[indices]
                return_batch = returns[indices]
                masks_batch = masks[indices]
                old_action_logits_batch = action_logits[indices]
                adv_targ = advantages[indices]

                # instruct
                instruct_indices = indices % n_rollout_threads
                instruct_batch = [self.instruction[i] for i in instruct_indices]

                yield (obs_batch, instruct_batch, actions_batch, value_preds_batch, return_batch, masks_batch,
                       old_action_logits_batch, adv_targ)
        else:
            rand = torch.randperm(batch_size, device=self.device)
            sampler = [rand[i * self.buffer_minibatch:(i + 1) * self.buffer_minibatch] for i in range(num_mini_batch)]

            obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
            actions = self.actions.reshape(-1, self.actions.shape[-1])
            value_preds = self.value_preds[:-1].reshape(-1, 1)
            returns = self.returns.reshape(-1, 1)
            masks = self.masks[:-1].reshape(-1, 1)
            action_logits = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
            advantages = self.advantages.reshape(-1, 1)

            for indices in sampler:
                # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
                obs_batch = obs[indices]
                actions_batch = actions[indices]
                value_preds_batch = value_preds[indices]
                return_batch = returns[indices]
                masks_batch = masks[indices]
                old_action_logits_batch = action_logits[indices]
                adv_targ = advantages[indices]

                # instruct
                instruct_indices = (indices % n_rollout_threads).to("cpu").tolist()
                instruct_batch = [self.instruction[i] for i in instruct_indices]

                yield (obs_batch, instruct_batch, actions_batch, value_preds_batch, return_batch, masks_batch,
                       old_action_logits_batch, adv_targ)


class ContinuousSeparatedReplayBuffer(SeparatedReplayBuffer):
    def __init__(self, all_args, obs_dim, act_dim, device: Optional[torch.device] = None):
        self.ep_len = all_args.episode_len
        self.rollouts_per_update = getattr(all_args, "rollouts_per_update", 1)
        self.total_steps = self.ep_len * self.rollouts_per_update
        self.num_env = all_args.num_envs
        self.gamma = all_args.buffer_gamma
        self.gae_lambda = all_args.buffer_lambda
        self.buffer_minibatch = all_args.buffer_minibatch
        self.alg_grpo_fix = all_args.alg_grpo_fix
        self.store_rollouts_on_cpu = getattr(all_args, "store_rollouts_on_cpu", True)
        self.device = device

        if self.store_rollouts_on_cpu:
            self.obs = np.zeros((self.total_steps + 1, self.num_env, *obs_dim), dtype=np.uint8)
            self.instruction = [""] * self.num_env
            self.value_preds = np.zeros((self.total_steps + 1, self.num_env, 1), dtype=np.float32)
            self.returns = np.zeros((self.total_steps, self.num_env, 1), dtype=np.float32)
            self.actions = np.zeros((self.total_steps, self.num_env, act_dim), dtype=np.float32)
            self.action_log_probs = np.zeros((self.total_steps, self.num_env, 1), dtype=np.float32)
            self.rewards = np.zeros((self.total_steps, self.num_env, 1), dtype=np.float32)
            self.masks = np.ones((self.total_steps + 1, self.num_env, 1), dtype=np.float32)
            self.advantages = np.zeros((self.total_steps, self.num_env, 1), dtype=np.float32)
        else:
            if self.device is None:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.obs = torch.zeros((self.total_steps + 1, self.num_env, *obs_dim), dtype=torch.uint8, device=self.device)
            self.instruction = [""] * self.num_env
            self.value_preds = torch.zeros((self.total_steps + 1, self.num_env, 1), dtype=torch.float32, device=self.device)
            self.returns = torch.zeros((self.total_steps, self.num_env, 1), dtype=torch.float32, device=self.device)
            self.actions = torch.zeros((self.total_steps, self.num_env, act_dim), dtype=torch.float32, device=self.device)
            self.action_log_probs = torch.zeros((self.total_steps, self.num_env, 1), dtype=torch.float32, device=self.device)
            self.rewards = torch.zeros((self.total_steps, self.num_env, 1), dtype=torch.float32, device=self.device)
            self.masks = torch.ones((self.total_steps + 1, self.num_env, 1), dtype=torch.float32, device=self.device)
            self.advantages = torch.zeros((self.total_steps, self.num_env, 1), dtype=torch.float32, device=self.device)

        self.step = 0

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        if self.step >= self.total_steps:
            raise ValueError(f"Buffer overflow: step={self.step}, total_steps={self.total_steps}")
        if self.store_rollouts_on_cpu:
            self.obs[self.step + 1] = obs.copy()
            self.actions[self.step] = actions.copy()
            self.action_log_probs[self.step] = action_log_probs.copy()
            self.value_preds[self.step] = value_preds.copy()
            self.rewards[self.step] = rewards.copy()
            self.masks[self.step + 1] = masks.copy()
        else:
            self.obs[self.step + 1].copy_(self._to_tensor(obs, torch.uint8))
            self.actions[self.step].copy_(self._to_tensor(actions, torch.float32))
            self.action_log_probs[self.step].copy_(self._to_tensor(action_log_probs, torch.float32))
            self.value_preds[self.step].copy_(self._to_tensor(value_preds, torch.float32))
            self.rewards[self.step].copy_(self._to_tensor(rewards, torch.float32))
            self.masks[self.step + 1].copy_(self._to_tensor(masks, torch.float32))

        self.step += 1
