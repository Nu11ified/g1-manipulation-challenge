#!/usr/bin/env python3
"""Agnostic PPO training framework.

`train.py` knows nothing about the pick-and-place task: no reward shaping, no
teacher policy, no stage labels. Callers (e.g. `demo.py`) inject those via:

  * `env_factory`: a "module:attr" reference to a pickable callable that
    returns an env exposing `reset(seed)`, `step(action) -> (obs, reward, done,
    info)`, `snapshot()`, `restore(state)`, and attributes `prev_metrics`,
    `controller` (used by teachers). The env's `step` must populate
    `info["stage"]` if the caller wants stage-weighted reset-state sampling.
  * `teacher_factory`: an optional "module:attr" reference to a callable
    returning a teacher with `reset()` and `act(env, obs) -> np.ndarray`. If
    omitted, training runs pure PPO from the reward function alone — slower,
    but the framework does not require a teacher.
  * `stage_weights`: dict[str, float] used only when `--demo-reset-prob > 0` to
    bias demo-state restarts toward later task phases.

The module-level reference strings (rather than callables) are required
because macOS uses spawn-based multiprocessing — workers must be able to
re-import the factories by name.
"""

from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from sim import EnvConfig, SCRIPT_DIR


# --------------------------------------------------------------------------- #
# Factory resolution.
# --------------------------------------------------------------------------- #
def _resolve(spec: str) -> Callable:
  """Resolve a "module:attr" reference. Raises if spec is None/empty."""
  if not spec:
    raise ValueError("factory spec must be a non-empty 'module:attr' string")
  module_name, _, attr = spec.partition(":")
  if not module_name or not attr:
    raise ValueError(f"invalid factory spec {spec!r}; expected 'module:attr'")
  return getattr(importlib.import_module(module_name), attr)


# --------------------------------------------------------------------------- #
# Vectorized env with optional teacher side-channel.
# --------------------------------------------------------------------------- #
def _worker(
  remote,
  parent_remote,
  env_factory_spec: str,
  env_cfg: EnvConfig,
  worker_seed: int,
  fixed_reset_seed: int | None,
  teacher_factory_spec: str | None,
) -> None:
  """Run one environment process for the spawn-based vectorized collector."""
  parent_remote.close()
  env_factory = _resolve(env_factory_spec)
  env = env_factory(env_cfg, seed=worker_seed, render_images=True, verbose=False, training=True)
  teacher = _resolve(teacher_factory_spec)(env.cfg) if teacher_factory_spec else None
  reset_rng = np.random.default_rng(worker_seed + 10_000)

  obs: dict[str, np.ndarray] | None = None
  demo_reset_states: list[dict[str, Any]] = []
  demo_reset_weights: np.ndarray | None = None
  demo_reset_probability = 0.0

  def auto_reset() -> dict[str, np.ndarray]:
    nonlocal obs
    use_demo_reset = (
      demo_reset_states
      and demo_reset_probability > 0.0
      and reset_rng.random() < demo_reset_probability
    )
    if use_demo_reset:
      probs = demo_reset_weights if demo_reset_weights is not None else None
      reset_idx = int(reset_rng.choice(len(demo_reset_states), p=probs))
      obs = env.restore(demo_reset_states[reset_idx])
    else:
      obs = env.reset(seed=fixed_reset_seed)
    if teacher is not None:
      teacher.reset()
    return obs

  try:
    while True:
      cmd, data = remote.recv()
      if cmd == "reset":
        if isinstance(data, dict):
          obs = env.restore(data)
        else:
          reset_seed = fixed_reset_seed if fixed_reset_seed is not None else data
          obs = env.reset(seed=reset_seed)
        if teacher is not None:
          teacher.reset()
        remote.send(obs)
      elif cmd == "step":
        obs, reward, done, info = env.step(data)
        if done:
          final = dict(info)
          final["terminal_observation"] = obs
          obs = auto_reset()
          info = final
        remote.send((obs, reward, done, info))
      elif cmd == "teacher_action":
        if teacher is None:
          raise RuntimeError("teacher action requested but no teacher factory configured")
        if obs is None:
          raise RuntimeError("teacher action requested before reset")
        remote.send(teacher.act(env, obs))
      elif cmd == "set_demo_resets":
        demo_reset_states = [_deep_copy(state) for state in data["states"]]
        weights = np.asarray(data["weights"], dtype=np.float64)
        demo_reset_weights = weights / max(weights.sum(), 1e-8)
        remote.send(True)
      elif cmd == "set_demo_reset_probability":
        demo_reset_probability = float(data)
        remote.send(True)
      elif cmd == "close":
        remote.close()
        break
      else:
        raise RuntimeError(f"unknown worker cmd {cmd}")
  finally:
    remote.close()


def _deep_copy(value: Any) -> Any:
  """Copy nested arrays/containers so demo reset states can be reused safely."""
  if isinstance(value, np.ndarray):
    return value.copy()
  if isinstance(value, dict):
    return {key: _deep_copy(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_deep_copy(item) for item in value]
  if isinstance(value, np.generic):
    return value.item()
  return value


class SubprocVecEnv:
  """Spawn-based vectorized environment wrapper used by PPO collection."""

  def __init__(
    self,
    num_envs: int,
    env_factory_spec: str,
    env_cfg: EnvConfig,
    base_seed: int,
    fixed_reset_seed: int | None,
    teacher_factory_spec: str | None = None,
  ):
    self.num_envs = num_envs
    self.teacher_enabled = teacher_factory_spec is not None
    ctx = mp.get_context("spawn")
    self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])
    self.processes = []
    for idx, (remote, work_remote) in enumerate(zip(self.remotes, self.work_remotes)):
      proc = ctx.Process(
        target=_worker,
        args=(
          work_remote,
          remote,
          env_factory_spec,
          env_cfg,
          base_seed + idx,
          fixed_reset_seed,
          teacher_factory_spec,
        ),
        daemon=True,
      )
      proc.start()
      work_remote.close()
      self.processes.append(proc)

  def reset(self) -> dict[str, np.ndarray]:
    for idx, remote in enumerate(self.remotes):
      remote.send(("reset", idx))
    return _stack_obs([remote.recv() for remote in self.remotes])

  def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    for remote, action in zip(self.remotes, actions):
      remote.send(("step", action))
    results = [remote.recv() for remote in self.remotes]
    obs, rewards, dones, infos = zip(*results)
    return _stack_obs(list(obs)), np.asarray(rewards, np.float32), np.asarray(dones, np.bool_), list(infos)

  def teacher_actions(self) -> np.ndarray:
    if not self.teacher_enabled:
      raise RuntimeError("teacher_actions called without a teacher factory")
    for remote in self.remotes:
      remote.send(("teacher_action", None))
    return np.stack([remote.recv() for remote in self.remotes], axis=0).astype(np.float32)

  def set_demo_resets(self, states: list[dict[str, Any]], weights: np.ndarray) -> None:
    payload = {"states": states, "weights": np.asarray(weights, dtype=np.float32)}
    for remote in self.remotes:
      remote.send(("set_demo_resets", payload))
    for remote in self.remotes:
      remote.recv()

  def set_demo_reset_probability(self, probability: float) -> None:
    for remote in self.remotes:
      remote.send(("set_demo_reset_probability", probability))
    for remote in self.remotes:
      remote.recv()

  def close(self) -> None:
    for remote in self.remotes:
      try:
        remote.send(("close", None))
      except Exception:
        pass
    for proc in self.processes:
      proc.join(timeout=1.0)


def _stack_obs(obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
  """Stack per-environment observations into a batch-first observation dict."""
  return {key: np.stack([obs[key] for obs in obs_list], axis=0) for key in obs_list[0]}


def _obs_to_torch(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
  """Move a NumPy observation dict onto the target torch device."""
  return {
    key: torch.as_tensor(value, dtype=torch.float32, device=device)
    for key, value in obs.items()
  }


def _atanh(x: torch.Tensor) -> torch.Tensor:
  """Numerically stable inverse tanh used for tanh-squashed action targets."""
  x = x.clamp(-0.98, 0.98)
  return 0.5 * (torch.log1p(x) - torch.log1p(-x))


# --------------------------------------------------------------------------- #
# Policy network: proprio MLP + two image CNNs fused into a shared trunk.
# --------------------------------------------------------------------------- #
class VisionEncoder(nn.Module):
  """Small CNN encoder used for both head and wrist camera streams."""

  def __init__(self, channels: int, embed_dim: int):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(channels, 16, kernel_size=5, stride=2, padding=2),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
      nn.AdaptiveAvgPool2d(1),
      nn.Flatten(),
      nn.Linear(64, embed_dim),
      nn.LayerNorm(embed_dim),
      nn.Tanh(),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


class RoutinePolicy(nn.Module):
  """Actor-critic policy over proprioception and two image observations."""

  def __init__(
    self,
    proprio_dim: int,
    image_channels: int,
    action_dim: int,
    image_embed_dim: int = 64,
  ):
    super().__init__()
    self.head_encoder = VisionEncoder(image_channels, image_embed_dim)
    self.wrist_encoder = VisionEncoder(image_channels, image_embed_dim)
    self.proprio_encoder = nn.Sequential(
      nn.Linear(proprio_dim, 256),
      nn.LayerNorm(256),
      nn.Tanh(),
      nn.Linear(256, 256),
      nn.Tanh(),
    )
    fused_dim = 256 + 2 * image_embed_dim
    self.trunk = nn.Sequential(
      nn.Linear(fused_dim, 256),
      nn.Tanh(),
      nn.Linear(256, 256),
      nn.Tanh(),
    )
    self.policy_head = nn.Linear(256, action_dim)
    self.value_head = nn.Linear(256, 1)
    # σ ≈ 0.16 pre-tanh: enough for exploration, small enough to preserve BC prior.
    self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

  def _features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
    prop = self.proprio_encoder(obs["proprio"])
    head = self.head_encoder(obs["head"])
    wrist = self.wrist_encoder(obs["wrist"])
    return self.trunk(torch.cat([prop, head, wrist], dim=-1))

  def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    feat = self._features(obs)
    return self.policy_head(feat), self.value_head(feat).squeeze(-1)

  def _dist(self, obs: dict[str, torch.Tensor]) -> tuple[Independent, torch.Tensor, torch.Tensor]:
    mean, value = self.forward(obs)
    std = self.log_std.exp().expand_as(mean)
    return Independent(Normal(mean, std), 1), mean, value

  def act(
    self, obs: dict[str, torch.Tensor], deterministic: bool = False
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dist, mean, value = self._dist(obs)
    pre_tanh = mean if deterministic else dist.rsample()
    action = torch.tanh(pre_tanh)
    correction = torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6)).sum(dim=-1)
    log_prob = dist.log_prob(pre_tanh) - correction
    return action, pre_tanh, log_prob, value

  def evaluate_actions(
    self, obs: dict[str, torch.Tensor], pre_tanh: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dist, _, value = self._dist(obs)
    action = torch.tanh(pre_tanh)
    correction = torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6)).sum(dim=-1)
    log_prob = dist.log_prob(pre_tanh) - correction
    entropy = dist.base_dist.entropy().sum(dim=-1)
    return log_prob, entropy, value


class ExportActor(nn.Module):
  """Deterministic actor wrapper used when exporting ONNX inference graphs."""

  def __init__(self, policy: RoutinePolicy):
    super().__init__()
    self.policy = policy

  def forward(self, proprio: torch.Tensor, head: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
    mean, _ = self.policy({"proprio": proprio, "head": head, "wrist": wrist})
    return torch.tanh(mean)


# --------------------------------------------------------------------------- #
# Rollout buffer, demo dataset, GAE.
# --------------------------------------------------------------------------- #
class RolloutBuffer:
  """Fixed-size rollout storage for PPO updates across all environments."""

  def __init__(self, horizon: int, num_envs: int, obs_template: dict[str, np.ndarray], action_dim: int):
    self.horizon = horizon
    self.num_envs = num_envs
    self.obs = {
      key: np.zeros((horizon, num_envs, *value.shape[1:]), dtype=np.float32)
      for key, value in obs_template.items()
    }
    self.actions = np.zeros((horizon, num_envs, action_dim), dtype=np.float32)
    self.pretanh = np.zeros((horizon, num_envs, action_dim), dtype=np.float32)
    self.log_probs = np.zeros((horizon, num_envs), dtype=np.float32)
    self.rewards = np.zeros((horizon, num_envs), dtype=np.float32)
    self.dones = np.zeros((horizon, num_envs), dtype=np.float32)
    self.values = np.zeros((horizon, num_envs), dtype=np.float32)
    self.advantages = np.zeros((horizon, num_envs), dtype=np.float32)
    self.returns = np.zeros((horizon, num_envs), dtype=np.float32)

  def flatten(self) -> dict[str, np.ndarray]:
    batch = self.horizon * self.num_envs
    flat = {
      "actions": self.actions.reshape(batch, self.actions.shape[-1]),
      "pretanh": self.pretanh.reshape(batch, self.pretanh.shape[-1]),
      "log_probs": self.log_probs.reshape(batch),
      "advantages": self.advantages.reshape(batch),
      "returns": self.returns.reshape(batch),
      "values": self.values.reshape(batch),
    }
    for key, value in self.obs.items():
      flat[key] = value.reshape(batch, *value.shape[2:])
    return flat


@dataclass
class DemoDataset:
  """Teacher-generated transitions used for behavior-cloning warm-starts."""

  obs: dict[str, np.ndarray]
  actions: np.ndarray
  reset_states: list[dict[str, Any]]
  reset_weights: np.ndarray
  successes: int
  episodes: int
  seeds: list[int]

  def sample(self, batch_size: int, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    indices = np.random.randint(0, len(self.actions), size=batch_size)
    obs = {
      key: torch.as_tensor(value[indices], dtype=torch.float32, device=device)
      for key, value in self.obs.items()
    }
    actions = torch.as_tensor(self.actions[indices], dtype=torch.float32, device=device)
    return obs, actions


def compute_gae(
  rewards: np.ndarray,
  dones: np.ndarray,
  values: np.ndarray,
  last_values: np.ndarray,
  gamma: float,
  gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Compute generalized advantage estimates and bootstrapped returns."""
  horizon = rewards.shape[0]
  advantages = np.zeros_like(rewards)
  last_adv = np.zeros_like(last_values)
  for t in reversed(range(horizon)):
    mask = 1.0 - dones[t]
    next_values = last_values if t == horizon - 1 else values[t + 1]
    delta = rewards[t] + gamma * next_values * mask - values[t]
    last_adv = delta + gamma * gae_lambda * mask * last_adv
    advantages[t] = last_adv
  returns = advantages + values
  return advantages, returns


def parse_seed_list(seed_spec: str) -> list[int]:
  """Parse a comma-separated list of integer teacher/demo seeds."""
  seeds = [int(part.strip()) for part in seed_spec.split(",") if part.strip()]
  if not seeds:
    raise ValueError("teacher seed list cannot be empty")
  return seeds


def collect_demo_dataset(
  env_factory_spec: str,
  teacher_factory_spec: str,
  env_cfg: EnvConfig,
  stage_weights: dict[str, float],
  required_successes: int,
  max_episodes: int,
  seeds: list[int],
  repeats_per_seed: int,
) -> DemoDataset:
  """Collect teacher trajectories used for behavior-cloning warm-starts."""
  env_factory = _resolve(env_factory_spec)
  teacher_factory = _resolve(teacher_factory_spec)
  env = env_factory(env_cfg, seed=seeds[0], render_images=True, verbose=False, training=False)

  successes = 0
  episodes = 0
  dataset_obs: dict[str, list[np.ndarray]] | None = None
  dataset_actions: list[np.ndarray] = []
  dataset_reset_states: list[dict[str, Any]] = []
  dataset_reset_weights: list[float] = []
  fallback_episodes: list[dict[str, Any]] = []

  for _ in range(repeats_per_seed):
    for seed in seeds:
      if successes >= required_successes or episodes >= max_episodes:
        break
      obs = env.reset(seed=seed)
      teacher = teacher_factory(env.cfg)
      teacher.reset()
      episode_obs = {key: [] for key in obs}
      episode_actions = []
      episode_reset_states: list[dict[str, Any]] = []
      episode_reset_weights: list[float] = []
      stage_label = "approach"
      best_stage_rank = 0
      best_lift = 0.0
      best_dist = float("inf")
      done = False
      info: dict[str, Any] = {}
      while not done:
        for key, value in obs.items():
          episode_obs[key].append(value.copy())
        action = teacher.act(env, obs)
        episode_actions.append(action.copy())
        weight = stage_weights.get(stage_label, 0.0)
        if weight > 0.0:
          episode_reset_states.append(env.snapshot())
          episode_reset_weights.append(weight)
        obs, _, done, info = env.step(action)
        stage_label = info.get("stage", stage_label)
        best_lift = max(best_lift, float(info.get("lift_height", 0.0)))
        if "stage" in info:
          rank = _stage_rank(stage_label, stage_weights)
          best_stage_rank = max(best_stage_rank, rank)
        if env.prev_metrics:
          best_dist = min(best_dist, float(env.prev_metrics.get("block_to_target", best_dist)))
      episodes += 1
      if info["episode"]["success"]:
        if dataset_obs is None:
          dataset_obs = {key: [] for key in episode_obs}
        for key in episode_obs:
          dataset_obs[key].append(np.stack(episode_obs[key], axis=0))
        dataset_actions.append(np.stack(episode_actions, axis=0))
        dataset_reset_states.extend(episode_reset_states)
        dataset_reset_weights.extend(episode_reset_weights)
        successes += 1
      elif episode_reset_states:
        fallback_episodes.append({
          "score": (best_stage_rank, best_lift, -best_dist),
          "obs": {key: np.stack(values, axis=0) for key, values in episode_obs.items()},
          "actions": np.stack(episode_actions, axis=0),
          "reset_states": episode_reset_states,
          "reset_weights": episode_reset_weights,
        })
    if successes >= required_successes or episodes >= max_episodes:
      break

  if dataset_obs is None or not dataset_actions:
    if not fallback_episodes:
      raise RuntimeError("demo dataset collection produced no usable trajectories")
    fallback_episodes.sort(key=lambda item: item["score"], reverse=True)
    chosen = fallback_episodes[:max(1, required_successes)]
    dataset_obs = {key: [] for key in chosen[0]["obs"]}
    for item in chosen:
      for key, value in item["obs"].items():
        dataset_obs[key].append(value)
      dataset_actions.append(item["actions"])
      dataset_reset_states.extend(item["reset_states"])
      dataset_reset_weights.extend(item["reset_weights"])
  if not dataset_reset_states:
    raise RuntimeError("demo dataset collection produced no reset states")

  return DemoDataset(
    obs={key: np.concatenate(values, axis=0) for key, values in dataset_obs.items()},
    actions=np.concatenate(dataset_actions, axis=0),
    reset_states=dataset_reset_states,
    reset_weights=np.asarray(dataset_reset_weights, dtype=np.float32),
    successes=successes,
    episodes=episodes,
    seeds=list(seeds),
  )


def _stage_rank(stage: str, stage_weights: dict[str, float]) -> int:
  """Map a stage label to its relative ordering in the stage-weight config."""
  ordered = sorted(stage_weights.items(), key=lambda kv: kv[1])
  for idx, (name, _) in enumerate(ordered):
    if name == stage:
      return idx
  return 0


# --------------------------------------------------------------------------- #
# PPO + BC updates.
# --------------------------------------------------------------------------- #
def pretrain_from_demo_dataset(
  policy: RoutinePolicy,
  optimizer: torch.optim.Optimizer,
  device: torch.device,
  dataset: DemoDataset,
  cfg: "TrainingConfig",
) -> tuple[dict[str, float], int]:
  """Run supervised behavior-cloning updates before PPO fine-tuning."""
  batch_size = len(dataset.actions)
  indices = np.arange(batch_size)
  metrics = defaultdict(list)

  for _ in range(cfg.demo_bc_epochs):
    np.random.shuffle(indices)
    for start in range(0, batch_size, cfg.minibatch_size):
      mb_idx = indices[start:start + cfg.minibatch_size]
      obs = {
        key: torch.as_tensor(dataset.obs[key][mb_idx], dtype=torch.float32, device=device)
        for key in ("proprio", "head", "wrist")
      }
      target_actions = torch.as_tensor(dataset.actions[mb_idx], dtype=torch.float32, device=device)
      # Match in pre-tanh space — tanh saturates at ±1, so post-tanh targets
      # at ±1 push logits to ±∞ with vanishing gradient and BC stalls.
      target_pretanh = _atanh(target_actions.clamp(-0.98, 0.98))
      pred_mean = policy.forward(obs)[0]
      bc_loss = torch.nn.functional.smooth_l1_loss(pred_mean, target_pretanh)

      optimizer.zero_grad(set_to_none=True)
      bc_loss.backward()
      nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
      optimizer.step()

      metrics["loss/demo_bc"].append(bc_loss.item())

  metrics["demo/episodes"] = [float(dataset.episodes)]
  metrics["demo/successes"] = [float(dataset.successes)]
  metrics["demo/success_rate"] = [float(dataset.successes / max(1, dataset.episodes))]
  metrics["demo/reset_states"] = [float(len(dataset.reset_states))]
  return {key: float(np.mean(values)) for key, values in metrics.items()}, batch_size


def ppo_update(
  policy: RoutinePolicy,
  optimizer: torch.optim.Optimizer,
  buffer: RolloutBuffer,
  device: torch.device,
  cfg: "TrainingConfig",
  demo_dataset: DemoDataset | None = None,
  demo_aux_weight: float = 0.0,
) -> dict[str, float]:
  """Run one PPO optimization phase over a filled rollout buffer."""
  batch = buffer.flatten()
  advantages = batch["advantages"]
  advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
  batch["advantages"] = advantages

  batch_size = len(batch["advantages"])
  indices = np.arange(batch_size)
  metrics = defaultdict(list)
  kl_cutoff = 1.5 * cfg.target_kl
  kl_exceeded = False

  for _ in range(cfg.update_epochs):
    if kl_exceeded:
      break
    np.random.shuffle(indices)
    for start in range(0, batch_size, cfg.minibatch_size):
      mb_idx = indices[start:start + cfg.minibatch_size]
      obs = {
        key: torch.as_tensor(batch[key][mb_idx], dtype=torch.float32, device=device)
        for key in ("proprio", "head", "wrist")
      }
      pretanh = torch.as_tensor(batch["pretanh"][mb_idx], dtype=torch.float32, device=device)
      old_log_probs = torch.as_tensor(batch["log_probs"][mb_idx], dtype=torch.float32, device=device)
      returns = torch.as_tensor(batch["returns"][mb_idx], dtype=torch.float32, device=device)
      adv = torch.as_tensor(batch["advantages"][mb_idx], dtype=torch.float32, device=device)
      old_values = torch.as_tensor(batch["values"][mb_idx], dtype=torch.float32, device=device)

      log_probs, entropy, values = policy.evaluate_actions(obs, pretanh)
      ratio = torch.exp(log_probs - old_log_probs)
      unclipped = ratio * adv
      clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
      policy_loss = -torch.min(unclipped, clipped).mean()

      value_delta = values - old_values
      value_clipped = old_values + value_delta.clamp(-cfg.value_clip_eps, cfg.value_clip_eps)
      value_loss_unclipped = (values - returns).pow(2)
      value_loss_clipped = (value_clipped - returns).pow(2)
      value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

      entropy_bonus = entropy.mean()
      loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_bonus
      if demo_dataset is not None and demo_aux_weight > 0.0:
        demo_obs, demo_actions = demo_dataset.sample(len(mb_idx), device)
        demo_target = _atanh(demo_actions.clamp(-0.98, 0.98))
        demo_aux_loss = torch.nn.functional.smooth_l1_loss(policy.forward(demo_obs)[0], demo_target)
        loss = loss + demo_aux_weight * demo_aux_loss
        metrics["loss/demo_aux"].append(demo_aux_loss.item())

      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
      optimizer.step()

      with torch.no_grad():
        # Schulman approximation: ((ratio - 1) - log(ratio)).mean() — unbiased,
        # always non-negative, lower variance than the raw log-ratio mean.
        log_ratio = log_probs - old_log_probs
        approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
        clip_fraction = (torch.abs(ratio - 1.0) > cfg.clip_eps).float().mean().item()
      metrics["loss/policy"].append(policy_loss.item())
      metrics["loss/value"].append(value_loss.item())
      metrics["loss/entropy"].append(entropy_bonus.item())
      metrics["stats/kl"].append(approx_kl)
      metrics["stats/clip_fraction"].append(clip_fraction)

      if approx_kl > kl_cutoff:
        kl_exceeded = True
        break

  return {key: float(np.mean(values)) for key, values in metrics.items()}


# --------------------------------------------------------------------------- #
# Export / checkpoint.
# --------------------------------------------------------------------------- #
def export_onnx(
  policy: RoutinePolicy,
  path: Path,
  obs_shapes: dict[str, tuple[int, ...]],
  device: torch.device,
) -> None:
  """Export the deterministic actor branch to an ONNX file."""
  actor = ExportActor(policy).to(device).eval()
  dummy = (
    torch.zeros((1, *obs_shapes["proprio"]), device=device),
    torch.zeros((1, *obs_shapes["head"]), device=device),
    torch.zeros((1, *obs_shapes["wrist"]), device=device),
  )
  torch.onnx.export(
    actor,
    dummy,
    path,
    input_names=["proprio", "head", "wrist"],
    output_names=["action"],
    opset_version=18,
  )


def save_checkpoint(
  path: Path,
  policy: RoutinePolicy,
  optimizer: torch.optim.Optimizer,
  update_idx: int,
  cfg: "TrainingConfig",
  obs_shapes: dict[str, tuple[int, ...]],
  env_cfg: EnvConfig,
) -> None:
  """Persist training state, model weights, and config for later resume."""
  payload = {
    "model": policy.state_dict(),
    "optimizer": optimizer.state_dict(),
    "update_idx": update_idx,
    "training_config": asdict(cfg),
    "obs_shapes": obs_shapes,
    "env_config": asdict(env_cfg),
  }
  torch.save(payload, path)


def load_checkpoint(
  path: Path, policy: RoutinePolicy, optimizer: torch.optim.Optimizer | None = None
) -> int:
  """Load model state and optionally optimizer state from a checkpoint."""
  payload = torch.load(path, map_location="cpu", weights_only=False)
  policy.load_state_dict(payload["model"])
  if optimizer is not None and "optimizer" in payload:
    optimizer.load_state_dict(payload["optimizer"])
  return int(payload.get("update_idx", 0))


# --------------------------------------------------------------------------- #
# Training config and top-level entry point.
# --------------------------------------------------------------------------- #
@dataclass
class TrainingConfig:
  """Hyperparameters and I/O settings for PPO training and demo warm-start."""

  num_envs: int = 8
  horizon: int = 256
  updates: int = 200
  minibatch_size: int = 256
  update_epochs: int = 4
  learning_rate: float = 3e-4
  gamma: float = 0.995
  gae_lambda: float = 0.95
  clip_eps: float = 0.2
  value_clip_eps: float = 0.2
  vf_coef: float = 0.5
  ent_coef: float = 0.01
  max_grad_norm: float = 1.0
  # Early-stop PPO epochs when the minibatch KL exceeds 1.5× this value.
  # Prevents the BC-pretrained mean from being shredded by large policy steps.
  target_kl: float = 0.03
  seed: int = 0
  device: str = "cpu"
  save_dir: Path = field(default_factory=lambda: SCRIPT_DIR / "artifacts" / "trained")
  save_every: int = 10
  resume: Path | None = None
  # Teacher warm-start (ignored when teacher_factory is None):
  demo_enabled: bool = False
  demo_updates: int = 16
  demo_collect_successes: int = 8
  demo_max_episodes: int = 24
  demo_bc_epochs: int = 32
  demo_seeds: list[int] = field(default_factory=lambda: [0])
  demo_repeat_per_seed: int = 8
  demo_aux_weight: float = 0.1
  demo_aux_decay: float = 0.95
  demo_aux_min_weight: float = 0.01
  demo_reset_prob: float = 0.25
  train_reset_seed: int | None = None
  success_window: int = 20
  stop_success: float = 0.7

  @staticmethod
  def from_namespace(args: argparse.Namespace, save_dir: Path | None = None) -> "TrainingConfig":
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return TrainingConfig(
      num_envs=args.num_envs,
      horizon=args.horizon,
      updates=args.updates,
      minibatch_size=args.minibatch_size,
      update_epochs=args.update_epochs,
      learning_rate=args.learning_rate,
      gamma=args.gamma,
      gae_lambda=args.gae_lambda,
      clip_eps=args.clip_eps,
      value_clip_eps=args.value_clip_eps,
      vf_coef=args.vf_coef,
      ent_coef=args.ent_coef,
      max_grad_norm=args.max_grad_norm,
      seed=args.seed,
      device=device,
      save_dir=Path(save_dir) if save_dir is not None else Path(args.save_dir),
      save_every=args.save_every,
      resume=args.resume,
      demo_enabled=args.demo,
      demo_updates=args.demo_updates,
      demo_collect_successes=args.demo_collect_successes,
      demo_max_episodes=args.demo_max_episodes,
      demo_bc_epochs=args.demo_bc_epochs,
      demo_seeds=parse_seed_list(args.demo_seeds),
      demo_repeat_per_seed=args.demo_repeat_per_seed,
      demo_aux_weight=args.demo_aux_weight,
      demo_aux_decay=args.demo_aux_decay,
      demo_aux_min_weight=args.demo_aux_min_weight,
      demo_reset_prob=args.demo_reset_prob,
      train_reset_seed=args.train_reset_seed,
      success_window=args.success_window,
      stop_success=args.stop_success,
    )


def _demo_aux_weight(cfg: TrainingConfig, update_idx: int) -> float:
  """Return the current BC auxiliary-loss weight for this PPO update."""
  if not cfg.demo_enabled:
    return 0.0
  if update_idx < cfg.demo_updates:
    return cfg.demo_aux_weight
  decay_steps = update_idx - cfg.demo_updates + 1
  return max(cfg.demo_aux_min_weight, cfg.demo_aux_weight * (cfg.demo_aux_decay ** decay_steps))


def _demo_reset_probability(cfg: TrainingConfig, update_idx: int) -> float:
  """Return the current probability of resetting from a demo state."""
  if not cfg.demo_enabled or cfg.demo_reset_prob <= 0.0:
    return 0.0
  if update_idx < cfg.demo_updates:
    return cfg.demo_reset_prob
  return 0.0


def _should_stop_on_success(
  stats: dict[str, deque], cfg: TrainingConfig, update_idx: int
) -> bool:
  """Check whether rolling success rate has reached the early-stop target."""
  if cfg.demo_enabled and update_idx < cfg.demo_updates:
    return False
  return (
    len(stats["episode/success"]) >= cfg.success_window
    and np.mean(stats["episode/success"]) >= cfg.stop_success
  )


def _print_update(
  update_idx: int,
  total_steps: int,
  start_time: float,
  stats: dict[str, deque],
  train_metrics: dict[str, float],
) -> None:
  """Print a compact one-line summary of the latest PPO update."""
  fps = int(total_steps / max(1e-6, time.time() - start_time))
  parts = [f"update={update_idx}", f"steps={total_steps}", f"fps={fps}"]
  for key in ("episode/return", "episode/length", "episode/success"):
    if stats[key]:
      parts.append(f"{key.split('/')[-1]}={np.mean(stats[key]):.3f}")
  for key in ("reward/cylinder_proximity", "reward/lift", "reward/blue_table", "reward/stage"):
    if stats[key]:
      parts.append(f"{key.split('/')[-1]}={np.mean(stats[key]):+.3f}")
  parts.extend(
    [
      f"pi={train_metrics.get('loss/policy', 0.0):+.3f}",
      f"v={train_metrics.get('loss/value', 0.0):+.3f}",
      f"kl={train_metrics.get('stats/kl', 0.0):.4f}",
    ]
  )
  if "loss/demo_bc" in train_metrics:
    parts.append(f"bc={train_metrics['loss/demo_bc']:+.4f}")
  if "loss/demo_aux" in train_metrics:
    parts.append(f"demo_aux={train_metrics['loss/demo_aux']:+.4f}")
  print(" ".join(parts), flush=True)


def run(
  training_cfg: TrainingConfig,
  env_cfg: EnvConfig,
  env_factory: str,
  teacher_factory: str | None = None,
  stage_weights: dict[str, float] | None = None,
) -> None:
  """Agnostic training entry point.

  Args:
    training_cfg: PPO/BC hyperparameters and I/O paths.
    env_cfg: environment configuration passed through to the env factory.
    env_factory: "module:attr" spec for the env factory.
    teacher_factory: optional "module:attr" spec for the teacher factory. If
      None, PPO learns from the reward function alone.
    stage_weights: optional dict mapping stage label -> sampling weight, used
      when `demo_reset_prob > 0` to bias demo-state restarts.
  """
  if training_cfg.demo_enabled and teacher_factory is None:
    raise ValueError("demo_enabled=True but no teacher_factory was provided")
  stage_weights = stage_weights or {}

  torch.manual_seed(training_cfg.seed)
  np.random.seed(training_cfg.seed)
  training_cfg.save_dir.mkdir(parents=True, exist_ok=True)

  envs = SubprocVecEnv(
    num_envs=training_cfg.num_envs,
    env_factory_spec=env_factory,
    env_cfg=env_cfg,
    base_seed=training_cfg.seed,
    fixed_reset_seed=training_cfg.train_reset_seed,
    teacher_factory_spec=teacher_factory if training_cfg.demo_enabled else None,
  )
  obs = envs.reset()
  obs_shapes = {key: value.shape[1:] for key, value in obs.items()}

  device = torch.device(training_cfg.device)
  action_dim = int(envs.num_envs and 7)  # env exposes action_dim via class attr; see below
  # The env's action_dim is a class constant on G1PickPlaceEnv; import lazily
  # so train.py doesn't import the env module itself.
  from sim import G1PickPlaceEnv
  action_dim = G1PickPlaceEnv.action_dim

  policy = RoutinePolicy(
    proprio_dim=obs_shapes["proprio"][0],
    image_channels=obs_shapes["head"][0],
    action_dim=action_dim,
  ).to(device)
  optimizer = torch.optim.Adam(policy.parameters(), lr=training_cfg.learning_rate, eps=1e-5)

  start_update = 0
  if training_cfg.resume and training_cfg.resume.exists():
    start_update = load_checkpoint(training_cfg.resume, policy, optimizer)
    print(f"resumed from {training_cfg.resume} at update {start_update}", flush=True)

  buffer = RolloutBuffer(training_cfg.horizon, training_cfg.num_envs, obs, action_dim)
  stats: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
  start_time = time.time()
  total_steps = start_update * training_cfg.horizon * training_cfg.num_envs
  demo_dataset: DemoDataset | None = None
  active_demo_reset_prob: float | None = None

  try:
    if training_cfg.demo_enabled:
      demo_dataset = collect_demo_dataset(
        env_factory,
        teacher_factory,
        env_cfg,
        stage_weights,
        training_cfg.demo_collect_successes,
        training_cfg.demo_max_episodes,
        training_cfg.demo_seeds,
        training_cfg.demo_repeat_per_seed,
      )
      demo_metrics, demo_steps = pretrain_from_demo_dataset(
        policy, optimizer, device, demo_dataset, training_cfg
      )
      total_steps += demo_steps
      print(
        "demo-pretrain "
        f"episodes={demo_metrics.get('demo/episodes', 0.0):.0f} "
        f"successes={demo_metrics.get('demo/successes', 0.0):.0f} "
        f"success_rate={demo_metrics.get('demo/success_rate', 0.0):.3f} "
        f"reset_states={demo_metrics.get('demo/reset_states', 0.0):.0f} "
        f"bc={demo_metrics.get('loss/demo_bc', 0.0):+.4f} "
        f"seeds={training_cfg.demo_seeds}",
        flush=True,
      )
      envs.set_demo_resets(demo_dataset.reset_states, demo_dataset.reset_weights)
      save_checkpoint(
        training_cfg.save_dir / "latest.pt", policy, optimizer, start_update,
        training_cfg, obs_shapes, env_cfg,
      )
      export_onnx(policy, training_cfg.save_dir / "routine.onnx", obs_shapes, device)

    for update_idx in range(start_update, training_cfg.updates):
      next_demo_reset_prob = _demo_reset_probability(training_cfg, update_idx)
      if next_demo_reset_prob != active_demo_reset_prob:
        envs.set_demo_reset_probability(next_demo_reset_prob)
        active_demo_reset_prob = next_demo_reset_prob
      for t in range(training_cfg.horizon):
        for key in buffer.obs:
          buffer.obs[key][t] = obs[key]
        with torch.no_grad():
          obs_t = _obs_to_torch(obs, device)
          action_t, pretanh_t, log_prob_t, value_t = policy.act(obs_t, deterministic=False)
          actions = action_t.cpu().numpy()
        next_obs, rewards, dones, infos = envs.step(actions)

        buffer.actions[t] = actions
        buffer.pretanh[t] = pretanh_t.cpu().numpy()
        buffer.log_probs[t] = log_prob_t.cpu().numpy()
        buffer.values[t] = value_t.cpu().numpy()
        buffer.rewards[t] = rewards
        buffer.dones[t] = dones.astype(np.float32)
        obs = next_obs
        total_steps += training_cfg.num_envs

        for info in infos:
          if "reward_terms" in info:
            for key, value in info["reward_terms"].items():
              stats[f"reward/{key}"].append(float(value))
          if "episode" in info:
            ep = info["episode"]
            stats["episode/return"].append(float(ep["return"]))
            stats["episode/length"].append(float(ep["length"]))
            stats["episode/success"].append(float(ep["success"]))

      with torch.no_grad():
        last_values = policy.forward(_obs_to_torch(obs, device))[1].cpu().numpy()
      buffer.advantages[:], buffer.returns[:] = compute_gae(
        buffer.rewards,
        buffer.dones,
        buffer.values,
        last_values,
        training_cfg.gamma,
        training_cfg.gae_lambda,
      )

      train_metrics = ppo_update(
        policy,
        optimizer,
        buffer,
        device,
        training_cfg,
        demo_dataset=demo_dataset,
        demo_aux_weight=_demo_aux_weight(training_cfg, update_idx),
      )
      _print_update(update_idx + 1, total_steps, start_time, stats, train_metrics)

      success_ready = _should_stop_on_success(stats, training_cfg, update_idx + 1)
      if (update_idx + 1) % training_cfg.save_every == 0 or update_idx + 1 == training_cfg.updates or success_ready:
        ckpt_path = training_cfg.save_dir / f"checkpoint_{update_idx + 1:05d}.pt"
        latest_path = training_cfg.save_dir / "latest.pt"
        onnx_path = training_cfg.save_dir / "routine.onnx"
        save_checkpoint(ckpt_path, policy, optimizer, update_idx + 1, training_cfg, obs_shapes, env_cfg)
        save_checkpoint(latest_path, policy, optimizer, update_idx + 1, training_cfg, obs_shapes, env_cfg)
        export_onnx(policy, onnx_path, obs_shapes, device)
        print(f"saved {ckpt_path.name} and routine.onnx", flush=True)
      if success_ready:
        print(
          f"stopping early: mean success over last {training_cfg.success_window} episodes = "
          f"{np.mean(stats['episode/success']):.3f}",
          flush=True,
        )
        break
  finally:
    envs.close()
