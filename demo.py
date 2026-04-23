#!/usr/bin/env python3
"""Pick-and-place task definition and training driver.

`demo.py` owns the task-specific pieces that `train.py` is agnostic to:

  * Reward/penalty shaping (`RewardConfig`, `compute_reward_terms`).
  * Curriculum stage labeling (`STAGES`, `current_stage`).
  * The scripted teacher that exists purely to reduce training time — with
    enough compute, PPO + the reward function alone would solve the task.
  * A training CLI that wires those pieces into the `train.py` framework and
    produces `routine.onnx`.

It also keeps the original scripted-rollout CLI for live demos and video capture.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from mujoco import viewer

from run import CameraRenderer, G1Controller
from sim import EnvConfig, G1PickPlaceEnv, _configure_wide_camera, _print_status


# --------------------------------------------------------------------------- #
# Reward / curriculum definition (formerly curriculum.py).
# --------------------------------------------------------------------------- #
STAGES = ("approach", "grasp", "lift", "transport", "place")


@dataclass
class RewardConfig:
  """Task-specific weights for dense reward shaping terms."""

  move_speed: float = 0.05
  source_progress: float = 1.25
  cylinder_progress: float = 2.0
  touch_bonus: float = 0.15
  grasp_bonus: float = 0.4
  lift_progress: float = 4.0
  transport_progress: float = 3.5
  place_bonus: float = 8.0
  release_bonus: float = 18.0
  time_penalty: float = 0.02
  speed_bonus: float = 4.0
  drop_penalty: float = 3.0
  fall_penalty: float = 8.0


@dataclass
class CurriculumConfig:
  """Thresholds used to label task stages and stage-transition bonuses."""

  lift_height: float = 0.04
  transport_distance: float = 0.45
  stage_bonus: float = 0.3


# Weights over stages for biasing demo-reset state sampling toward later phases
# (only used when the trainer's `--demo-reset-prob` > 0).
STAGE_WEIGHTS: dict[str, float] = {
  "approach": 0.0,
  "grasp": 1.0,
  "lift": 2.0,
  "transport": 3.0,
  "place": 4.0,
}


def current_stage(metrics: dict[str, Any], cfg: CurriculumConfig) -> str:
  """Return the current curriculum stage from measured simulator metrics."""
  if metrics["placed"]:
    return "place"
  if metrics["lift_height"] > cfg.lift_height:
    if metrics["block_to_target"] < cfg.transport_distance:
      return "transport"
    return "lift"
  if metrics["touching"] or metrics["grasp_score"] > 0.35:
    return "grasp"
  return "approach"


def compute_reward_terms(
  prev: dict[str, Any],
  current: dict[str, Any],
  reward_cfg: RewardConfig,
  curriculum_cfg: CurriculumConfig,
  step_count: int,
  max_steps: int,
  training: bool,
) -> tuple[dict[str, float], str]:
  """Compute dense reward terms and the resulting curriculum stage label."""
  terms = {
    "movement": reward_cfg.move_speed * min(1.0, float(current["lin_vel"][:2].dot(current["lin_vel"][:2]) ** 0.5)),
    "brown_table": reward_cfg.source_progress * (prev["base_to_source"] - current["base_to_source"]),
    "cylinder_proximity": reward_cfg.cylinder_progress * (prev["palm_to_block"] - current["palm_to_block"]),
    "touch": reward_cfg.touch_bonus * float(current["touching"]),
    "grasp": reward_cfg.grasp_bonus * current["grasp_score"],
    "lift": reward_cfg.lift_progress * max(0.0, current["lift_height"] - prev["lift_height"]),
    "place": reward_cfg.place_bonus * float(current["placed"]),
    "time": -reward_cfg.time_penalty,
  }

  carrying = max(current["grasp_score"], float(current["lift_height"] > curriculum_cfg.lift_height))
  terms["blue_table"] = (
    reward_cfg.transport_progress
    * (prev["block_to_target"] - current["block_to_target"])
    * carrying
  )

  if current["placed"] and not prev["placed"]:
    terms["release"] = reward_cfg.release_bonus
    terms["speed"] = reward_cfg.speed_bonus * (1.0 - step_count / max_steps)
  if current["dropped"] and prev["lift_height"] > curriculum_cfg.lift_height:
    terms["drop"] = -reward_cfg.drop_penalty
  if current["fallen"]:
    terms["fall"] = -reward_cfg.fall_penalty

  stage = current_stage(current, curriculum_cfg)
  if training:
    prev_stage = current_stage(prev, curriculum_cfg)
    stage_gain = STAGES.index(stage) - STAGES.index(prev_stage)
    if stage_gain > 0:
      terms["stage"] = curriculum_cfg.stage_bonus * stage_gain
  return terms, stage


# --------------------------------------------------------------------------- #
# Module-level reward bindings: these must be importable by name so the
# spawn-based SubprocVecEnv workers can pickle the `env_fn` that references
# them. Training CLI updates `_active_reward_cfg` / `_active_curriculum_cfg`
# before launching workers.
# --------------------------------------------------------------------------- #
_active_reward_cfg = RewardConfig()
_active_curriculum_cfg = CurriculumConfig()


def reward_fn(
  prev: dict[str, Any],
  current: dict[str, Any],
  step_count: int,
  max_steps: int,
  training: bool,
) -> tuple[dict[str, float], str]:
  """Module-level reward binding used by subprocess-safe env factories."""
  return compute_reward_terms(
    prev, current, _active_reward_cfg, _active_curriculum_cfg, step_count, max_steps, training
  )


def build_env_config(
  image_size: int,
  max_steps: int,
  block_spawn_x: tuple[float, float] | None = None,
  block_spawn_y: tuple[float, float] | None = None,
) -> EnvConfig:
  """Build the environment config used by demo and training entry points."""
  cfg = EnvConfig(image_size=(image_size, image_size), max_steps=max_steps)
  if block_spawn_x is not None:
    cfg.block_spawn_x = block_spawn_x
  if block_spawn_y is not None:
    cfg.block_spawn_y = block_spawn_y
  return cfg


def _spawn_range_from_args(args: argparse.Namespace) -> tuple[
  tuple[float, float] | None, tuple[float, float] | None
]:
  """Parse optional CLI spawn-range overrides for the block position."""
  x = None
  if args.block_spawn_x_min is not None and args.block_spawn_x_max is not None:
    x = (args.block_spawn_x_min, args.block_spawn_x_max)
  y = None
  if args.block_spawn_y_min is not None and args.block_spawn_y_max is not None:
    y = (args.block_spawn_y_min, args.block_spawn_y_max)
  return x, y


def make_env(
  env_config: EnvConfig,
  seed: int,
  render_images: bool = True,
  verbose: bool = False,
  training: bool = False,
) -> G1PickPlaceEnv:
  """Pickable env factory used by workers and the training CLI."""
  return G1PickPlaceEnv(
    env_config=env_config,
    seed=seed,
    render_images=render_images,
    verbose=verbose,
    training=training,
    reward_fn=reward_fn,
  )


# --------------------------------------------------------------------------- #
# Scripted teacher.
# --------------------------------------------------------------------------- #
@dataclass
class DemoScriptConfig:
  """Hardcoded teacher routine over the high-level action space."""

  target_standoff_xy: tuple[float, float] = (0.40, -0.22)
  pregrasp_offset: tuple[float, float, float] = (0.02, 0.0, 0.11)
  grasp_offset: tuple[float, float, float] = (0.0, 0.0, -0.005)
  lift_offset: tuple[float, float, float] = (0.0, 0.0, 0.22)
  carry_pose: tuple[float, float, float] = (0.22, -0.20, 0.30)
  place_hover_offset: tuple[float, float, float] = (0.02, 0.0, 0.12)
  place_drop_offset: tuple[float, float, float] = (0.0, 0.0, 0.03)
  reach_gain: float = 12.0
  close_grasp_score: float = 0.55
  lift_height: float = 0.08
  target_distance: float = 0.16
  head_area_near: float = 0.24
  wrist_area_visible: float = 0.015
  wrist_area_grasp: float = 0.05
  head_center_gain: float = 2.0
  wrist_center_gain: float = 2.5


class ScriptedDemoPolicy:
  """Finite-state teacher over the routine action space.

  The teacher exists only to shortcut the otherwise-long random-exploration
  phase of PPO. With enough training time the reward function alone is
  sufficient.

  Usage contract (the agnostic `train.py` framework relies on this):
      teacher.reset() -> None
      teacher.act(env, obs) -> np.ndarray[shape=(action_dim,)]
  """

  def __init__(self, env_cfg: EnvConfig, script_cfg: DemoScriptConfig | None = None):
    self.env_cfg = env_cfg
    self.cfg = script_cfg or DemoScriptConfig()
    self.reset()

  def reset(self) -> None:
    """Reset the teacher state at the start of each episode."""
    self.phase = "approach"

  def _rel(self, base_pos: np.ndarray, base_quat: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Express a world-space point in the robot base frame."""
    return G1Controller._quat_apply_inverse(base_quat, point - base_pos)

  def _reach_cmd(self, current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """Convert a reach-target delta into normalized high-level actions."""
    delta = self.cfg.reach_gain * (desired - current) / self.env_cfg.reach_delta
    return np.clip(delta, -1.0, 1.0)

  def _detect_red(self, image: np.ndarray) -> dict[str, float] | None:
    """Estimate the red object centroid and visible area from a CHW image."""
    rgb = np.transpose(image, (1, 2, 0))
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (r > 0.35) & (r > g * 1.4) & (r > b * 1.4)
    ys, xs = np.where(mask)
    if len(xs) == 0:
      return None
    height, width = rgb.shape[:2]
    return {
      "cx": float(xs.mean() / width),
      "cy": float(ys.mean() / height),
      "area": float(mask.mean()),
    }

  def act(self, env: G1PickPlaceEnv, obs: dict[str, np.ndarray]) -> np.ndarray:
    metrics = env.prev_metrics
    reach_target = env.controller.reach_target.copy()
    grip_closed = env.controller.grip_closed

    base_pos = metrics["base_pos"]
    base_quat = metrics["base_quat"]
    block_rel = self._rel(base_pos, base_quat, metrics["block_pos"])
    target_rel = self._rel(base_pos, base_quat, np.asarray(self.env_cfg.target_center, dtype=np.float32))
    head_red = self._detect_red(obs["head"])
    wrist_red = self._detect_red(obs["wrist"])
    grasped = metrics["grasp_score"] > self.cfg.close_grasp_score
    lifted = metrics["lift_height"] > self.cfg.lift_height

    if metrics["placed"]:
      self.phase = "release"
    elif self.phase == "approach" and (
      (
        metrics["base_to_source"] < 0.72
        and (
          (wrist_red is not None and wrist_red["area"] > self.cfg.wrist_area_visible)
          or (head_red is not None and head_red["area"] > self.cfg.head_area_near)
        )
      )
      or (block_rel[0] < 0.34 and abs(block_rel[1]) < 0.14)
    ):
      self.phase = "pregrasp"
    elif self.phase == "pregrasp" and metrics["palm_to_block"] < 0.10:
      self.phase = "grasp"
    elif self.phase == "grasp" and grasped and metrics["palm_to_block"] < 0.07:
      self.phase = "lift"
    elif self.phase == "lift" and lifted:
      self.phase = "transport"
    elif self.phase == "lift" and metrics["grasp_score"] < 0.2 and metrics["palm_to_block"] > 0.09:
      # Recovery: grasp slipped before we could lift — fall back to pregrasp.
      self.phase = "pregrasp"
    elif self.phase == "transport" and (
      np.linalg.norm(target_rel[:2] - np.asarray(self.cfg.target_standoff_xy)) < self.cfg.target_distance
      or (metrics["block_to_target"] < 0.35 and -0.10 < target_rel[0] < 0.25)
    ):
      self.phase = "place"
    elif self.phase == "place" and metrics["target_support"]:
      self.phase = "release"

    action = np.zeros(7, dtype=np.float32)

    if self.phase == "approach":
      # Yaw toward the block using true relative pose — vision-only centering
      # failed on ~94% of random cube spawns because the block leaves the head
      # FOV before the state machine can reach pregrasp. Yaw is the primary
      # alignment channel; strafe (action[1]) is a fine-tune after vision locks.
      heading_err = float(np.arctan2(block_rel[1], max(block_rel[0], 0.05)))
      if abs(heading_err) > 0.05:
        action[2] = float(np.clip(-1.4 * heading_err, -1.0, 1.0))
      if block_rel[0] > 0.30 or metrics["base_to_source"] > 0.75:
        action[0] = 1.0
      else:
        action[0] = 0.4
      if head_red is not None:
        x_err = head_red["cx"] - 0.5
        action[1] = float(np.clip(-self.cfg.head_center_gain * x_err, -0.6, 0.6))
      desired_reach = np.asarray(self.cfg.carry_pose, dtype=np.float32)
      action[3:6] = self._reach_cmd(reach_target, desired_reach)
      action[6] = -1.0
      return action

    if self.phase == "pregrasp":
      if head_red is not None:
        action[1] = float(np.clip(-self.cfg.head_center_gain * (head_red["cx"] - 0.5), -0.8, 0.8))
      if metrics["palm_to_block"] > 0.18:
        action[0] = 0.25
      if wrist_red is not None:
        action[1] += float(np.clip(-3.0 * (wrist_red["cx"] - 0.5), -0.6, 0.6))
        action[0] = max(action[0], 0.2 if wrist_red["cy"] < 0.30 else 0.0)
      desired_reach = block_rel + np.asarray(self.cfg.pregrasp_offset, dtype=np.float32)
      if metrics["palm_to_block"] < 0.24:
        desired_reach = block_rel + np.array([0.01, 0.0, 0.06], dtype=np.float32)
      action[3:6] = self._reach_cmd(reach_target, desired_reach)
      action[6] = -1.0
      return action

    if self.phase == "grasp":
      desired_reach = block_rel + np.asarray(self.cfg.grasp_offset, dtype=np.float32)
      if wrist_red is not None:
        desired_reach[1] += 0.08 * (wrist_red["cx"] - 0.5)
        desired_reach[2] += 0.10 * (0.58 - wrist_red["cy"])
      action[3:6] = self._reach_cmd(reach_target, desired_reach)
      action[6] = 1.0
      return action

    if self.phase == "lift":
      desired_reach = block_rel + np.asarray(self.cfg.lift_offset, dtype=np.float32)
      action[3:6] = self._reach_cmd(reach_target, desired_reach)
      action[6] = 1.0
      return action

    if self.phase == "transport":
      if target_rel[1] > self.cfg.target_standoff_xy[1] + 0.12:
        action[1] = 1.0
      elif target_rel[1] > self.cfg.target_standoff_xy[1] + 0.05:
        action[1] = 0.6
      elif target_rel[1] < self.cfg.target_standoff_xy[1] - 0.12:
        action[1] = -1.0
      elif target_rel[1] < self.cfg.target_standoff_xy[1] - 0.05:
        action[1] = -0.6
      if target_rel[0] > self.cfg.target_standoff_xy[0] + 0.12:
        action[0] = 0.9
      elif target_rel[0] < self.cfg.target_standoff_xy[0] - 0.08:
        action[0] = -0.6
      desired_reach = np.asarray(self.cfg.carry_pose, dtype=np.float32)
      action[3:6] = self._reach_cmd(reach_target, desired_reach)
      action[6] = 1.0
      return action

    if self.phase == "place":
      desired_reach = target_rel + np.asarray(self.cfg.place_hover_offset, dtype=np.float32)
      if metrics["block_to_target"] < 0.18:
        desired_reach = target_rel + np.asarray(self.cfg.place_drop_offset, dtype=np.float32)
      action[1] = float(np.clip(-2.0 * (target_rel[1] - self.cfg.target_standoff_xy[1]), -0.5, 0.5))
      action[3:6] = self._reach_cmd(reach_target, desired_reach)
      action[6] = 1.0
      return action

    if self.phase == "release":
      desired_reach = target_rel + np.asarray(self.cfg.place_hover_offset, dtype=np.float32)
      action[3:6] = self._reach_cmd(reach_target, desired_reach)
      action[6] = -1.0
      if not grip_closed:
        action[0] = -0.15
      return action

    return action


def make_teacher(env_cfg: EnvConfig) -> ScriptedDemoPolicy:
  """Pickable teacher factory."""
  return ScriptedDemoPolicy(env_cfg)


# --------------------------------------------------------------------------- #
# Scripted-rollout CLI (live demo + video capture).
# --------------------------------------------------------------------------- #
def _compose_demo_frame(renderer: CameraRenderer) -> np.ndarray:
  """Render a 2x2 tiled frame for demo recordings."""
  overhead = renderer.render("overhead")
  side = renderer.render("side_view")
  head = renderer.render("head_cam")
  wrist = renderer.render("wrist_cam")
  top = np.concatenate([overhead, side], axis=1)
  bottom = np.concatenate([head, wrist], axis=1)
  return np.concatenate([top, bottom], axis=0)


def _capture_frame(
  writer: imageio.Writer | None,
  renderer: CameraRenderer | None,
  next_capture_time: float,
  sim_time: float,
  record_fps: int,
) -> float:
  """Capture one frame once simulated time reaches the next sample boundary."""
  if writer is None or renderer is None or sim_time + 1e-9 < next_capture_time:
    return next_capture_time
  writer.append_data(_compose_demo_frame(renderer))
  return next_capture_time + 1.0 / record_fps


def run_scripted_demo(args: argparse.Namespace) -> None:
  """Run the scripted teacher headlessly, in the viewer, or with recording."""
  spawn_x, spawn_y = _spawn_range_from_args(args)
  env = make_env(
    build_env_config(args.image_size, args.max_steps, spawn_x, spawn_y),
    seed=args.seed,
    render_images=True,
    verbose=True,
    training=False,
  )
  policy = ScriptedDemoPolicy(env.cfg)
  record_path = args.record.resolve() if args.record is not None else None
  writer: imageio.Writer | None = None
  record_renderer: CameraRenderer | None = None

  if record_path is not None:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(record_path), fps=args.record_fps, codec="libx264", quality=8)
    record_renderer = CameraRenderer(env.model, env.data, width=320, height=240)

  try:
    if args.viewer:
      print("Launching viewer. On macOS, use `mjpython demo.py run --viewer ...`.")
      with viewer.launch_passive(env.model, env.data) as v:
        _configure_wide_camera(v.cam)
        deadline = time.perf_counter()
        for episode in range(args.episodes):
          obs = env.reset(seed=args.seed + episode)
          policy.reset()
          done = False
          last_info: dict[str, Any] = {}
          sim_time = 0.0
          next_capture_time = 0.0
          next_capture_time = _capture_frame(
            writer, record_renderer, next_capture_time, sim_time, args.record_fps
          )
          while v.is_running() and not done:
            action = policy.act(env, obs)
            obs, reward, done, info = env.step(action)
            _print_status(env.step_count, reward, info)
            last_info = info
            sim_time += env.control_dt
            next_capture_time = _capture_frame(
              writer, record_renderer, next_capture_time, sim_time, args.record_fps
            )
            v.sync()
            deadline += env.control_dt
            time.sleep(max(0.0, deadline - time.perf_counter()))
          print(f"episode={episode} phase={policy.phase} summary={last_info.get('episode', {})}")
          if not v.is_running():
            break
    else:
      for episode in range(args.episodes):
        obs = env.reset(seed=args.seed + episode)
        policy.reset()
        done = False
        last_info: dict[str, Any] = {}
        sim_time = 0.0
        next_capture_time = 0.0
        next_capture_time = _capture_frame(
          writer, record_renderer, next_capture_time, sim_time, args.record_fps
        )
        while not done:
          action = policy.act(env, obs)
          obs, reward, done, info = env.step(action)
          _print_status(env.step_count, reward, info)
          last_info = info
          sim_time += env.control_dt
          next_capture_time = _capture_frame(
            writer, record_renderer, next_capture_time, sim_time, args.record_fps
          )
        print(f"episode={episode} phase={policy.phase} summary={last_info.get('episode', {})}")
  finally:
    if writer is not None:
      writer.close()
      print(f"saved_video={record_path}")


# --------------------------------------------------------------------------- #
# Training CLI: thin wrapper over the agnostic `train.py` framework.
# --------------------------------------------------------------------------- #
def run_training(args: argparse.Namespace) -> None:
  """Translate task-specific CLI args into the generic training entry point."""
  # Lazy import so non-training CLIs don't pay the torch import cost.
  from train import TrainingConfig, run as run_trainer

  global _active_reward_cfg, _active_curriculum_cfg
  _active_reward_cfg = RewardConfig()
  _active_curriculum_cfg = CurriculumConfig()

  spawn_x, spawn_y = _spawn_range_from_args(args)
  env_cfg = build_env_config(args.image_size, args.max_steps, spawn_x, spawn_y)
  training_cfg = TrainingConfig.from_namespace(args, save_dir=args.save_dir)
  run_trainer(
    training_cfg=training_cfg,
    env_cfg=env_cfg,
    env_factory="demo:make_env",
    teacher_factory="demo:make_teacher" if args.demo else None,
    stage_weights=STAGE_WEIGHTS,
  )


# --------------------------------------------------------------------------- #
# Argparse plumbing.
# --------------------------------------------------------------------------- #
def _add_run_args(parser: argparse.ArgumentParser) -> None:
  """Register CLI flags for scripted demo rollouts and recording."""
  parser.add_argument("--episodes", type=int, default=1)
  parser.add_argument("--image-size", type=int, default=64)
  parser.add_argument("--max-steps", type=int, default=700)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--viewer", action="store_true", help="Open the passive MuJoCo viewer during the rollout")
  parser.add_argument("--record", type=Path, default=None, help="Optional MP4 path for a multi-camera recording")
  parser.add_argument("--record-fps", type=int, default=20, help="Recording frame rate when --record is set")
  parser.add_argument("--block-spawn-x-min", type=float, default=None)
  parser.add_argument("--block-spawn-x-max", type=float, default=None)
  parser.add_argument("--block-spawn-y-min", type=float, default=None)
  parser.add_argument("--block-spawn-y-max", type=float, default=None)


def _add_train_args(parser: argparse.ArgumentParser) -> None:
  """Register CLI flags for PPO and optional teacher warm-start training."""
  parser.add_argument("--num-envs", type=int, default=8)
  parser.add_argument("--horizon", type=int, default=256)
  parser.add_argument("--updates", type=int, default=200)
  parser.add_argument("--minibatch-size", type=int, default=256)
  parser.add_argument("--update-epochs", type=int, default=4)
  parser.add_argument("--learning-rate", type=float, default=3e-4)
  parser.add_argument("--gamma", type=float, default=0.995)
  parser.add_argument("--gae-lambda", type=float, default=0.95)
  parser.add_argument("--clip-eps", type=float, default=0.2)
  parser.add_argument("--value-clip-eps", type=float, default=0.2)
  parser.add_argument("--vf-coef", type=float, default=0.5)
  parser.add_argument("--ent-coef", type=float, default=0.01)
  parser.add_argument("--max-grad-norm", type=float, default=1.0)
  parser.add_argument("--image-size", type=int, default=64)
  parser.add_argument("--max-steps", type=int, default=220)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", type=str, default=None, help="Default: cuda if available else cpu")
  parser.add_argument("--save-dir", type=Path, default=Path("artifacts/trained"))
  parser.add_argument("--save-every", type=int, default=10)
  parser.add_argument("--resume", type=Path, default=None)
  parser.add_argument("--demo", action="store_true", help="Warm-start PPO using scripted demo actions")
  parser.add_argument("--demo-updates", type=int, default=16)
  parser.add_argument("--demo-collect-successes", type=int, default=8)
  parser.add_argument("--demo-max-episodes", type=int, default=24)
  parser.add_argument("--demo-bc-epochs", type=int, default=32)
  parser.add_argument("--demo-seeds", type=str, default="0")
  parser.add_argument("--demo-repeat-per-seed", type=int, default=8)
  parser.add_argument("--demo-aux-weight", type=float, default=0.1)
  parser.add_argument("--demo-aux-decay", type=float, default=0.95)
  parser.add_argument("--demo-aux-min-weight", type=float, default=0.01)
  parser.add_argument("--demo-reset-prob", type=float, default=0.25)
  parser.add_argument("--train-reset-seed", type=int, default=None)
  parser.add_argument("--success-window", type=int, default=20)
  parser.add_argument("--stop-success", type=float, default=0.7)
  parser.add_argument("--block-spawn-x-min", type=float, default=None)
  parser.add_argument("--block-spawn-x-max", type=float, default=None)
  parser.add_argument("--block-spawn-y-min", type=float, default=None)
  parser.add_argument("--block-spawn-y-max", type=float, default=None)


def build_parser() -> argparse.ArgumentParser:
  """Build the top-level task CLI with `run` and `train` subcommands."""
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="command", required=True)

  run_p = sub.add_parser("run", help="Run the scripted teacher (viewer/record/headless)")
  _add_run_args(run_p)
  run_p.set_defaults(func=run_scripted_demo)

  train_p = sub.add_parser("train", help="Train routine.onnx with the teacher + reward function")
  _add_train_args(train_p)
  train_p.set_defaults(func=run_training)

  return parser


def main() -> None:
  """CLI entry point for the task driver."""
  args = build_parser().parse_args()
  args.func(args)


if __name__ == "__main__":
  main()
