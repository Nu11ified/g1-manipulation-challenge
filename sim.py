#!/usr/bin/env python3
"""MuJoCo pick-and-place environment and ONNX routine runner.

The routine policy operates at 50 Hz over stable pretrained subskills:
walker for locomotion, right_reacher for arm motion, and direct finger control
for grasp/release. Rewards are computed every control step from simulator state
and contacts rather than only at episode end.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np
import onnxruntime as ort

from run import CameraRenderer, G1Controller, ONNXPolicy, SCRIPT_DIR, set_armature


# Reward signature: (prev_metrics, current_metrics, step_count, max_steps, training) -> (terms, stage)
# The env sums terms.values() to get the scalar reward. Stage is an arbitrary
RewardFn = Callable[[dict, dict, int, int, bool], tuple[dict[str, float], str]]


@dataclass
class EnvConfig:
  image_size: tuple[int, int] = (64, 64)
  control_decimation: int = 4
  max_steps: int = 600
  max_vx: float = 0.7
  max_vy: float = 0.5
  max_yaw: float = 0.8
  reach_delta: float = 0.025
  reach_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (-0.2, 0.55),
    (-0.6, 0.25),
    (-0.1, 0.55),
  )
  source_center: tuple[float, float, float] = (0.351, 0.0, 0.713)
  source_half_extents: tuple[float, float, float] = (0.4, 0.25, 0.02)
  target_center: tuple[float, float, float] = (-0.3, -0.8, 0.613)
  target_half_extents: tuple[float, float, float] = (0.35, 0.25, 0.02)
  block_radius: float = 0.02
  block_half_height: float = 0.035
  block_spawn_x: tuple[float, float] = (-0.02, 0.18)
  block_spawn_y: tuple[float, float] = (-0.1, 0.1)


class RoutineONNXPolicy:
  """Inference wrapper for the exported routine.onnx policy."""

  def __init__(self, model_path: str | Path):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    self.session = ort.InferenceSession(
      str(model_path), sess_options, providers=["CPUExecutionProvider"]
    )
    self.input_names = [node.name for node in self.session.get_inputs()]
    self.output_name = self.session.get_outputs()[0].name

  def __call__(self, obs: dict[str, np.ndarray]) -> np.ndarray:
    feeds = {}
    for name in self.input_names:
      value = obs[name]
      if value.ndim == len(obs[name].shape):
        value = value[None]
      feeds[name] = value.astype(np.float32, copy=False)
    return self.session.run([self.output_name], feeds)[0][0]


class G1PickPlaceEnv:
  """Headless control environment for PPO training or ONNX policy inference."""

  action_dim = 7

  def __init__(
    self,
    env_config: EnvConfig | None = None,
    seed: int = 0,
    render_images: bool = True,
    verbose: bool = False,
    training: bool = False,
    reward_fn: RewardFn | None = None,
  ):
    self.cfg = env_config or EnvConfig()
    self.rng = np.random.default_rng(seed)
    self.render_images = render_images
    self.verbose = verbose
    self.training = training
    self.reward_fn = reward_fn

    with open(SCRIPT_DIR / "model_config.json") as f:
      self.sim_config = json.load(f)

    self.model = mujoco.MjModel.from_xml_path(str(SCRIPT_DIR / "scene.xml"))
    self.model.opt.timestep = 0.005
    set_armature(self.model, self.sim_config["joint_names"])
    self.data = mujoco.MjData(self.model)

    self.walker = ONNXPolicy(str(SCRIPT_DIR / "walker.onnx"))
    self.croucher = ONNXPolicy(str(SCRIPT_DIR / "croucher.onnx"))
    self.rotator = ONNXPolicy(str(SCRIPT_DIR / "rotator.onnx"))
    self.reacher = ONNXPolicy(str(SCRIPT_DIR / "right_reacher.onnx"))
    if self.verbose:
      self.controller = G1Controller(
        self.model,
        self.data,
        self.walker,
        self.croucher,
        self.rotator,
        self.sim_config,
        right_reacher=self.reacher,
      )
    else:
      with contextlib.redirect_stdout(io.StringIO()):
        self.controller = G1Controller(
          self.model,
          self.data,
          self.walker,
          self.croucher,
          self.rotator,
          self.sim_config,
          right_reacher=self.reacher,
        )
    self.controller.reach_active = True

    self.renderer = None
    if render_images:
      self.renderer = CameraRenderer(
        self.model,
        self.data,
        width=self.cfg.image_size[0],
        height=self.cfg.image_size[1],
      )

    self._cache_ids()
    self.control_dt = self.model.opt.timestep * self.cfg.control_decimation
    self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
    self.episode_return = 0.0
    self.step_count = 0
    self.success_streak = 0
    self.prev_metrics: dict[str, Any] = {}

  @staticmethod
  def _copy_tree(value: Any) -> Any:
    if isinstance(value, np.ndarray):
      return value.copy()
    if isinstance(value, dict):
      return {key: G1PickPlaceEnv._copy_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
      return [G1PickPlaceEnv._copy_tree(item) for item in value]
    if isinstance(value, np.generic):
      return value.item()
    return value

  def _cache_ids(self) -> None:
    name2body = lambda name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
    name2geom = lambda name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
    name2site = lambda name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)

    self.pelvis_body_id = name2body("pelvis")
    self.red_block_body_id = name2body("red_block")
    self.red_block_joint_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_joint"
    )
    self.right_palm_site_id = name2site("right_palm")

    self.red_geom_ids = {
      name2geom("red_cylinder"),
      name2geom("red_cap_top"),
      name2geom("red_cap_bot"),
    }
    self.source_geom_ids = {name2geom("table_top")}
    self.target_geom_ids = {name2geom("table_white_top")}

    hand_names = [
      "right_wrist_pitch_link",
      "right_wrist_yaw_link",
      "right_hand_thumb_0_link",
      "right_hand_thumb_1_link",
      "right_hand_thumb_2_link",
      "right_hand_middle_0_link",
      "right_hand_middle_1_link",
      "right_hand_index_0_link",
      "right_hand_index_1_link",
    ]
    self.hand_body_ids = {name2body(name) for name in hand_names}

    self.robot_body_ids = set()
    for body_id in range(self.model.nbody):
      probe = body_id
      while probe > 0:
        if probe == self.pelvis_body_id:
          self.robot_body_ids.add(body_id)
          break
        probe = int(self.model.body_parentid[probe])

  def _reset_robot_pose(self) -> None:
    mujoco.mj_resetData(self.model, self.data)
    joint_names = self.sim_config["joint_names"]
    self.data.qpos[0] = -0.6
    self.data.qpos[1] = 0.0
    self.data.qpos[2] = 0.76
    self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for name, value in self.sim_config["default_joint_pos"].items():
      if name in joint_names:
        self.data.qpos[7 + joint_names.index(name)] = value

    block_adr = self.model.jnt_qposadr[self.red_block_joint_id]
    source = np.array(self.cfg.source_center, dtype=np.float32)
    z = source[2] + self.cfg.source_half_extents[2] + self.cfg.block_half_height + 0.002
    x = self.rng.uniform(*self.cfg.block_spawn_x)
    y = self.rng.uniform(*self.cfg.block_spawn_y)
    self.data.qpos[block_adr:block_adr + 7] = [x, y, z, 1.0, 0.0, 0.0, 0.0]

    qvel_adr = self.model.jnt_dofadr[self.red_block_joint_id]
    self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0

    mujoco.mj_forward(self.model, self.data)
    self.controller.last_action[:] = 0.0
    self.controller.last_arm_action[:] = 0.0
    self.controller.last_arm_target = None
    self.controller.frozen_arm_pos = None
    self.controller.reach_target[:] = [0.28, -0.2, 0.18]
    self.controller.reach_orientation[:] = 0.0
    self.controller.reach_active = True
    self.controller.grip_closed = False
    self.controller.lin_vel_x = 0.0
    self.controller.lin_vel_y = 0.0
    self.controller.ang_vel_z = 0.0
    self.prev_action[:] = 0.0

  def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
    if seed is not None:
      self.rng = np.random.default_rng(seed)
    self._reset_robot_pose()
    self.step_count = 0
    self.success_streak = 0
    self.episode_return = 0.0
    self.prev_metrics = self._measure()
    return self._get_obs(self.prev_metrics)

  def snapshot(self) -> dict[str, Any]:
    """Capture a serializable simulator + controller state for later restoration."""
    return {
      "qpos": self.data.qpos.copy(),
      "qvel": self.data.qvel.copy(),
      "ctrl": self.data.ctrl.copy(),
      "act": None if self.data.act is None else self.data.act.copy(),
      "time": float(self.data.time),
      "controller": {
        "last_action": self.controller.last_action.copy(),
        "last_arm_action": self.controller.last_arm_action.copy(),
        "last_arm_target": None if self.controller.last_arm_target is None else self.controller.last_arm_target.copy(),
        "frozen_arm_pos": None if self.controller.frozen_arm_pos is None else self.controller.frozen_arm_pos.copy(),
        "reach_target": self.controller.reach_target.copy(),
        "reach_orientation": self.controller.reach_orientation.copy(),
        "reach_active": bool(self.controller.reach_active),
        "grip_closed": bool(self.controller.grip_closed),
        "lin_vel_x": float(self.controller.lin_vel_x),
        "lin_vel_y": float(self.controller.lin_vel_y),
        "ang_vel_z": float(self.controller.ang_vel_z),
      },
      "prev_action": self.prev_action.copy(),
      "step_count": int(self.step_count),
      "success_streak": int(self.success_streak),
      "episode_return": float(self.episode_return),
      "prev_metrics": self._copy_tree(self.prev_metrics),
    }

  def restore(self, snapshot: dict[str, Any]) -> dict[str, np.ndarray]:
    """Restore a previously captured simulator + controller state."""
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[:] = np.asarray(snapshot["qpos"], dtype=np.float64)
    self.data.qvel[:] = np.asarray(snapshot["qvel"], dtype=np.float64)
    self.data.ctrl[:] = np.asarray(snapshot["ctrl"], dtype=np.float64)
    if self.data.act is not None and snapshot.get("act") is not None:
      self.data.act[:] = np.asarray(snapshot["act"], dtype=np.float64)
    self.data.time = float(snapshot.get("time", 0.0))
    mujoco.mj_forward(self.model, self.data)

    controller = snapshot["controller"]
    self.controller.last_action[:] = np.asarray(controller["last_action"], dtype=np.float32)
    self.controller.last_arm_action[:] = np.asarray(controller["last_arm_action"], dtype=np.float32)
    self.controller.last_arm_target = (
      None if controller["last_arm_target"] is None
      else np.asarray(controller["last_arm_target"], dtype=np.float32)
    )
    self.controller.frozen_arm_pos = (
      None if controller["frozen_arm_pos"] is None
      else np.asarray(controller["frozen_arm_pos"], dtype=np.float32)
    )
    self.controller.reach_target[:] = np.asarray(controller["reach_target"], dtype=np.float32)
    self.controller.reach_orientation[:] = np.asarray(controller["reach_orientation"], dtype=np.float32)
    self.controller.reach_active = bool(controller["reach_active"])
    self.controller.grip_closed = bool(controller["grip_closed"])
    self.controller.lin_vel_x = float(controller["lin_vel_x"])
    self.controller.lin_vel_y = float(controller["lin_vel_y"])
    self.controller.ang_vel_z = float(controller["ang_vel_z"])

    self.prev_action[:] = np.asarray(snapshot["prev_action"], dtype=np.float32)
    self.step_count = int(snapshot["step_count"])
    self.success_streak = int(snapshot["success_streak"])
    self.episode_return = float(snapshot["episode_return"])
    self.prev_metrics = self._copy_tree(snapshot["prev_metrics"])
    return self._get_obs(self.prev_metrics)

  def _geom_contact_summary(self) -> dict[str, Any]:
    hand_touch_bodies = set()
    robot_touch = False
    source_support = False
    target_support = False

    for i in range(self.data.ncon):
      contact = self.data.contact[i]
      geom1 = int(contact.geom1)
      geom2 = int(contact.geom2)
      if geom1 in self.red_geom_ids:
        other_geom = geom2
      elif geom2 in self.red_geom_ids:
        other_geom = geom1
      else:
        continue

      other_body = int(self.model.geom_bodyid[other_geom])
      if other_geom in self.source_geom_ids:
        source_support = True
      if other_geom in self.target_geom_ids:
        target_support = True
      if other_body in self.robot_body_ids:
        robot_touch = True
      if other_body in self.hand_body_ids:
        hand_touch_bodies.add(other_body)

    return {
      "hand_touch_bodies": hand_touch_bodies,
      "robot_touch": robot_touch,
      "source_support": source_support,
      "target_support": target_support,
    }

  def _measure(self) -> dict[str, Any]:
    base_pos, base_quat = self.controller._get_base_pose()
    lin_vel, ang_vel = self.controller._get_base_velocities()
    proj_gravity = self.controller._get_projected_gravity()
    palm_world = self.data.site_xpos[self.right_palm_site_id].copy()
    block_pos = self.data.xpos[self.red_block_body_id].copy()

    contact = self._geom_contact_summary()
    palm_to_block = float(np.linalg.norm(palm_world - block_pos))
    base_to_source = float(np.linalg.norm(base_pos[:2] - np.array(self.cfg.source_center[:2])))
    block_to_target = float(np.linalg.norm(block_pos[:2] - np.array(self.cfg.target_center[:2])))
    base_to_target = float(np.linalg.norm(base_pos[:2] - np.array(self.cfg.target_center[:2])))
    source_surface = self.cfg.source_center[2] + self.cfg.source_half_extents[2]
    target_surface = self.cfg.target_center[2] + self.cfg.target_half_extents[2]
    block_bottom = float(block_pos[2] - self.cfg.block_half_height)
    lift_height = max(0.0, block_bottom - source_surface)

    if self.prev_metrics:
      block_speed = float(np.linalg.norm(block_pos - self.prev_metrics["block_pos"]) / self.control_dt)
    else:
      block_speed = 0.0

    grasp_score = 0.0
    if contact["hand_touch_bodies"]:
      grasp_score += min(1.0, len(contact["hand_touch_bodies"]) / 3.0)
    if self.controller.grip_closed and contact["hand_touch_bodies"]:
      grasp_score += 0.3
    if palm_to_block < 0.08:
      grasp_score += 0.3
    grasp_score = float(np.clip(grasp_score, 0.0, 1.0))

    placed = (
      contact["target_support"]
      and not contact["robot_touch"]
      and abs(block_pos[0] - self.cfg.target_center[0]) < self.cfg.target_half_extents[0] - 0.015
      and abs(block_pos[1] - self.cfg.target_center[1]) < self.cfg.target_half_extents[1] - 0.015
      and abs(block_bottom - target_surface) < 0.03
      and block_speed < 0.2
    )

    return {
      "base_pos": base_pos,
      "base_quat": base_quat,
      "lin_vel": lin_vel,
      "ang_vel": ang_vel,
      "proj_gravity": proj_gravity,
      "palm_world": palm_world,
      "block_pos": block_pos,
      "block_speed": block_speed,
      "base_to_source": base_to_source,
      "palm_to_block": palm_to_block,
      "block_to_target": block_to_target,
      "base_to_target": base_to_target,
      "lift_height": lift_height,
      "touching": bool(contact["hand_touch_bodies"]),
      "grasp_score": grasp_score,
      "placed": bool(placed),
      "robot_touch": bool(contact["robot_touch"]),
      "target_support": bool(contact["target_support"]),
      "source_support": bool(contact["source_support"]),
      "fallen": bool(self.data.qpos[2] < 0.48 or proj_gravity[2] > -0.4),
      "dropped": bool(block_pos[2] < 0.35),
    }

  def _render_camera(self, name: str) -> np.ndarray:
    if self.renderer is None:
      width, height = self.cfg.image_size
      return np.zeros((3, height, width), dtype=np.float32)
    image = self.renderer.render(name).astype(np.float32) / 255.0
    return np.transpose(image, (2, 0, 1))

  def _get_obs(self, metrics: dict[str, Any]) -> dict[str, np.ndarray]:
    joint_pos = self.controller._get_joint_positions()
    joint_vel = self.controller._get_joint_velocities()
    arm_pos = self.controller._get_arm_joint_positions()
    arm_vel = self.controller._get_arm_joint_velocities()
    base_pos = metrics["base_pos"]
    base_quat = metrics["base_quat"]
    palm_rel = G1Controller._quat_apply_inverse(base_quat, metrics["palm_world"] - base_pos)
    block_rel = G1Controller._quat_apply_inverse(base_quat, metrics["block_pos"] - base_pos)
    source_rel = G1Controller._quat_apply_inverse(
      base_quat, np.asarray(self.cfg.source_center, dtype=np.float32) - base_pos
    )
    target_rel = G1Controller._quat_apply_inverse(
      base_quat, np.asarray(self.cfg.target_center, dtype=np.float32) - base_pos
    )
    flags = np.array(
      [
        float(self.controller.grip_closed),
        float(metrics["touching"]),
        float(metrics["grasp_score"]),
        float(metrics["lift_height"] > 0.04),
        float(metrics["placed"]),
        self.step_count / self.cfg.max_steps,
      ],
      dtype=np.float32,
    )
    proprio = np.concatenate(
      [
        metrics["lin_vel"],
        metrics["ang_vel"],
        metrics["proj_gravity"],
        joint_pos,
        joint_vel,
        arm_pos,
        arm_vel,
        self.controller.reach_target.astype(np.float32),
        palm_rel.astype(np.float32),
        block_rel.astype(np.float32),
        source_rel.astype(np.float32),
        target_rel.astype(np.float32),
        np.array(
          [
            metrics["base_to_source"],
            metrics["palm_to_block"],
            metrics["block_to_target"],
            metrics["lift_height"],
            metrics["block_speed"],
            self.controller.lin_vel_x,
            self.controller.lin_vel_y,
            self.controller.ang_vel_z,
            self.control_dt,
            float(self.step_count) / self.cfg.max_steps,
            float(self.success_streak > 0),
          ],
          dtype=np.float32,
        ),
        flags,
        self.prev_action,
      ]
    ).astype(np.float32)
    return {
      "proprio": proprio,
      "head": self._render_camera("head_cam"),
      "wrist": self._render_camera("wrist_cam"),
    }

  def _reward(self, current: dict[str, Any]) -> tuple[float, dict[str, float], bool, str]:
    if self.reward_fn is None:
      terms: dict[str, float] = {}
      stage = "default"
    else:
      terms, stage = self.reward_fn(
        self.prev_metrics,
        current,
        self.step_count,
        self.cfg.max_steps,
        self.training,
      )
    total = float(sum(terms.values()))
    self.success_streak = self.success_streak + 1 if current["placed"] else 0
    done = bool(current["fallen"] or current["dropped"] or self.success_streak >= 12)
    return total, terms, done, stage

  def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    action = np.clip(action, -1.0, 1.0)
    self.prev_action[:] = action
    self.step_count += 1

    self.controller.lin_vel_x = float(action[0] * self.cfg.max_vx)
    self.controller.lin_vel_y = float(action[1] * self.cfg.max_vy)
    self.controller.ang_vel_z = float(action[2] * self.cfg.max_yaw)
    self.controller.reach_target[0] = np.clip(
      self.controller.reach_target[0] + action[3] * self.cfg.reach_delta,
      *self.cfg.reach_bounds[0],
    )
    self.controller.reach_target[1] = np.clip(
      self.controller.reach_target[1] + action[4] * self.cfg.reach_delta,
      *self.cfg.reach_bounds[1],
    )
    self.controller.reach_target[2] = np.clip(
      self.controller.reach_target[2] + action[5] * self.cfg.reach_delta,
      *self.cfg.reach_bounds[2],
    )
    self.controller.grip_closed = bool(action[6] > 0.0)
    self.controller.reach_active = True

    target_pos = self.controller.step()
    for _ in range(self.cfg.control_decimation):
      self.controller.apply_pd_control(target_pos)
      mujoco.mj_step(self.model, self.data)

    current = self._measure()
    reward, terms, terminal, stage = self._reward(current)
    truncated = self.step_count >= self.cfg.max_steps
    done = bool(terminal or truncated)

    self.episode_return += reward
    info = {
      "reward_terms": terms,
      "success": bool(self.success_streak >= 12),
      "placed": bool(current["placed"]),
      "touching": bool(current["touching"]),
      "grasp_score": float(current["grasp_score"]),
      "lift_height": float(current["lift_height"]),
      "stage": stage,
    }
    if done:
      info["episode"] = {
        "return": float(self.episode_return),
        "length": int(self.step_count),
        "success": bool(self.success_streak >= 12),
      }

    self.prev_metrics = current
    return self._get_obs(current), reward, done, info


def _print_status(step_idx: int, reward: float, info: dict[str, Any]) -> None:
  if step_idx % 25 != 0 and not info.get("placed", False):
    return
  parts = [
    f"step={step_idx}",
    f"reward={reward:+.3f}",
    f"touch={int(info['touching'])}",
    f"grasp={info['grasp_score']:.2f}",
    f"lift={info['lift_height']:.3f}",
    f"placed={int(info['placed'])}",
    f"stage={info['stage']}",
  ]
  print(" ".join(parts))


def _configure_wide_camera(cam: Any) -> None:
  cam.lookat[:] = np.array([0.0, -0.35, 0.78], dtype=np.float32)
  cam.distance = 2.35
  cam.azimuth = 150.0
  cam.elevation = -22.0


def run_policy(args: argparse.Namespace) -> None:
  env_cfg = EnvConfig(image_size=(args.image_size, args.image_size), max_steps=args.max_steps)
  if args.block_spawn_x_min is not None and args.block_spawn_x_max is not None:
    env_cfg.block_spawn_x = (args.block_spawn_x_min, args.block_spawn_x_max)
  if args.block_spawn_y_min is not None and args.block_spawn_y_max is not None:
    env_cfg.block_spawn_y = (args.block_spawn_y_min, args.block_spawn_y_max)
  env = G1PickPlaceEnv(
    env_cfg,
    seed=args.seed,
    render_images=True,
    verbose=True,
  )
  policy = RoutineONNXPolicy(args.policy)
  obs = env.reset()

  if args.viewer:
    from mujoco import viewer

    print("Launching viewer. On macOS, use `mjpython sim.py --viewer ...`.")
    with viewer.launch_passive(env.model, env.data) as v:
      _configure_wide_camera(v.cam)
      while v.is_running():
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        if not args.summary_only:
          _print_status(env.step_count, reward, info)
        v.sync()
        if done:
          print(f"episode={info['episode']}")
          time.sleep(0.25)
          obs = env.reset()
  else:
    success_count = 0
    total_return = 0.0
    runs = args.runs if args.runs is not None else args.episodes
    for episode in range(runs):
      obs = env.reset()
      done = False
      while not done:
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        if not args.summary_only:
          _print_status(env.step_count, reward, info)
      episode_info = info["episode"]
      success_count += int(episode_info["success"])
      total_return += episode_info["return"]
      print(f"episode={episode} {episode_info}")
    print(
      "summary "
      f"runs={runs} successes={success_count} "
      f"success_rate={success_count / max(1, runs):.3f} "
      f"avg_return={total_return / max(1, runs):.3f}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Run routine.onnx in the G1 pick-place simulator")
  parser.add_argument("--policy", type=Path, default=SCRIPT_DIR / "routine.onnx")
  parser.add_argument("--episodes", type=int, default=3)
  parser.add_argument("--runs", type=int, default=None, help="Alias for headless evaluation episodes")
  parser.add_argument("--image-size", type=int, default=64)
  parser.add_argument("--max-steps", type=int, default=600)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--viewer", action="store_true", help="Open the MuJoCo passive viewer")
  parser.add_argument("--summary-only", action="store_true", help="Skip step prints and show episode summaries")
  parser.add_argument("--block-spawn-x-min", type=float, default=None)
  parser.add_argument("--block-spawn-x-max", type=float, default=None)
  parser.add_argument("--block-spawn-y-min", type=float, default=None)
  parser.add_argument("--block-spawn-y-max", type=float, default=None)
  return parser


def main() -> None:
  args = build_arg_parser().parse_args()
  run_policy(args)


if __name__ == "__main__":
  main()
