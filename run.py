#!/usr/bin/env python3
"""G1 Table Red Block — standalone MuJoCo scene with walker + reacher policies.

Converted from the LuckyEngine G1-Table-Red-Block.hscene. Runs the G1 robot
with trained Walker/Croucher/Rotator/Reacher ONNX policies in a scene with
a table, red cylindrical block, and multiple cameras (head, wrist, overhead).

Controls (press keys in the GLFW viewer window):
  Arrow Keys   : Walk forward/back, strafe left/right
  ; / '        : Turn left / right
  ,            : Toggle crouch mode
  [ / ]        : Height down / up
  \\           : Stop (zero velocity)
  /            : Toggle arm freeze
  .            : Toggle reach mode (right arm)
  Numpad 8/2   : Reach target forward/backward
  Numpad 4/6   : Reach target left/right
  Numpad 7/1   : Reach target up/down
  Numpad 5     : Reset reach target (auto mode)
  U/J, Y/H, 9/0 : Reach orientation roll/pitch/yaw
  R            : Reset reach orientation
  Space        : Reset robot + zero velocity
  C            : Cycle camera view in main window
  1            : Toggle head camera window
  2            : Toggle wrist camera window

Prerequisites:
  pip install mujoco onnxruntime numpy opencv-python

Usage:
  python run.py
  python run.py --no-cameras    # Disable camera windows (faster)
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# ONNX Policy
# --------------------------------------------------------------------------- #
class ONNXPolicy:
  """ONNX policy wrapper for CPU inference."""

  def __init__(self, model_path: str):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    self.session = ort.InferenceSession(
      model_path, sess_options, providers=["CPUExecutionProvider"]
    )
    self.input_name = self.session.get_inputs()[0].name
    self.output_name = self.session.get_outputs()[0].name

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    if obs.ndim == 1:
      obs = obs.reshape(1, -1)
    obs = obs.astype(np.float32)
    return self.session.run([self.output_name], {self.input_name: obs})[0][0]


# --------------------------------------------------------------------------- #
# G1 Controller (walker + croucher + rotator + reacher)
# --------------------------------------------------------------------------- #
class G1Controller:
  """Full G1 controller with locomotion mode switching and arm reaching."""

  # GLFW key codes
  KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
  KEY_SEMICOLON, KEY_APOSTROPHE = 59, 39
  KEY_COMMA, KEY_PERIOD, KEY_SLASH, KEY_BACKSLASH = 44, 46, 47, 92
  KEY_LEFT_BRACKET, KEY_RIGHT_BRACKET = 91, 93
  KEY_KP_8, KEY_KP_2, KEY_KP_6, KEY_KP_4, KEY_KP_7, KEY_KP_1, KEY_KP_5 = (
    328, 322, 326, 324, 327, 321, 325
  )
  KEY_U, KEY_J, KEY_Y, KEY_H, KEY_9, KEY_0, KEY_R = 85, 74, 89, 72, 57, 48, 82
  KEY_COMMA_GRIP = 44  # , = Grip toggle

  WALKER_HEIGHT = 0.80

  def __init__(self, model, data, walker, croucher, rotator, config,
               right_reacher=None):
    self.model = model
    self.data = data
    self.walker_policy = walker
    self.croucher_policy = croucher
    self.rotator_policy = rotator
    self.right_reacher_policy = right_reacher
    self.config = config

    # --- Input mode: WALK or REACH ---
    # . toggles between them. Same keys (arrows, ;/') do different things.
    self.input_mode: Literal["walk", "reach"] = "walk"

    # Walk state
    self.lin_vel_x = 0.0
    self.lin_vel_y = 0.0
    self.ang_vel_z = 0.0
    self.vel_step_linear = 0.2
    self.vel_step_angular = 0.2
    self.vel_max_linear = 2.0
    self.vel_max_angular = 1.0

    # Reach state
    self.reach_active = False
    self.reach_target = np.array([0.3, -0.2, 0.2], dtype=np.float32)
    self.reach_orientation = np.zeros(3, dtype=np.float32)
    self.reach_step = 0.05
    self.last_arm_action = np.zeros(7, dtype=np.float32)
    self.last_arm_target = None
    self.arm_max_delta = 0.012
    # Frozen arm position — holds the last reach position when switching to walk
    self.frozen_arm_pos = None  # None = use defaults, array = hold position

    self.last_action = np.zeros(29, dtype=np.float32)

    # Right hand grip state
    self.grip_closed = False

    self._build_joint_mappings()
    self._build_reacher_mappings()
    self._compute_pd_gains()
    self._cache_actuator_ids()
    self._cache_finger_actuators()

    print("\n=== G1 Table Red Block Controller ===")
    print("  .         : Toggle WALK / REACH mode")
    print("  --- WALK mode ---")
    print("  Arrows    : Walk forward/back, strafe left/right")
    print("  ; / '     : Turn left / right")
    print("  \\         : Stop")
    print("  --- REACH mode ---")
    print("  Up/Down   : Reach forward / backward")
    print("  Left/Right: Reach left / right")
    print("  ; / '     : Reach up / down")
    print("  \\         : Reset reach to default")
    print("  --- Always ---")
    print("  ,         : Toggle grip (close/open right hand)")
    print("  Space     : Reset robot")
    print("=" * 40)

  def _build_joint_mappings(self):
    self.joint_names = self.config["joint_names"]
    self.num_joints = len(self.joint_names)
    self.joint_qpos_indices = {n: 7 + i for i, n in enumerate(self.joint_names)}
    self.joint_qvel_indices = {n: 6 + i for i, n in enumerate(self.joint_names)}

    self.default_joint_pos = np.zeros(self.num_joints, dtype=np.float32)
    for name, value in self.config["default_joint_pos"].items():
      if name in self.joint_names:
        self.default_joint_pos[self.joint_names.index(name)] = value

    self.action_scales = np.array(
      [self.config["action_scales"][n] for n in self.joint_names], dtype=np.float32
    )

    arm_patterns = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                    "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
    self.arm_indices = []
    for i, name in enumerate(self.joint_names):
      if any(p in name for p in arm_patterns):
        self.arm_indices.append(i)

  def _build_reacher_mappings(self):
    rc = self.config.get("right_reacher", {})
    self.right_arm_joint_names = rc.get("arm_joint_names", [
      "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint", "right_elbow_joint",
      "right_wrist_roll_joint", "right_wrist_pitch_joint",
      "right_wrist_yaw_joint",
    ])
    self.right_arm_indices = [
      self.joint_names.index(n) for n in self.right_arm_joint_names
      if n in self.joint_names
    ]
    arm_scales = rc.get("arm_action_scales", {})
    self.arm_action_scales = np.array([
      arm_scales.get(n, self.action_scales[self.joint_names.index(n)])
      for n in self.right_arm_joint_names
    ], dtype=np.float32)
    arm_defaults = rc.get("arm_default_pos", {})
    self.arm_default_pos = np.array([
      arm_defaults.get(n, self.default_joint_pos[self.joint_names.index(n)])
      for n in self.right_arm_joint_names
    ], dtype=np.float32)
    self.right_palm_site_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_SITE, "right_palm"
    )

  def _compute_pd_gains(self):
    S5020, D5020, E5020 = 14.2506, 0.9072, 25.0
    S7520_14, D7520_14, E7520_14 = 40.1792, 2.5579, 88.0
    S7520_22, D7520_22, E7520_22 = 99.0984, 6.3088, 139.0
    S4010, D4010, E4010 = 16.7783, 1.0681, 5.0

    self.kp = np.zeros(self.num_joints, dtype=np.float32)
    self.kd = np.zeros(self.num_joints, dtype=np.float32)
    self.effort_limit = np.zeros(self.num_joints, dtype=np.float32)

    for i, name in enumerate(self.joint_names):
      if "elbow" in name or "shoulder" in name or "wrist_roll" in name:
        self.kp[i], self.kd[i], self.effort_limit[i] = S5020, D5020, E5020
      elif "hip_pitch" in name or "hip_yaw" in name or name == "waist_yaw_joint":
        self.kp[i], self.kd[i], self.effort_limit[i] = S7520_14, D7520_14, E7520_14
      elif "hip_roll" in name or "knee" in name:
        self.kp[i], self.kd[i], self.effort_limit[i] = S7520_22, D7520_22, E7520_22
      elif "wrist_pitch" in name or "wrist_yaw" in name:
        self.kp[i], self.kd[i], self.effort_limit[i] = S4010, D4010, E4010
      elif "ankle" in name or name in ("waist_pitch_joint", "waist_roll_joint"):
        self.kp[i], self.kd[i], self.effort_limit[i] = S5020 * 2, D5020 * 2, E5020 * 2
      else:
        self.kp[i], self.kd[i], self.effort_limit[i] = S5020, D5020, E5020

  # --- Keyboard ---
  def key_callback(self, key: int) -> None:
    # Grip toggle (works in any mode)
    if key == self.KEY_COMMA_GRIP:
      self.grip_closed = not self.grip_closed
      print(f"[GRIP] Right hand: {'CLOSED' if self.grip_closed else 'OPEN'}")
      return

    # Toggle input mode
    if key == self.KEY_PERIOD:
      if self.right_reacher_policy is None:
        print("[WARN] No right reacher policy loaded")
        return
      if self.input_mode == "walk":
        self.input_mode = "reach"
        self.reach_active = True
        # Init reach target to a sensible default in front of pelvis
        self.reach_target[:] = [0.3, -0.2, 0.2]
        self.reach_orientation[:] = 0.0
        self.last_arm_target = self._get_arm_joint_positions() + self.arm_default_pos
        print("[MODE] >>> REACH — arrows move hand, ;/' = up/down, \\ = reset target")
      else:
        self.input_mode = "walk"
        self.reach_active = False
        # Freeze arm where it is — read current right arm joint positions
        if self.last_arm_target is not None:
          self.frozen_arm_pos = self.last_arm_target.copy()
        self.last_arm_target = None
        print("[MODE] >>> WALK — arm holds position, arrows move robot")
      return

    # Route keys based on mode
    if self.input_mode == "walk":
      self._handle_walk_key(key)
    else:
      self._handle_reach_key(key)

  def _handle_walk_key(self, key: int) -> None:
    if key == self.KEY_UP:
      self.lin_vel_x = np.clip(self.lin_vel_x + self.vel_step_linear, -self.vel_max_linear, self.vel_max_linear)
    elif key == self.KEY_DOWN:
      self.lin_vel_x = np.clip(self.lin_vel_x - self.vel_step_linear, -self.vel_max_linear, self.vel_max_linear)
    elif key == self.KEY_LEFT:
      self.lin_vel_y = np.clip(self.lin_vel_y + self.vel_step_linear, -self.vel_max_linear, self.vel_max_linear)
    elif key == self.KEY_RIGHT:
      self.lin_vel_y = np.clip(self.lin_vel_y - self.vel_step_linear, -self.vel_max_linear, self.vel_max_linear)
    elif key == self.KEY_SEMICOLON:
      self.ang_vel_z = np.clip(self.ang_vel_z + self.vel_step_angular, -self.vel_max_angular, self.vel_max_angular)
    elif key == self.KEY_APOSTROPHE:
      self.ang_vel_z = np.clip(self.ang_vel_z - self.vel_step_angular, -self.vel_max_angular, self.vel_max_angular)
    elif key == self.KEY_BACKSLASH or key == self.KEY_SLASH:
      self.lin_vel_x = self.lin_vel_y = self.ang_vel_z = 0.0
      print("[WALK] STOPPED")
      return
    else:
      return
    print(f"[WALK] vel: x={self.lin_vel_x:.1f} y={self.lin_vel_y:.1f} yaw={self.ang_vel_z:.1f}")

  def _handle_reach_key(self, key: int) -> None:
    if key == self.KEY_UP:
      self.reach_target[0] = np.clip(self.reach_target[0] + self.reach_step, -0.3, 0.6)
    elif key == self.KEY_DOWN:
      self.reach_target[0] = np.clip(self.reach_target[0] - self.reach_step, -0.3, 0.6)
    elif key == self.KEY_LEFT:
      self.reach_target[1] = np.clip(self.reach_target[1] + self.reach_step, -0.6, 0.3)
    elif key == self.KEY_RIGHT:
      self.reach_target[1] = np.clip(self.reach_target[1] - self.reach_step, -0.6, 0.3)
    elif key == self.KEY_SEMICOLON:
      self.reach_target[2] = np.clip(self.reach_target[2] + self.reach_step, -0.4, 0.6)
    elif key == self.KEY_APOSTROPHE:
      self.reach_target[2] = np.clip(self.reach_target[2] - self.reach_step, -0.4, 0.6)
    elif key == self.KEY_BACKSLASH or key == self.KEY_SLASH:
      self.reach_target[:] = [0.3, -0.2, 0.2]
      self.reach_orientation[:] = 0.0
      print("[REACH] Target reset to default")
      return
    else:
      return
    print(f"[REACH] target: fwd={self.reach_target[0]:.2f} side={self.reach_target[1]:.2f} up={self.reach_target[2]:.2f}")

  # --- State helpers ---
  def _get_base_pose(self):
    return self.data.qpos[:3].copy(), self.data.qpos[3:7].copy()

  @staticmethod
  def _quat_apply_inverse(quat, vec):
    w, xyz = quat[0], quat[1:4]
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)

  def _get_base_velocities(self):
    lin_vel_world = self.data.qvel[:3].copy()
    ang_vel_body = self.data.qvel[3:6].copy()
    _, quat = self._get_base_pose()
    return self._quat_apply_inverse(quat, lin_vel_world), ang_vel_body

  def _get_projected_gravity(self):
    _, quat = self._get_base_pose()
    return self._quat_apply_inverse(quat, np.array([0.0, 0.0, -1.0]))

  def _get_joint_positions(self):
    pos = np.zeros(self.num_joints, dtype=np.float32)
    for i, n in enumerate(self.joint_names):
      pos[i] = self.data.qpos[self.joint_qpos_indices[n]] - self.default_joint_pos[i]
    return pos

  def _get_joint_velocities(self):
    vel = np.zeros(self.num_joints, dtype=np.float32)
    for i, n in enumerate(self.joint_names):
      vel[i] = self.data.qvel[self.joint_qvel_indices[n]]
    return vel

  def _get_arm_joint_positions(self):
    pos = np.zeros(len(self.right_arm_indices), dtype=np.float32)
    for i, idx in enumerate(self.right_arm_indices):
      n = self.joint_names[idx]
      pos[i] = self.data.qpos[self.joint_qpos_indices[n]] - self.arm_default_pos[i]
    return pos

  def _get_arm_joint_velocities(self):
    vel = np.zeros(len(self.right_arm_indices), dtype=np.float32)
    for i, idx in enumerate(self.right_arm_indices):
      vel[i] = self.data.qvel[self.joint_qvel_indices[self.joint_names[idx]]]
    return vel

  def _get_palm_pos_in_pelvis(self):
    palm_world = self.data.site_xpos[self.right_palm_site_id].copy()
    pos, quat = self._get_base_pose()
    return self._quat_apply_inverse(quat, palm_world - pos)

  def _get_palm_orientation_in_pelvis(self):
    mat = self.data.site_xmat[self.right_palm_site_id].reshape(3, 3)
    palm_q = np.zeros(4)
    mujoco.mju_mat2Quat(palm_q, mat.flatten())
    _, pelvis_q = self._get_base_pose()
    pinv = np.array([pelvis_q[0], -pelvis_q[1], -pelvis_q[2], -pelvis_q[3]])
    w1, x1, y1, z1 = pinv
    w2, x2, y2, z2 = palm_q
    rel = np.array([
      w1*w2 - x1*x2 - y1*y2 - z1*z2,
      w1*x2 + x1*w2 + y1*z2 - z1*y2,
      w1*y2 - x1*z2 + y1*w2 + z1*x2,
      w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
    w, x, y, z = rel
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp = np.clip(2*(w*y - z*x), -1, 1)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw], dtype=np.float32)

  # --- Step ---
  def step(self) -> np.ndarray:
    # Build walker observation (always runs — keeps legs stable)
    lin_vel, ang_vel = self._get_base_velocities()
    proj_gravity = self._get_projected_gravity()
    joint_pos = self._get_joint_positions()
    joint_vel = self._get_joint_velocities()

    cmd = np.array([self.lin_vel_x, self.lin_vel_y, self.ang_vel_z], dtype=np.float32)

    obs = np.concatenate([
      lin_vel, ang_vel, proj_gravity, joint_pos, joint_vel, self.last_action, cmd,
    ]).astype(np.float32)

    # Walker policy (handles legs, waist, standing, walking, turning)
    action = self.walker_policy(obs)
    target_pos = self.default_joint_pos + action * self.action_scales

    # Arms: left arm always at default, right arm holds last reach position
    for idx in self.arm_indices:
      target_pos[idx] = self.default_joint_pos[idx]

    # Right arm: if we have a frozen position from a previous reach, hold it
    if not self.reach_active and self.frozen_arm_pos is not None:
      for i, full_idx in enumerate(self.right_arm_indices):
        target_pos[full_idx] = self.frozen_arm_pos[i]

    # Right arm reacher overlay (when in reach mode)
    if self.reach_active and self.right_reacher_policy is not None:
      reacher_obs = np.concatenate([
        self.reach_target,
        self.reach_orientation,
        self._get_palm_pos_in_pelvis(),
        self._get_palm_orientation_in_pelvis(),
        self._get_arm_joint_positions(),
        self._get_arm_joint_velocities(),
        self.last_arm_action,
        proj_gravity.astype(np.float32),
      ]).astype(np.float32)

      arm_action = self.right_reacher_policy(reacher_obs)
      arm_target = self.arm_default_pos + arm_action * self.arm_action_scales

      if self.last_arm_target is not None:
        delta = np.clip(arm_target - self.last_arm_target, -self.arm_max_delta, self.arm_max_delta)
        arm_target = self.last_arm_target + delta
      self.last_arm_target = arm_target.copy()

      for i, full_idx in enumerate(self.right_arm_indices):
        target_pos[full_idx] = arm_target[i]
      self.last_arm_action = arm_action.copy()

    self.last_action = action.copy()
    return target_pos

  def _cache_actuator_ids(self):
    """Cache actuator IDs once at init instead of looking up every step."""
    self.actuator_ids = []
    for name in self.joint_names:
      self.actuator_ids.append(
        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
      )

  def _cache_finger_actuators(self):
    """Cache right hand finger actuator IDs and their closed targets."""
    # (actuator_id, closed_position) — targets at joint limits for a power grasp
    self.right_finger_actuators = []
    finger_closed = {
      "right_hand_thumb_0_joint":  0.8,     # curl thumb inward
      "right_hand_thumb_1_joint": -0.9,     # flex thumb
      "right_hand_thumb_2_joint": -1.5,     # curl thumb tip
      "right_hand_index_0_joint":  1.4,     # curl index
      "right_hand_index_1_joint":  1.5,     # curl index tip
      "right_hand_middle_0_joint": 1.4,     # curl middle
      "right_hand_middle_1_joint": 1.5,     # curl middle tip
    }
    for name, closed_val in finger_closed.items():
      aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
      if aid >= 0:
        self.right_finger_actuators.append((aid, closed_val))

  def apply_pd_control(self, target_pos):
    for i, act_id in enumerate(self.actuator_ids):
      if act_id >= 0:
        self.data.ctrl[act_id] = target_pos[i]
    # Apply grip
    for act_id, closed_val in self.right_finger_actuators:
      self.data.ctrl[act_id] = closed_val if self.grip_closed else 0.0


# --------------------------------------------------------------------------- #
# Camera renderer (uses mujoco.Renderer — reliable offscreen rendering)
# --------------------------------------------------------------------------- #
class CameraRenderer:
  """Offscreen renderer for robot-mounted cameras using mujoco.Renderer."""

  def __init__(self, model, data, width=320, height=240):
    self.model = model
    self.data = data
    self.renderer = mujoco.Renderer(model, height, width)

  def render(self, camera_name: str) -> np.ndarray:
    """Render from a named camera, return RGB array (H, W, 3)."""
    self.renderer.update_scene(self.data, camera=camera_name)
    return self.renderer.render().copy()


# --------------------------------------------------------------------------- #
# Armature setup
# --------------------------------------------------------------------------- #
def set_armature(model, joint_names):
  ARM_5020 = 0.00360972
  ARM_7520_14 = 0.01017752
  ARM_7520_22 = 0.02510192
  ARM_4010 = 0.00425000
  ARM_2x5020 = 0.00721945

  for i, name in enumerate(joint_names):
    dof = 6 + i
    if "elbow" in name or "shoulder" in name or "wrist_roll" in name:
      model.dof_armature[dof] = ARM_5020
    elif "hip_pitch" in name or "hip_yaw" in name or name == "waist_yaw_joint":
      model.dof_armature[dof] = ARM_7520_14
    elif "hip_roll" in name or "knee" in name:
      model.dof_armature[dof] = ARM_7520_22
    elif "wrist_pitch" in name or "wrist_yaw" in name:
      model.dof_armature[dof] = ARM_4010
    elif "ankle" in name or name in ("waist_pitch_joint", "waist_roll_joint"):
      model.dof_armature[dof] = ARM_2x5020
    else:
      model.dof_armature[dof] = ARM_5020


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
  parser = argparse.ArgumentParser(description="G1 Table Red Block — MuJoCo standalone")
  parser.add_argument("--no-cameras", action="store_true", help="Disable camera windows")
  parser.add_argument("--cam-fps", type=int, default=10, help="Camera render FPS (default: 10)")
  args = parser.parse_args()

  # Load config
  config_path = SCRIPT_DIR / "model_config.json"
  with open(config_path) as f:
    config = json.load(f)
  joint_names = config["joint_names"]

  # Load scene
  xml_path = SCRIPT_DIR / "scene.xml"
  print(f"Loading scene: {xml_path}")
  model = mujoco.MjModel.from_xml_path(str(xml_path))
  model.opt.timestep = 0.005  # 200 Hz — must match training
  set_armature(model, joint_names)

  data = mujoco.MjData(model)

  # Init robot pose — spawn behind the table, facing it
  data.qpos[0] = -0.6  # x: back from table
  data.qpos[2] = 0.76
  data.qpos[3:7] = [1, 0, 0, 0]
  for name, value in config["default_joint_pos"].items():
    if name in joint_names:
      data.qpos[7 + joint_names.index(name)] = value
  mujoco.mj_forward(model, data)

  # Load policies
  print("Loading ONNX policies...")
  walker = ONNXPolicy(str(SCRIPT_DIR / "walker.onnx"))
  croucher = ONNXPolicy(str(SCRIPT_DIR / "croucher.onnx"))
  rotator = ONNXPolicy(str(SCRIPT_DIR / "rotator.onnx"))

  right_reacher = None
  rr_path = SCRIPT_DIR / "right_reacher.onnx"
  if rr_path.exists():
    right_reacher = ONNXPolicy(str(rr_path))
    print("  Right reacher loaded.")

  # Create controller
  ctrl = G1Controller(model, data, walker, croucher, rotator, config,
                      right_reacher=right_reacher)

  # Warm up ONNX models (first call triggers JIT compilation)
  print("Warming up policies...")
  _dummy99 = np.zeros((1, 99), dtype=np.float32)
  _dummy101 = np.zeros((1, 101), dtype=np.float32)
  _dummy36 = np.zeros((1, 36), dtype=np.float32)
  walker(_dummy99)
  croucher(_dummy101)
  rotator(_dummy99)
  if right_reacher:
    right_reacher(_dummy36)
  print("  Policies warm.")

  # Camera renderer (offscreen, for head/wrist cam windows)
  cam_renderer = None
  cv2 = None
  show_head_cam = not args.no_cameras
  show_wrist_cam = not args.no_cameras
  if not args.no_cameras:
    try:
      import cv2 as _cv2
      cv2 = _cv2
      cam_renderer = CameraRenderer(model, data, 320, 240)
      # Warm up renderer (first call compiles shaders)
      cam_renderer.render("head_cam")
      cam_renderer.render("wrist_cam")
      print("  Camera renderer ready (head_cam, wrist_cam).")
    except ImportError:
      print("  [WARN] opencv-python not installed — camera windows disabled.")
      print("  Install with: pip install opencv-python")
      show_head_cam = show_wrist_cam = False
    except Exception as e:
      print(f"  [WARN] Camera renderer init failed: {e}")
      show_head_cam = show_wrist_cam = False

  # Print controls
  print(f"\n{'='*50}")
  print("G1 TABLE RED BLOCK — MuJoCo Standalone")
  print(f"{'='*50}")
  print("  .          Toggle WALK / REACH mode")
  print("  --- WALK mode ---")
  print("  Arrows     Walk fwd/back, strafe L/R")
  print("  ; / '      Turn left / right")
  print("  \\          Stop")
  print("  --- REACH mode ---")
  print("  Up/Down    Reach forward / backward")
  print("  Left/Right Reach left / right")
  print("  ; / '      Reach up / down")
  print("  \\          Reset reach target")
  print("  --- Always ---")
  print("  Space      Reset robot")
  print(f"{'='*50}\n")

  # Mutable state for key callback
  state = {"reset": False}

  def on_key(keycode: int) -> None:
    if keycode == 32:  # Space
      state["reset"] = True
    else:
      ctrl.key_callback(keycode)

  # ------------------------------------------------------------------- #
  # Simulation loop using launch_passive (MuJoCo's built-in viewer)
  # ------------------------------------------------------------------- #
  from mujoco import viewer

  decimation = 4
  control_step = 0
  target_pos = ctrl.default_joint_pos.copy()
  sim_time = 0.0
  last_cam_render = 0.0
  cam_interval = 1.0 / args.cam_fps

  print("Launching MuJoCo viewer...")

  with viewer.launch_passive(model, data, key_callback=on_key) as v:
    # Reset clock AFTER viewer opens — prevents catchup lag burst on startup
    t0 = time.time()
    while v.is_running():
      # Handle spacebar reset
      if state["reset"]:
        mujoco.mj_resetData(model, data)
        data.qpos[0] = -0.6
        data.qpos[2] = 0.76
        data.qpos[3:7] = [1, 0, 0, 0]
        for name, value in config["default_joint_pos"].items():
          if name in joint_names:
            data.qpos[7 + joint_names.index(name)] = value
        mujoco.mj_forward(model, data)
        ctrl.last_action[:] = 0
        ctrl.last_arm_action[:] = 0
        ctrl.lin_vel_x = ctrl.lin_vel_y = ctrl.ang_vel_z = 0.0
        ctrl.reach_active = False
        ctrl.last_arm_target = None
        ctrl.frozen_arm_pos = None
        ctrl.grip_closed = False
        ctrl.input_mode = "walk"
        target_pos = ctrl.default_joint_pos.copy()
        state["reset"] = False
        print("[RESET] Robot reset → WALK mode")

      # Step physics in real time (cap catchup to avoid jitter snowball)
      wall = time.time() - t0
      max_catchup = 0.05  # Never try to catch up more than 50ms per frame
      if wall - sim_time > max_catchup:
        sim_time = wall - max_catchup
      while sim_time < wall:
        if control_step % decimation == 0:
          target_pos = ctrl.step()
        ctrl.apply_pd_control(target_pos)
        mujoco.mj_step(model, data)
        control_step += 1
        sim_time += model.opt.timestep

      # Sync viewer
      v.sync()

      # Render camera views at lower FPS
      if cam_renderer and cv2 and (show_head_cam or show_wrist_cam):
        now = time.time()
        if now - last_cam_render >= cam_interval:
          last_cam_render = now
          if show_head_cam:
            img = cam_renderer.render("head_cam")
            cv2.imshow("Head Camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
          if show_wrist_cam:
            img = cam_renderer.render("wrist_cam")
            cv2.imshow("Wrist Camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
          cv2.waitKey(1)

  # Cleanup
  if cv2:
    try:
      cv2.destroyAllWindows()
    except Exception:
      pass
  print("Done.")


if __name__ == "__main__":
  main()
