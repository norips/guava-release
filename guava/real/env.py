"""Real-robot environment and arm with the interface the Guava tools expect."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from guava.config import RobotConfig
from guava.real.bridge_client import BridgeClient, BridgeError
from guava.real.frames import simeef_to_tcp, square_resize, tcp_to_simeef
from guava.robot.arm import RobotArm


@dataclass
class RealConfig:
    """configs/real/*.yaml"""
    bridge_url: str = "http://127.0.0.1:8765"
    sam3_url: str = "http://127.0.0.1:8114"
    llm_url: str = "http://127.0.0.1:8000/v1"
    model: str = "AIcell/guava-v13b-qwen3.5-4b"
    # Height of the tabletop in the robot base frame [m]. The model sees z=0 on the table.
    table_z: float = 0.0
    image_size: int = 512
    crop: list[int] | None = None          # [x0, y0, w, h] in the camera image; None = centered square
    max_depth: float = 2.5                 # [m] farther depth is treated as invalid
    grasp: str = "pca"                     # pca | graspgen
    graspgen_url: str = "tcp://127.0.0.1:5556"
    grasp_force: float | None = None       # [N] None = bridge default
    min_grasp_width: float = 0.003         # [m] below this after a grasp the fingers closed on nothing
    linear_speed: float | None = None      # [m/s] None = bridge default
    home_joints: list[float] = field(default_factory=lambda: list(RobotConfig().home_joints))

    @classmethod
    def load(cls, path: str | Path) -> "RealConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
        return cls(**raw)

    @property
    def model_z_offset(self) -> float:
        """model z = base z + offset (guava.tools.coordinate_frame convention)."""
        return -float(self.table_z)


@dataclass
class _EnvCfg:
    # Fields of SimConfig read by the tools.
    perception: str = "sam3"
    motion_speed: float = 1.0
    object_aliases: dict = field(default_factory=dict)


class RealEnv:
    """Stands in for guava.sim.env.RoboEnv inside ToolContext."""

    def __init__(self, bridge: BridgeClient, cfg: RealConfig):
        self.bridge = bridge
        self.real_cfg = cfg
        self.cfg = _EnvCfg()
        self._current_joints = np.zeros(7)

    # Pose helpers -------------------------------------------------------
    def eef_pose_base(self) -> tuple[np.ndarray, np.ndarray]:
        """(pos, quat_wxyz) of the simulator-equivalent eef in the robot base frame."""
        s = self.bridge.state()
        self._current_joints = np.asarray(s["joints"], dtype=np.float64)
        return tcp_to_simeef(s["tcp_position"], s["tcp_quat_xyzw"])

    def base_wxyz_xyz(self) -> np.ndarray:
        return np.array([1.0, 0, 0, 0, 0, 0, 0])

    def joint_positions(self) -> np.ndarray:
        self._current_joints = np.asarray(self.bridge.state()["joints"], dtype=np.float64)
        return self._current_joints.copy()

    def gripper_width(self) -> float:
        w = self.bridge.state()["gripper_width"]
        return float(w) if w is not None else 0.0

    # Perception ---------------------------------------------------------
    def capture_rgbd(self):
        """(rgb, depth [m], K, T_base_cam) at the model's resolution; used by tools._refresh."""
        rgb, depth, K, T_base_cam = self.bridge.rgbd(fresh=True)
        depth = depth.astype(np.float32)
        depth[(depth <= 0) | (depth > self.real_cfg.max_depth) | ~np.isfinite(depth)] = 0.0
        rgb, depth, K = square_resize(rgb, depth, K, self.real_cfg.image_size,
                                      tuple(self.real_cfg.crop) if self.real_cfg.crop else None)
        return rgb, depth, K, T_base_cam

    def capture(self) -> np.ndarray:
        return self.capture_rgbd()[0]


class RealRobotArm(RobotArm):
    """RobotArm whose motions and gripper go through the bridge.

    Inherited unchanged: align_down(), rotate(), position(), rotation(), init_orientation().
    """

    def __init__(self, env: RealEnv, cfg: RobotConfig, real_cfg: RealConfig):
        super().__init__(env, ik_solver=None, cfg=cfg)
        self.real_cfg = real_cfg

    def move(self, target_pos, target_quat=None) -> str:
        target = np.asarray(target_pos, dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError(f"target_pos must have 3 finite elements, got {target_pos}")
        if target_quat is None:
            if self._last_commanded_quat is None:
                self._last_commanded_quat = self.env.eef_pose_base()[1]
            quat = self._last_commanded_quat
        else:
            quat = np.asarray(target_quat, dtype=np.float64)
            quat = quat / np.linalg.norm(quat)
            self._last_commanded_quat = quat.copy()
        tcp_pos, tcp_quat = simeef_to_tcp(target + np.asarray(self.cfg.tcp_offset), quat)
        try:
            res = self.env.bridge.move_pose(tcp_pos, tcp_quat, speed=self.real_cfg.linear_speed)
        except BridgeError as exc:
            if "workspace" not in str(exc):
                raise
            # The bridge speaks panda_hand_tcp / base frame; tell the model in its own coordinates.
            shown = target.copy()
            shown[2] += self.real_cfg.model_z_offset
            raise RuntimeError(f"target {np.round(shown, 3).tolist()} is outside the robot's allowed workspace; "
                               "the arm did not move") from None
        if not res.get("ok"):
            # Not reached (contact, obstruction): the gripper stays where it stopped and the model is told.
            # Re-seed the orientation reference so the next move does not twist back toward the old goal.
            self._last_commanded_quat = self.env.eef_pose_base()[1]
            return res.get("result", "blocked")
        return "success"

    def grasp(self) -> str:
        res = self.env.bridge.grasp(force=self.real_cfg.grasp_force)
        width = res.get("width")
        if width is not None and width > self.real_cfg.min_grasp_width:
            return "grasped"
        return "closed"

    def release(self) -> str:
        self.env.bridge.open_gripper()
        return "released"

    def home_pose(self) -> str:
        self.env.bridge.home(self.cfg.home_joints)
        self._last_commanded_quat = self.env.eef_pose_base()[1]
        return "success"

    def gripper_width(self) -> float:
        return self.env.gripper_width()
