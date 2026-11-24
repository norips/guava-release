"""Frame and image conventions between the real Franka and the simulator the model was trained in.

End-effector
------------
The tools (and the model's rotation / position feedback) use robosuite's ``gripper0_right_eef``
frame. Relative to the Franka ``panda_hand_tcp`` frame (franka_description, used by CRISP):

* rotation: ``R_eef = R_tcp @ Rz(-90 deg)`` -- the eef X axis is the jaw closing axis (tcp Y);
* translation: the eef origin is 6.9 mm closer to the flange
  (robosuite: link7 -> eef = 0.1065 + 0.097 = 0.2035 m, URDF: link7 -> tcp = 0.107 + 0.1034 = 0.2104 m).

Camera
------
The simulator renders 512x512 images. Real frames are center-cropped to a square and resized,
with the intrinsics adjusted accordingly.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

# Pose of the simulator eef frame in the panda_hand_tcp frame.
T_TCP_SIMEEF = np.eye(4)
T_TCP_SIMEEF[:3, :3] = Rotation.from_euler("z", -90, degrees=True).as_matrix()
T_TCP_SIMEEF[:3, 3] = [0.0, 0.0, -(0.2104 - 0.2035)]
T_SIMEEF_TCP = np.linalg.inv(T_TCP_SIMEEF)


def wxyz_to_xyzw(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([x, y, z, w], dtype=np.float64)


def xyzw_to_wxyz(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([w, x, y, z], dtype=np.float64)


def pose_matrix(position, quat_xyzw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = position
    return T


def tcp_to_simeef(tcp_pos, tcp_quat_xyzw) -> tuple[np.ndarray, np.ndarray]:
    """panda_hand_tcp pose (base frame) -> sim eef (pos, quat wxyz)."""
    T = pose_matrix(tcp_pos, tcp_quat_xyzw) @ T_TCP_SIMEEF
    return T[:3, 3].copy(), xyzw_to_wxyz(Rotation.from_matrix(T[:3, :3]).as_quat())


def simeef_to_tcp(eef_pos, eef_quat_wxyz) -> tuple[np.ndarray, np.ndarray]:
    """sim eef target (pos, quat wxyz) -> panda_hand_tcp target (pos, quat xyzw)."""
    T = pose_matrix(eef_pos, wxyz_to_xyzw(eef_quat_wxyz)) @ T_SIMEEF_TCP
    return T[:3, 3].copy(), Rotation.from_matrix(T[:3, :3]).as_quat()


def square_resize(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, size: int,
                  crop: tuple[int, int, int, int] | None = None):
    """Crop (x0, y0, w, h; default: centered square) and resize RGB + depth to size x size.

    Depth is resized with nearest-neighbour sampling so no depth is invented at object borders.
    Returns (rgb, depth, K) for the new image.
    """
    h, w = depth.shape
    if crop is None:
        side = min(h, w)
        crop = ((w - side) // 2, (h - side) // 2, side, side)
    x0, y0, cw, ch = crop
    if x0 < 0 or y0 < 0 or x0 + cw > w or y0 + ch > h:
        raise ValueError(f"crop {crop} is outside the {w}x{h} image")
    rgb_c = rgb[y0:y0 + ch, x0:x0 + cw]
    dep_c = depth[y0:y0 + ch, x0:x0 + cw]
    sx, sy = size / cw, size / ch
    rgb_r = np.asarray(Image.fromarray(rgb_c).resize((size, size), Image.Resampling.BILINEAR))
    dep_r = np.asarray(Image.fromarray(dep_c.astype(np.float32), mode="F").resize((size, size), Image.Resampling.NEAREST))
    K2 = K.astype(np.float64).copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    # Pixel-centre convention: u' + 0.5 = (u + 0.5 - x0) * sx
    K2[0, 2] = (K[0, 2] + 0.5 - x0) * sx - 0.5
    K2[1, 2] = (K[1, 2] + 0.5 - y0) * sy - 0.5
    return rgb_r.copy(), dep_r.astype(np.float32).copy(), K2


# --------------------------------------------------------------------------------------------------
# Wrist camera helpers
# --------------------------------------------------------------------------------------------------

# Orientation of the simulator `frontview` camera (OpenCV convention: x right, y down, z forward) in the
# robot base frame (robosuite table_arena.xml, MuJoCo quat wxyz 0.56 0.43 0.43 0.56; MuJoCo cameras
# look along -z with y up). align() resolves top/left/right/front/back in this view, so a wrist camera
# looking down gets the same directions as in training.
FRONTVIEW_R_BASE_CAM = (Rotation.from_quat([0.43, 0.43, 0.56, 0.56]).as_matrix()
                        @ np.diag([1.0, -1.0, -1.0]))
FRONTVIEW_R_BASE_CAM = Rotation.from_matrix(FRONTVIEW_R_BASE_CAM).as_matrix()  # re-orthonormalize


def rotate_k90(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray, k: int):
    """Rotate the image by k * 90 deg counter-clockwise (np.rot90) and update K and the camera pose so
    that unprojected points are unchanged."""
    k %= 4
    K = K.astype(np.float64).copy()
    T = T_base_cam.astype(np.float64).copy()
    M = np.eye(4)
    M[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # p_cam = M p_cam' for one CCW turn
    for _ in range(k):
        w = rgb.shape[1]
        rgb, depth = np.rot90(rgb), np.rot90(depth)
        K = np.array([[K[1, 1], 0, K[1, 2]], [0, K[0, 0], w - 1 - K[0, 2]], [0, 0, 1]])
        T = T @ M
    return np.ascontiguousarray(rgb), np.ascontiguousarray(depth), K, T


def auto_rotation_k(T_base_cam: np.ndarray) -> int:
    """Quarter turns that make the image 'up' point toward the robot base (-x), as in the sim front view,
    so 'left' / 'right' in the image match the training camera."""
    R = T_base_cam[:3, :3]
    ups = []
    for k in range(4):
        up = -R[:, 1]  # image up (camera -y) in base
        ups.append(-up[0])
        R = R @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    return int(np.argmax(ups))
