"""Eye-to-hand calibration of the third-person camera with an ArUco marker on the gripper.

1. Print an ArUco marker (OpenCV DICT_4X4_50, default id 0), measure its black square side.
2. Fix it rigidly to the hand (e.g. on a small plate held between the closed fingers), facing the camera.
3. Bring the arm to a pose where the marker is well visible (bridge home, or `pose-gui` + --no-home).
4. Run, on the Guava workstation:

    .venv/bin/python -m guava.real.calibrate --config configs/real/franka.yaml --marker-size 0.05

The arm visits --samples poses around the start pose (+-8 cm, +-20 deg), detects the marker in the
full-resolution color image and solves AX=XB (cv2.calibrateHandEye, eye-to-hand). The resulting
T_base_cam is printed as YAML for `camera.extrinsics` of the bridge config (pixi_fer_ros2/config/guava/bridge.yaml);
restart the bridge afterwards. The consistency of the marker pose in the hand frame across samples
is reported: a few millimetres is good, more than ~1 cm means the marker moved or detections are poor.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]


def _T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def detect(cv2, rgb, K, marker_size, marker_id, dictionary):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or marker_id not in ids.ravel():
        return None
    c = corners[list(ids.ravel()).index(marker_id)].reshape(4, 2).astype(np.float64)
    s = marker_size / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, c, K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    return _T(cv2.Rodrigues(rvec)[0], tvec)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "configs/real/franka.yaml")
    p.add_argument("--marker-size", type=float, required=True, help="Side of the black square [m]")
    p.add_argument("--marker-id", type=int, default=0)
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--translation", type=float, default=0.08, help="Max offset from the start pose [m]")
    p.add_argument("--rotation", type=float, default=20.0, help="Max rotation from the start pose [deg]")
    p.add_argument("--no-home", action="store_true", help="Start from the current pose")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import cv2
    from guava.real.bridge_client import BridgeClient
    from guava.real.env import RealConfig
    from guava.real.frames import pose_matrix

    cfg = RealConfig.load(args.config)
    bridge = BridgeClient(cfg.bridge_url)
    if not args.no_home:
        bridge.home(cfg.home_joints)
    s0 = bridge.state()
    p0, R0 = np.asarray(s0["tcp_position"]), Rotation.from_quat(s0["tcp_quat_xyzw"])
    rng = np.random.default_rng(args.seed)

    if input(f"The arm will visit {args.samples} poses within {args.translation * 100:.0f} cm / {args.rotation:.0f} deg "
             "of the current pose. Workspace clear? [y/N] ").strip().lower() != "y":
        return

    R_g2b, t_g2b, R_t2c, t_t2c, T_base_tcp_all, T_cam_marker_all = [], [], [], [], [], []
    for i in range(args.samples + 1):
        if i == 0:
            pos, rot = p0, R0
        else:
            pos = p0 + rng.uniform(-args.translation, args.translation, 3)
            rot = Rotation.from_rotvec(np.radians(rng.uniform(-args.rotation, args.rotation, 3))) * R0
        try:
            bridge.move_pose(pos, rot.as_quat())
        except Exception as exc:  # noqa: BLE001
            print(f"  sample {i}: move rejected ({exc})")
            continue
        import time
        time.sleep(0.7)
        rgb, _, K, _ = bridge.rgbd()
        s = bridge.state()
        T_cm = detect(cv2, rgb, K, args.marker_size, args.marker_id, cv2.aruco.DICT_4X4_50)
        if T_cm is None:
            print(f"  sample {i}: marker not detected")
            continue
        T_bt = pose_matrix(s["tcp_position"], s["tcp_quat_xyzw"])
        T_tb = np.linalg.inv(T_bt)  # eye-to-hand: pass base->gripper in place of gripper->base
        R_g2b.append(T_tb[:3, :3]); t_g2b.append(T_tb[:3, 3])
        R_t2c.append(T_cm[:3, :3]); t_t2c.append(T_cm[:3, 3])
        T_base_tcp_all.append(T_bt); T_cam_marker_all.append(T_cm)
        print(f"  sample {i}: ok (marker at {np.round(T_cm[:3, 3], 3).tolist()} m from the camera)")
    bridge.move_pose(p0, R0.as_quat())

    if len(R_g2b) < 5:
        raise SystemExit(f"Only {len(R_g2b)} usable samples; improve marker visibility and retry.")
    R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_PARK)
    T_base_cam = _T(R, t)

    # Consistency: marker pose in the TCP frame should be constant.
    T_tcp_marker = np.array([np.linalg.inv(Tb) @ T_base_cam @ Tc for Tb, Tc in zip(T_base_tcp_all, T_cam_marker_all)])
    spread = np.linalg.norm(T_tcp_marker[:, :3, 3] - T_tcp_marker[:, :3, 3].mean(0), axis=1)
    print(f"\n{len(R_g2b)} samples, marker position spread in the hand frame: "
          f"mean {spread.mean() * 1000:.1f} mm, max {spread.max() * 1000:.1f} mm")
    q = Rotation.from_matrix(T_base_cam[:3, :3]).as_quat()
    print("\nPaste into pixi_fer_ros2/config/guava/bridge.yaml (camera:) and restart the bridge:\n")
    print("  extrinsics:")
    print(f"    position: [{', '.join(f'{v:.4f}' for v in T_base_cam[:3, 3])}]")
    print(f"    quat_xyzw: [{', '.join(f'{v:.5f}' for v in q)}]")


if __name__ == "__main__":
    main()
