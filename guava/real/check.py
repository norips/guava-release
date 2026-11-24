"""Bring-up checks for the real-robot setup. Run each time the camera or the table moves.

    .venv/bin/python -m guava.real.check --config configs/real/franka.yaml [--object carrot] [--hover carrot]

Writes to --output (default runs/real/check):
  overlay.png   model-resolution image with the projected TCP (red cross) and the configured
                table plane over the workspace (green grid). The cross must sit between the
                fingertips and the grid on the table, otherwise recalibrate (guava.real.calibrate).
  depth.png     depth visualisation.
Prints the gripper state exactly as the model sees it, depth statistics (check the scale with a
tape measure) and an estimate of `table_z` from the depth image.
With --object, segments the object with SAM3 and prints get_position_and_size in both frames.
With --hover, also moves the TCP 10 cm above that object (asks first) to validate the calibration
physically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]


def _project(points_base: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    T_cam_base = np.linalg.inv(T_base_cam)
    p = (T_cam_base[:3, :3] @ points_base.T).T + T_cam_base[:3, 3]
    uv = (K @ p.T).T
    out = np.full((len(points_base), 2), np.nan)
    front = p[:, 2] > 1e-6
    out[front] = uv[front, :2] / uv[front, 2:3]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "configs/real/franka.yaml")
    p.add_argument("--output", type=Path, default=ROOT / "runs/real/check")
    p.add_argument("--object", help="Object name to localize with SAM3")
    p.add_argument("--hover", help="Object name: move the TCP 10 cm above it (asks for confirmation)")
    args = p.parse_args(argv)

    from guava.real.env import RealConfig
    from guava.real.frames import tcp_to_simeef
    from guava.real.run import build_context
    from guava.tools.tools import _gripper_state_text, _refresh, get_position_and_size

    cfg = RealConfig.load(args.config)
    name = args.hover or args.object
    ctx = build_context(cfg, need_sam3=bool(name))
    args.output.mkdir(parents=True, exist_ok=True)

    # --- robot state -------------------------------------------------------
    s = ctx.env.bridge.state()
    eef_p, eef_q = tcp_to_simeef(s["tcp_position"], s["tcp_quat_xyzw"])
    print("joints            :", np.round(s["joints"], 3).tolist())
    print("panda_hand_tcp    :", np.round(s["tcp_position"], 4).tolist(), "quat xyzw", np.round(s["tcp_quat_xyzw"], 3).tolist())
    print("sim eef (base)    :", np.round(eef_p, 4).tolist(), "quat wxyz", np.round(eef_q, 3).tolist())
    print("gripper width [m] :", s["gripper_width"])
    print("model sees        :", _gripper_state_text(ctx))

    # --- camera ------------------------------------------------------------
    rgb_raw, depth_raw, K_raw, T_base_cam = ctx.env.bridge.rgbd()
    valid = depth_raw > 0
    h, w = depth_raw.shape
    center = depth_raw[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10]
    print(f"\ncamera            : {w}x{h}, fx={K_raw[0, 0]:.1f} fy={K_raw[1, 1]:.1f}, valid depth {valid.mean():.0%}")
    print(f"depth at centre   : {np.median(center[center > 0]) if (center > 0).any() else float('nan'):.3f} m "
          "(measure the distance to check `depth_scale`)")
    print("camera in base [m]:", np.round(T_base_cam[:3, 3], 3).tolist())

    _refresh(ctx)
    rgb, depth, K, T = ctx.last_rgb, ctx.last_depth, ctx.last_K, ctx.last_pose_mat

    # table_z estimate: dominant height of the scene points inside the workspace footprint
    vs, us = np.nonzero(depth > 0)
    z = depth[vs, us]
    pts_cam = np.stack([(us - K[0, 2]) * z / K[0, 0], (vs - K[1, 2]) * z / K[1, 1], z], axis=1)
    pts = (T[:3, :3] @ pts_cam.T).T + T[:3, 3]
    sel = (pts[:, 0] > 0.25) & (pts[:, 0] < 0.8) & (np.abs(pts[:, 1]) < 0.4) & (np.abs(pts[:, 2]) < 0.3)
    if sel.sum() > 500:
        hist, edges = np.histogram(pts[sel, 2], bins=np.arange(-0.3, 0.3, 0.004))
        table_est = float(edges[np.argmax(hist)] + 0.002)
        print(f"\ntable_z estimate  : {table_est:+.3f} m (config: {cfg.table_z:+.3f}); "
              "trust it only if the overlay grid matches the table")
    else:
        print("\ntable_z estimate  : not enough depth points in front of the robot")

    # --- overlay -------------------------------------------------------------
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    gx, gy = np.meshgrid(np.linspace(0.3, 0.75, 10), np.linspace(-0.35, 0.35, 15))
    grid = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, cfg.table_z)], axis=1)
    for u, v in _project(grid, K, T):
        if np.isfinite(u):
            draw.ellipse([u - 2, v - 2, u + 2, v + 2], fill=(0, 255, 0))
    (u, v), = _project(np.asarray([s["tcp_position"]]), K, T)
    if np.isfinite(u):
        draw.line([u - 10, v, u + 10, v], fill=(255, 0, 0), width=2)
        draw.line([u, v - 10, u, v + 10], fill=(255, 0, 0), width=2)
    img.save(args.output / "overlay.png")
    d = np.clip(depth / max(1e-6, np.percentile(depth[depth > 0], 99) if (depth > 0).any() else 1.0), 0, 1)
    Image.fromarray((255 * (1 - d) * (depth > 0)).astype(np.uint8)).save(args.output / "depth.png")
    print(f"\nwrote {args.output / 'overlay.png'} and depth.png")

    # --- perception / physical check ------------------------------------------
    if name:
        res = json.loads(get_position_and_size(ctx, name))
        model_center = list(res["center"])
        model_center[2] += cfg.model_z_offset
        print(f"\n{name}: base frame {res['center']}  model frame {np.round(model_center, 3).tolist()}  "
              f"extents {res['extents']}")
    if args.hover:
        target = np.asarray(res["center"], dtype=float)
        target[2] += res["extents"][2] / 2 + 0.10
        if input(f"Move the TCP to {np.round(target, 3).tolist()} (10 cm above {name})? [y/N] ").strip().lower() == "y":
            ctx.arm.align_down()
            print("move:", ctx.arm.move(target.tolist()))
            print("model sees:", _gripper_state_text(ctx))


if __name__ == "__main__":
    main()
