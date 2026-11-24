"""HTTP client for the robot bridge (pixi_fer_ros2/guava_bridge/server.py)."""
from __future__ import annotations

import base64
import io
import zlib

import numpy as np
import requests


class BridgeError(RuntimeError):
    pass


def _unpack(d: dict) -> np.ndarray:
    return np.load(io.BytesIO(zlib.decompress(base64.b64decode(d["npy_zlib_b64"]))), allow_pickle=False)


class BridgeClient:
    def __init__(self, url: str, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def _get(self, path: str, **params) -> dict:
        r = self._session.get(f"{self.url}{path}", params=params, timeout=self.timeout)
        return self._check(r)

    def _post(self, path: str, body: dict | None = None) -> dict:
        r = self._session.post(f"{self.url}{path}", json=body or {}, timeout=self.timeout)
        return self._check(r)

    @staticmethod
    def _check(r: requests.Response) -> dict:
        try:
            data = r.json()
        except ValueError:
            r.raise_for_status()
            raise
        if r.status_code != 200:
            raise BridgeError(data.get("error", f"HTTP {r.status_code}"))
        return data

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        try:
            return bool(self._get("/health").get("ok"))
        except Exception:
            return False

    def state(self) -> dict:
        return self._get("/state")

    def rgbd(self, fresh: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(rgb HxWx3 uint8, depth HxW float32 [m], K 3x3, T_base_cam 4x4)."""
        d = self._get("/rgbd", fresh=int(fresh))
        return (_unpack(d["rgb"]), _unpack(d["depth"]),
                np.asarray(d["K"], dtype=np.float64), np.asarray(d["T_base_cam"], dtype=np.float64))

    def move_pose(self, position, quat_xyzw, speed: float | None = None) -> dict:
        body = {"position": [float(v) for v in position], "quat_xyzw": [float(v) for v in quat_xyzw]}
        if speed is not None:
            body["speed"] = float(speed)
        return self._post("/move_pose", body)

    def home(self, joints=None) -> dict:
        return self._post("/home", {"joints": [float(j) for j in joints]} if joints is not None else {})

    def grasp(self, force: float | None = None) -> dict:
        return self._post("/gripper/grasp", {"force": force} if force else {})

    def open_gripper(self) -> dict:
        return self._post("/gripper/open")

    def stop(self) -> dict:
        return self._post("/stop")

    def recover(self) -> dict:
        return self._post("/recover")
