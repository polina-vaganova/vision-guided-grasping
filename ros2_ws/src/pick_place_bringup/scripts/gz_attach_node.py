#!/usr/bin/env python3
"""
gz_attach_node.py — Gazebo object attachment via high-frequency teleportation.

Provides two ROS 2 services:
  /attach_object  (std_srvs/SetBool)  — attach nearest box to gripper
  /detach_object  (std_srvs/SetBool)  — release attached box

Mechanism:
  On attach: finds nearest box to grasp pose, starts a background thread
  that teleports the box to the gripper position at 50 Hz using gz set_pose.
  On detach: stops the thread, box stays at its last position.
"""

import math
import random
import struct
import subprocess
import threading
import time
from typing import Optional, Tuple, Dict

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import SetBool

import tf2_ros
from tf2_ros import TransformException

try:
    import gz.transport13 as _gz_transport
    _gz_node = _gz_transport.Node()
    _USE_TRANSPORT = True
except Exception:
    _gz_node = None
    _USE_TRANSPORT = False


# ── Protobuf helpers (gz.msgs.Pose binary encoding) ──────────────────────────
# gz.msgs.Pose:      header=1, name=2, id=3, position=4, orientation=5
# gz.msgs.Vector3d:  header=1, x=2, y=3, z=4
# gz.msgs.Quaternion: header=1, x=2, y=3, z=4, w=5

def _varint(v: int) -> bytes:
    buf = []
    while v > 0x7F:
        buf.append((v & 0x7F) | 0x80)
        v >>= 7
    buf.append(v)
    return bytes(buf)

def _pb_str(field: int, s: str) -> bytes:
    b = s.encode()
    return _varint((field << 3) | 2) + _varint(len(b)) + b

def _pb_dbl(field: int, v: float) -> bytes:
    return _varint((field << 3) | 1) + struct.pack('<d', v)

def _pb_emb(field: int, b: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(b)) + b

def _build_pose_msg(model: str, x: float, y: float, z: float,
                    qx: float, qy: float, qz: float, qw: float) -> bytes:
    pos = _pb_dbl(2, x) + _pb_dbl(3, y) + _pb_dbl(4, z)
    ori = _pb_dbl(2, qx) + _pb_dbl(3, qy) + _pb_dbl(4, qz) + _pb_dbl(5, qw)
    return _pb_str(2, model) + _pb_emb(4, pos) + _pb_emb(5, ori)


# ── Gazebo helpers ───────────────────────────────────────────────────────────

def _gz_set_pose(world: str, model: str,
                 x: float, y: float, z: float,
                 qx: float = 0.0, qy: float = 0.0,
                 qz: float = 0.0, qw: float = 1.0) -> bool:
    if _USE_TRANSPORT:
        # Fire-and-forget via gz.transport13 — no blocking wait for response.
        # timeout=1 ms means we send the request but don't wait for reply;
        # Gazebo processes the pose change within the next simulation step.
        msg = _build_pose_msg(model, x, y, z, qx, qy, qz, qw)
        _gz_node.request_raw(
            f'/world/{world}/set_pose',
            msg, 'gz.msgs.Pose', 'gz.msgs.Boolean', 1,
        )
        return True
    # Fallback: subprocess (slow, ~300 ms per call)
    proto = (
        f'name: "{model}" '
        f'position {{ x: {x:.5f} y: {y:.5f} z: {z:.5f} }} '
        f'orientation {{ x: {qx:.5f} y: {qy:.5f} z: {qz:.5f} w: {qw:.5f} }}'
    )
    try:
        result = subprocess.run(
            ["gz", "service",
             "-s", f"/world/{world}/set_pose",
             "--reqtype", "gz.msgs.Pose",
             "--reptype", "gz.msgs.Boolean",
             "--timeout", "300",
             "--req", proto],
            capture_output=True, text=True, timeout=1.0,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ── Node ─────────────────────────────────────────────────────────────────────

class GzAttachNode(Node):

    def __init__(self):
        super().__init__("gz_attach_node")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("world_name",    "pick_and_place")
        self.declare_parameter("gripper_link",  "tool0")
        self.declare_parameter("robot_spawn_x",  0.0)
        self.declare_parameter("robot_spawn_y", -0.58)
        self.declare_parameter("robot_spawn_z",  0.780)
        self.declare_parameter("track_hz",      100.0)
        self.declare_parameter("gripper_z_offset", 0.0)

        def gp(n):
            return self.get_parameter(n).value

        self._world       = gp("world_name")
        self._grip_link   = gp("gripper_link")
        self._spawn       = (gp("robot_spawn_x"), gp("robot_spawn_y"), gp("robot_spawn_z"))
        self._track_dt    = 1.0 / gp("track_hz")
        self._gz_z_off    = gp("gripper_z_offset")

        # ── All box names ─────────────────────────────────────────────────
        self._all_box_names = [f"box_{i:02d}" for i in range(1, 11)]

        # ── Box dimensions ────────────────────────────────────────────────
        self._box_dims = {
            'box_01': (0.05, 0.04, 0.03), 'box_02': (0.07, 0.05, 0.04),
            'box_03': (0.08, 0.06, 0.05), 'box_04': (0.10, 0.07, 0.06),
            'box_05': (0.12, 0.09, 0.07), 'box_06': (0.14, 0.10, 0.07),
            'box_07': (0.16, 0.12, 0.08), 'box_08': (0.18, 0.13, 0.09),
            'box_09': (0.20, 0.15, 0.10), 'box_10': (0.23, 0.17, 0.12),
        }

        # ── State ─────────────────────────────────────────────────────────
        self._box_poses: Dict[str, Tuple[float, float, float]] = {}
        self._box_names_cache: list = []
        self._placed_boxes: set = set()
        self._attached_model: Optional[str] = None
        self._track_thread: Optional[threading.Thread] = None
        self._track_stop = threading.Event()
        self._last_grasp_gz: Optional[Tuple[float, float, float]] = None

        # ── TF2 ───────────────────────────────────────────────────────────
        self._tf_buf = tf2_ros.Buffer()
        self._tf_lis = tf2_ros.TransformListener(self._tf_buf, self)

        # ── Services ──────────────────────────────────────────────────────
        self.create_service(SetBool, "/attach_object", self._attach_cb)
        self.create_service(SetBool, "/detach_object", self._detach_cb)

        # ── Subscriber ────────────────────────────────────────────────────
        self.create_subscription(PoseStamped, "/grasp_pose", self._grasp_cb, 10)

        # ── Randomize box positions at startup ────────────────────────────
        self._init_timer = self.create_timer(3.0, self._randomize_boxes)

        self.get_logger().info(
            f"gz_attach_node ready — world='{self._world}'  "
            f"gripper='{self._grip_link}'  track={1.0/self._track_dt:.0f}Hz"
        )

    # ── Startup randomization ────────────────────────────────────────────

    def _randomize_boxes(self) -> None:
        """One-shot: place 5-7 random boxes on table with safe margins."""
        self._init_timer.cancel()

        table_z = 0.780
        x_min, x_max = -0.30, 0.30
        y_min, y_max = -0.15, 0.15  # keep boxes within UR5 reach

        rng = random.Random()
        n_boxes = rng.randint(5, 7)
        chosen = rng.sample(self._all_box_names, n_boxes)
        placed = []

        for name in chosen:
            l, w, h = self._box_dims[name]
            r = math.hypot(l, w) / 2.0 + 0.08
            ok = False
            for _ in range(200):
                x = rng.uniform(x_min, x_max)
                y = rng.uniform(y_min, y_max)
                yaw = rng.uniform(-math.pi, math.pi)
                if all(math.hypot(x - px, y - py) >= r + pr for px, py, pr in placed):
                    qz = math.sin(yaw / 2.0)
                    qw = math.cos(yaw / 2.0)
                    _gz_set_pose(self._world, name, x, y, table_z + h / 2, qz=qz, qw=qw)
                    placed.append((x, y, r))
                    ok = True
                    break
            if not ok:
                _gz_set_pose(self._world, name, 2.0, 2.0, table_z + h / 2)

        # Park unchosen boxes off-table
        for name in self._all_box_names:
            if name not in chosen:
                l, w, h = self._box_dims[name]
                _gz_set_pose(self._world, name, 2.0, 2.0, table_z + h / 2)

        self.get_logger().info(
            f"Placed {len(placed)}/{n_boxes} boxes on table"
        )

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _grasp_cb(self, msg: PoseStamped) -> None:
        self._last_grasp_gz = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        )

    def _attach_cb(self, request, response):
        if self._attached_model is not None:
            response.success = False
            response.message = f"already attached to {self._attached_model}"
            return response

        model = self._find_nearest_box()
        if model is None:
            response.success = False
            response.message = "no box found near grasp pose"
            return response

        self._attached_model = model
        self._track_stop.clear()
        self._track_thread = threading.Thread(
            target=self._tracking_loop, args=(model,), daemon=True
        )
        self._track_thread.start()

        response.success = True
        response.message = f"attached to {model}"
        self.get_logger().info(f"Attached '{model}'")
        return response

    def _detach_cb(self, request, response):
        if self._attached_model is None:
            response.success = False
            response.message = "nothing attached"
            return response

        model = self._attached_model
        self._track_stop.set()
        if self._track_thread is not None:
            self._track_thread.join(timeout=2.0)

        # Drop into a 2×3 grid inside the tray (up to 6 boxes, no overlap).
        # Tray spans roughly x ∈ [0.55, 1.10], y ∈ [-0.37, +0.37].
        # Cell centers picked so boxes (max ~0.16m wide) fit without touching.
        TRAY_GRID = [
            (0.58, -0.30),  # 1
            (0.58, -0.18),  # 2
            (0.58, -0.06),  # 3
            (0.70, -0.24),  # 4
        ]
        h = self._box_dims.get(model, (0.1, 0.1, 0.06))[2]
        tray_z = 0.720
        idx = len(self._placed_boxes) % len(TRAY_GRID)
        x, y = TRAY_GRID[idx]
        _gz_set_pose(self._world, model, x, y, tray_z + h / 2)
        self.get_logger().info(
            f"Placed '{model}' into tray cell {idx + 1}/6 at ({x:.2f}, {y:.2f})"
        )

        self._attached_model = None
        self._track_thread = None
        self._placed_boxes.add(model)

        response.success = True
        response.message = f"detached {model}"
        self.get_logger().info(f"Detached '{model}' → dropped to tray")
        return response

    # ── Helpers ────────────────────────────────────────────────────────────

    def _discover_boxes(self) -> None:
        try:
            result = subprocess.run(
                ["gz", "topic", "-e", "-t",
                 f"/world/{self._world}/pose/info", "-n", "1"],
                capture_output=True, text=True, timeout=5.0,
            )
            if result.returncode != 0:
                return
            lines = result.stdout.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('name: "box_'):
                    model_name = line.split('"')[1]
                    x = y = z = 0.0
                    for j in range(i + 1, min(i + 15, len(lines))):
                        l = lines[j].strip()
                        if l.startswith('x:'):
                            x = float(l.split(':')[1])
                        elif l.startswith('y:'):
                            y = float(l.split(':')[1])
                        elif l.startswith('z:'):
                            z = float(l.split(':')[1])
                            break
                    if model_name not in self._placed_boxes:
                        self._box_poses[model_name] = (x, y, z)
                i += 1
        except Exception:
            pass

    def _find_nearest_box(self) -> Optional[str]:
        # Prefer current gripper TF position (robot is already at grasp pose),
        # fall back to YOLO estimate only if TF is unavailable.
        gripper_pos = self._get_gripper_gz_pose()
        if gripper_pos is not None:
            gx, gy, _ = gripper_pos
        elif self._last_grasp_gz is not None:
            gx, gy, _ = self._last_grasp_gz
        else:
            return None
        self._discover_boxes()
        best_name = None
        best_dist = float("inf")
        for name, (bx, by, bz) in self._box_poses.items():
            if name in self._placed_boxes:
                continue
            d = math.sqrt((gx - bx) ** 2 + (gy - by) ** 2)
            if d < best_dist:
                best_dist = d
                best_name = name
        if best_name is not None:
            self.get_logger().info(
                f"Nearest box: '{best_name}' dist={best_dist:.3f}m"
            )
        if best_dist > 0.5:
            self.get_logger().warn(
                f"Nearest box '{best_name}' is {best_dist:.2f}m away — "
                "too far to attach (all on-table boxes may be gone)"
            )
            return None
        return best_name

    def _get_gripper_gz_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            t = self._tf_buf.lookup_transform(
                "world", self._grip_link,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except TransformException:
            return None
        tx = t.transform.translation.x + self._spawn[0]
        ty = t.transform.translation.y + self._spawn[1]
        tz = t.transform.translation.z + self._spawn[2] - self._gz_z_off
        return tx, ty, tz

    def _tracking_loop(self, model: str) -> None:
        self.get_logger().info(f"Tracking '{model}' at {1.0/self._track_dt:.0f}Hz")
        # Never let the box sink below table top + half its height,
        # otherwise Gazebo's collision solver will eject it through the table.
        table_top = 0.780
        h = self._box_dims.get(model, (0.1, 0.1, 0.06))[2]
        z_floor = table_top + h / 2 + 0.005
        while not self._track_stop.is_set():
            pos = self._get_gripper_gz_pose()
            if pos is not None:
                x, y, z = pos
                z = max(z, z_floor)
                _gz_set_pose(self._world, model, x, y, z)
                self._box_poses[model] = (x, y, z)
            self._track_stop.wait(timeout=self._track_dt)
        self.get_logger().info(f"Stopped tracking '{model}'")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GzAttachNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
