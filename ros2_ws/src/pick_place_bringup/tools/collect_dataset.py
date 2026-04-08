#!/usr/bin/env python3
"""
collect_dataset.py — Automated dataset collection from Gazebo for YOLOv8-OBB.

Run INSIDE ros2-gazebo container after sim.launch.py is already running:

    source /opt/ros/jazzy/setup.bash
    source /ros2_ws/install/setup.bash
    python3 /ros2_ws/src/pick_place_bringup/tools/collect_dataset.py [--samples 600]

Algorithm per sample:
  1. Randomly place all 10 boxes on the table (non-overlapping)
  2. Wait for Gazebo physics to settle (1.0 s)
  3. Capture one frame from /camera/image_raw
  4. Read actual poses from /world/pick_and_place/pose/info
  5. Build YOLO OBB annotations from actual poses
  6. Save image + annotation

Output  →  /worlds/dataset/
    images/train/*.png     labels/train/*.txt
    images/val/*.png       labels/val/*.txt
    dataset.yaml

YOLO OBB annotation format (one line per visible box):
    class_id  x1 y1  x2 y2  x3 y3  x4 y4
    (4 clockwise corners, coordinates normalised to [0,1])

Camera model (from camera_info.yaml):
    position (0, 0, 1.90),  looking straight down
    fx = fy = 554.26,  cx = 320,  cy = 240,  640 × 480
    projection:  u = cx + X/D * fx
                 v = cy − Y/D * fy      (y_sign = −1, from pose_estimator)
    where D = 1.90 − Z_point
"""

import argparse
import math
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


# ── Scene constants ──────────────────────────────────────────────────────────

WORLD      = "pick_and_place"
CAM_Z      = 1.90
TABLE_Z    = 0.780
IMG_W, IMG_H = 640, 480
FX = FY    = 554.26
CX, CY     = 320.0, 240.0

# Box catalogue:  name → (length, width, height, class_id)
#   length = dimension along X when yaw = 0
#   width  = dimension along Y when yaw = 0
#   All boxes belong to class 0 ("box") — arbitrary sizes supported.
BOX_CATALOGUE = {
    "box_01": (0.05, 0.04, 0.03, 0),
    "box_02": (0.07, 0.05, 0.04, 0),
    "box_03": (0.08, 0.06, 0.05, 0),
    "box_04": (0.10, 0.07, 0.06, 0),
    "box_05": (0.12, 0.09, 0.07, 0),
    "box_06": (0.14, 0.10, 0.07, 0),
    "box_07": (0.16, 0.12, 0.08, 0),
    "box_08": (0.18, 0.13, 0.09, 0),
    "box_09": (0.20, 0.15, 0.10, 0),
    "box_10": (0.23, 0.17, 0.12, 0),
}

CLASS_NAMES = ["box"]

# Safe placement zone on the table surface (Gazebo world X, Y in metres)
# Chosen to keep boxes inside camera FOV and away from robot mount area
TABLE_X_MIN, TABLE_X_MAX = -0.38, 0.38
TABLE_Y_MIN, TABLE_Y_MAX = -0.18, 0.33

# Minimum centre-to-centre separation between any two boxes (m)
# Reduced to 0.10 to fit 10 boxes in the table zone without excessive fallbacks
MIN_SEP = 0.10

# Train / val split
VAL_FRACTION = 0.20


# ── Projection helpers ───────────────────────────────────────────────────────

def world_to_pixel(x: float, y: float, z: float) -> Tuple[float, float]:
    """
    Project a 3-D Gazebo world point to image pixel (u, v).

    Camera pose: (0, 0, 1.90), pitch=π/2 around Y → camera +X axis = world -Z (looking down).
    Empirically determined axis mapping:
        u = cx - Y_world / D * fx   (+Y_world → LEFT,  −Y_world → RIGHT)
        v = cy - X_world / D * fy   (+X_world → UP,    −X_world → DOWN)
    where D = CAM_Z − z_point.
    """
    D = CAM_Z - z
    if D <= 0:
        return -1.0, -1.0
    u = CX - y / D * FX
    v = CY - x / D * FY
    return u, v


def box_corners_world(cx: float, cy: float, z_top: float,
                      length: float, width: float,
                      yaw: float) -> List[Tuple[float, float, float]]:
    """
    Return 4 top-surface corners of a box in Gazebo world frame.

    Corner order (clockwise when viewed from above):
        0: (−l/2, −w/2)   1: (+l/2, −w/2)
        3: (−l/2, +w/2)   2: (+l/2, +w/2)
    Clockwise sequence:  0 → 1 → 2 → 3
    """
    hl, hw = length / 2.0, width / 2.0
    local = [(-hl, -hw), (+hl, -hw), (+hl, +hw), (-hl, +hw)]
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    corners = []
    for lx, ly in local:
        wx = cx + cos_y * lx - sin_y * ly
        wy = cy + sin_y * lx + cos_y * ly
        corners.append((wx, wy, z_top))
    return corners


def project_box(cx: float, cy: float,
                length: float, width: float, height: float,
                yaw: float, class_id: int) -> Optional[str]:
    """
    Build one YOLO OBB annotation line for a box.

    Returns None if the box is mostly outside the frame.
    """
    z_top = TABLE_Z + height
    corners_3d = box_corners_world(cx, cy, z_top, length, width, yaw)

    pixels = [world_to_pixel(wx, wy, wz) for wx, wy, wz in corners_3d]

    # Discard box if any corner is more than half-image outside frame
    in_frame = sum(
        1 for u, v in pixels
        if -IMG_W // 2 <= u <= IMG_W * 1.5 and -IMG_H // 2 <= v <= IMG_H * 1.5
    )
    if in_frame < 3:
        return None

    # Discard box if it's too small in pixels (nearly invisible → bad label)
    us = [u for u, v in pixels]
    vs = [v for u, v in pixels]
    bbox_w = max(us) - min(us)
    bbox_h = max(vs) - min(vs)
    if bbox_w < 20 or bbox_h < 20:
        return None

    # Normalise to [0, 1]
    pts_norm = " ".join(
        f"{u / IMG_W:.6f} {v / IMG_H:.6f}" for u, v in pixels
    )
    return f"{class_id} {pts_norm}"


# ── Placement helpers ────────────────────────────────────────────────────────

def random_pose(rng: random.Random) -> Tuple[float, float, float]:
    """Return a random (x, y, yaw) within the safe table zone."""
    x   = rng.uniform(TABLE_X_MIN, TABLE_X_MAX)
    y   = rng.uniform(TABLE_Y_MIN, TABLE_Y_MAX)
    yaw = rng.uniform(-math.pi, math.pi)
    return x, y, yaw


def _box_radius(length: float, width: float) -> float:
    """Half-diagonal of box footprint + 10 cm safety margin."""
    return math.hypot(length, width) / 2.0 + 0.10


def overlaps(x1: float, y1: float, r1: float,
             placed: List[Tuple[float, float, float]]) -> bool:
    """Return True if circle (x1,y1,r1) overlaps any already-placed circle."""
    for x2, y2, r2 in placed:
        if math.hypot(x1 - x2, y1 - y2) < (r1 + r2):
            return True
    return False


def sample_poses(rng: random.Random,
                 box_names: List[str],
                 max_tries: int = 300) -> List[Tuple[float, float, float]]:
    """
    Sample non-overlapping random poses, one per box.
    Uses each box's actual footprint so large boxes get more clearance.
    """
    poses: List[Tuple[float, float, float]] = []
    placed: List[Tuple[float, float, float]] = []  # (x, y, radius)
    for name in box_names:
        length, width, _, _ = BOX_CATALOGUE[name]
        r = _box_radius(length, width)
        for _ in range(max_tries):
            x = rng.uniform(TABLE_X_MIN, TABLE_X_MAX)
            y = rng.uniform(TABLE_Y_MIN, TABLE_Y_MAX)
            yaw = rng.uniform(-math.pi, math.pi)
            if not overlaps(x, y, r, placed):
                poses.append((x, y, yaw))
                placed.append((x, y, r))
                break
        else:
            # Fallback: skip this box entirely — better than bad annotation
            poses.append(None)
            placed.append((x, y, r))
    return poses


# ── Gazebo communication ─────────────────────────────────────────────────────

def gz_set_pose(model: str, x: float, y: float, z: float,
                yaw: float) -> bool:
    """
    Teleport a Gazebo model via the UserCommands plugin service.
    Returns False if the service call fails (boxes stay in place — that's OK,
    we always annotate from actual poses read back via gz topic).
    """
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    req = (
        f'name: "{model}" '
        f'position {{ x: {x:.5f} y: {y:.5f} z: {z:.5f} }} '
        f'orientation {{ x: 0 y: 0 z: {qz:.6f} w: {qw:.6f} }}'
    )
    try:
        result = subprocess.run(
            [
                "gz", "service",
                "-s", f"/world/{WORLD}/set_pose",
                "--reqtype", "gz.msgs.Pose",
                "--reptype", "gz.msgs.Boolean",
                "--timeout", "500",
                "--req", req,
            ],
            capture_output=True, text=True, timeout=2.0,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def gz_read_poses() -> dict:
    """
    Read actual model poses from /world/<world>/pose/info gz topic.
    Returns dict: model_name → (x, y, yaw).
    This is the ground-truth source for annotations regardless of whether
    gz_set_pose succeeded.
    """
    import re
    try:
        result = subprocess.run(
            ["gz", "topic", "-e", "-t",
             f"/world/{WORLD}/pose/info", "-n", "1"],
            capture_output=True, text=True, timeout=5.0,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}

    text = result.stdout
    poses = {}
    # Split by pose blocks
    blocks = re.split(r'(?=\bpose\s*\{)', text)
    for block in blocks:
        name_m = re.search(r'name:\s*"([^"]+)"', block)
        if not name_m:
            continue
        name = name_m.group(1)
        if name not in BOX_CATALOGUE:
            continue
        x_m = re.search(r'position\s*\{[^}]*x:\s*([-\d.e+]+)', block)
        y_m = re.search(r'position\s*\{[^}]*y:\s*([-\d.e+]+)', block)
        qz_m = re.search(r'orientation\s*\{[^}]*z:\s*([-\d.e+]+)', block)
        qw_m = re.search(r'orientation\s*\{[^}]*w:\s*([-\d.e+]+)', block)
        x   = float(x_m.group(1))  if x_m  else 0.0
        y   = float(y_m.group(1))  if y_m  else 0.0
        qz  = float(qz_m.group(1)) if qz_m else 0.0
        qw  = float(qw_m.group(1)) if qw_m else 1.0
        yaw = 2.0 * math.atan2(qz, qw)
        poses[name] = (x, y, yaw)
    return poses


# ── ROS 2 image capture ──────────────────────────────────────────────────────

class ImageCapture(Node):
    """Minimal ROS 2 node: grab one frame from /camera/image_raw on demand."""

    def __init__(self):
        super().__init__("dataset_collector")
        self._latest: Optional[np.ndarray] = None
        self.create_subscription(Image, "/camera/image_raw", self._cb, 1)

    def _cb(self, msg: Image) -> None:
        # Convert ROS Image (RGB8) → numpy BGR (for cv2.imwrite)
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1
        )
        if arr.shape[2] == 3:
            self._latest = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            self._latest = arr[:, :, :3].copy()

    def capture(self, timeout: float = 5.0) -> Optional[np.ndarray]:
        """Spin until a new frame arrives; return the numpy image or None."""
        self._latest = None
        deadline = time.time() + timeout
        while self._latest is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self._latest


# ── Dataset saving ───────────────────────────────────────────────────────────

def save_sample(img: np.ndarray, annotations: List[str],
                idx: int, split: str, out_dir: Path) -> None:
    name = f"frame_{idx:06d}"
    img_path = out_dir / "images" / split / f"{name}.png"
    lbl_path = out_dir / "labels" / split / f"{name}.txt"
    cv2.imwrite(str(img_path), img)
    with open(lbl_path, "w") as f:
        f.write("\n".join(annotations) + ("\n" if annotations else ""))


def write_yaml(out_dir: Path) -> None:
    yaml_text = f"""\
# YOLOv8-OBB dataset — boxes on table, collected from Gazebo Harmonic
path: {out_dir}
train: images/train
val:   images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    (out_dir / "dataset.yaml").write_text(yaml_text)
    print(f"Wrote {out_dir / 'dataset.yaml'}")


# ── Main collection loop ─────────────────────────────────────────────────────

def collect(n_samples: int, out_dir: Path, seed: int = 42) -> None:
    rng = random.Random(seed)

    # Create output directories
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ROS 2 node
    rclpy.init()
    node = ImageCapture()
    print("Waiting for first camera frame …")
    frame = node.capture(timeout=10.0)
    if frame is None:
        print("ERROR: no frame received. Is sim.launch.py running?")
        rclpy.shutdown()
        sys.exit(1)
    print(f"Camera OK ({frame.shape[1]}×{frame.shape[0]}). Starting collection …\n")

    all_box_names = list(BOX_CATALOGUE.keys())

    for i in range(n_samples):
        split = "val" if rng.random() < VAL_FRACTION else "train"

        # ── 1. Pick 5–7 random boxes for this frame ────────────────────────
        n_boxes = rng.randint(5, 7)
        box_names = rng.sample(all_box_names, n_boxes)

        # ── 2. Sample random poses (size-aware, no interpenetration) ──────
        poses = sample_poses(rng, box_names)

        # Skip boxes that couldn't be placed without overlap
        active = [(name, pose) for name, pose in zip(box_names, poses)
                  if pose is not None]
        if len(active) < 3:
            continue

        # ── 3. Move boxes in Gazebo (active on table, rest parked off-scene)
        active_names = {name for name, _ in active}

        def _move(item):
            name, pose = item
            length, width, height, _ = BOX_CATALOGUE[name]
            z_center = TABLE_Z + height / 2.0
            if pose is not None and name in active_names:
                bx, by, byaw = pose
                gz_set_pose(name, bx, by, z_center, byaw)
            else:
                # Park unused box far off-table (outside camera FOV)
                gz_set_pose(name, 2.0, 2.0, z_center, 0.0)

        all_items = list(zip(box_names, poses)) + [
            (name, None) for name in all_box_names if name not in box_names
        ]
        with ThreadPoolExecutor(max_workers=10) as ex:
            list(ex.map(_move, all_items))

        # ── 3. Wait for set_pose to execute on Gazebo side ────────────────
        # set_pose CLI may timeout (200ms) but the service executes async.
        # 2.0s gives physics engine time to settle without boxes sliding together.
        time.sleep(2.0)

        # ── 4. Capture frame ───────────────────────────────────────────────
        frame = node.capture(timeout=1.0)
        if frame is None:
            print(f"  [{i+1}/{n_samples}] WARNING: no frame — skipping")
            continue

        # ── 5. Read ACTUAL poses (ground truth for annotations) ────────────
        # Commanded poses may differ from actual if set_pose executed late.
        actual_poses = gz_read_poses()
        if not actual_poses:
            print(f"  [{i+1}/{n_samples}] WARNING: could not read poses — skipping")
            continue

        # ── 7. Build annotations from ACTUAL poses ─────────────────────────
        annotations = []
        for name in box_names:
            if name not in actual_poses:
                continue
            bx, by, byaw = actual_poses[name]
            length, width, height, class_id = BOX_CATALOGUE[name]
            ann = project_box(bx, by, length, width, height, byaw, class_id)
            if ann is not None:
                annotations.append(ann)

        # ── 8. Save ────────────────────────────────────────────────────────
        save_sample(frame, annotations, i, split, out_dir)

        if (i + 1) % 50 == 0 or i == 0:
            n_train = len(list((out_dir / "images" / "train").glob("*.png")))
            n_val   = len(list((out_dir / "images" / "val").glob("*.png")))
            print(f"  [{i+1:4d}/{n_samples}]  train={n_train}  val={n_val}  "
                  f"anns={len(annotations)}")

    write_yaml(out_dir)
    rclpy.shutdown()

    n_train = len(list((out_dir / "images" / "train").glob("*.png")))
    n_val   = len(list((out_dir / "images" / "val").glob("*.png")))
    print(f"\nDone!  train={n_train}  val={n_val}  total={n_train+n_val}")
    print(f"Dataset saved to: {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect YOLOv8-OBB dataset from Gazebo")
    parser.add_argument("--samples", type=int, default=600,
                        help="Number of images to collect (default: 600)")
    parser.add_argument("--output",  type=str, default="/worlds/dataset",
                        help="Output directory (default: /worlds/dataset)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    print(f"Collecting {args.samples} samples → {out_dir}")
    collect(args.samples, out_dir, args.seed)


if __name__ == "__main__":
    main()
