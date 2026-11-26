#!/usr/bin/env python3
"""
gen_dataset_stereo.py — Generate stereo YOLO-OBB dataset from Gazebo.

For each scene:
  1. Randomize 1-6 boxes on table (position + yaw)
  2. Park unused boxes
  3. Let physics settle
  4. Capture LEFT and RIGHT camera frames
  5. Compute YOLO-OBB labels via forward projection of known poses
  6. Write images + labels + occasional preview

Runs INSIDE the Docker container. Assumes sim is already running with
stereo cameras at (-0.20, 0, 2.20) and (+0.20, 0, 2.20).
"""

import math
import os
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from PIL import Image as PilImage, ImageDraw

import gz.transport13 as _gz_transport

# ── Config ──────────────────────────────────────────────────────────────────

N_SCENES = int(os.environ.get("N_SCENES", "800"))
TRAIN_RATIO = 0.9
OUTPUT_DIR = "/experiments/dataset_stereo_v1"
PREVIEW_DIR = "/experiments/dataset_preview"
WORLD = "pick_and_place"

# Progress preview scenes
PREVIEW_SCENES = {0, 4, 99, 299, 499, 699, N_SCENES - 1}

# Box dimensions from SDF
BOX_DIMS = {
    "box_01": (0.05, 0.04, 0.03),
    "box_02": (0.07, 0.05, 0.04),
    "box_03": (0.08, 0.06, 0.05),
    "box_04": (0.10, 0.07, 0.06),
    "box_05": (0.12, 0.09, 0.07),
    "box_06": (0.14, 0.10, 0.07),
}
BOX_NAMES = list(BOX_DIMS.keys())

# Scene randomization
TABLE_Z = 0.780
X_MIN, X_MAX = -0.30, 0.30
Y_MIN, Y_MAX = -0.20, 0.20
PARK_X, PARK_Y = 0.0, -7.0

# Cameras (stereo pair at Z=2.20, baseline 0.40)
CAM_LEFT = (-0.20, 0.0, 2.20)
CAM_RIGHT = (+0.20, 0.0, 2.20)

# Intrinsics (identical for both cameras)
FX = FY = 554.26
CX, CY = 320.0, 240.0
IMG_W, IMG_H = 640, 480

# Sign conventions (match pose_estimator_params.yaml)
X_SIGN = -1.0
Y_SIGN = -1.0


# ── Protobuf helpers for gz set_pose ────────────────────────────────────────

def _varint(v):
    buf = []
    while v > 0x7F:
        buf.append((v & 0x7F) | 0x80)
        v >>= 7
    buf.append(v)
    return bytes(buf)


def _pb_str(field, s):
    b = s.encode()
    return _varint((field << 3) | 2) + _varint(len(b)) + b


def _pb_dbl(field, v):
    return _varint((field << 3) | 1) + struct.pack("<d", v)


def _pb_emb(field, b):
    return _varint((field << 3) | 2) + _varint(len(b)) + b


def _build_pose_msg(model, x, y, z, qx, qy, qz, qw):
    pos = _pb_dbl(2, x) + _pb_dbl(3, y) + _pb_dbl(4, z)
    ori = _pb_dbl(2, qx) + _pb_dbl(3, qy) + _pb_dbl(4, qz) + _pb_dbl(5, qw)
    return _pb_str(2, model) + _pb_emb(4, pos) + _pb_emb(5, ori)


_gz_node = _gz_transport.Node()


def gz_set_pose(model, x, y, z, yaw=0.0):
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    msg = _build_pose_msg(model, x, y, z, 0.0, 0.0, qz, qw)
    _gz_node.request_raw(
        f"/world/{WORLD}/set_pose",
        msg, "gz.msgs.Pose", "gz.msgs.Boolean", 1,
    )


# ── Projection ─────────────────────────────────────────────────────────────

def project_point(x_w, y_w, z_w, cam_pos):
    """Forward-project a world point to pixel (u, v) for given camera."""
    cam_x, cam_y, cam_z = cam_pos
    D = cam_z - z_w
    if D <= 0.01:
        return None
    u = CX + Y_SIGN * (y_w - cam_y) * FX / D
    v = CY + X_SIGN * (x_w - cam_x) * FY / D
    return (u, v)


def box_top_corners(x, y, z_top, yaw, lx, ly):
    """Return 4 corners of box top face in world coords."""
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    hx, hy = lx / 2.0, ly / 2.0
    corners_local = [(-hx, -hy), (+hx, -hy), (+hx, +hy), (-hx, +hy)]
    out = []
    for lx_c, ly_c in corners_local:
        wx = x + lx_c * cos_y - ly_c * sin_y
        wy = y + lx_c * sin_y + ly_c * cos_y
        out.append((wx, wy, z_top))
    return out


def box_obb_pixels(box, cam_pos):
    """Return 4 pixel coords of box top face OBB, or None if any corner missing."""
    lx, ly, lz = box["dim"]
    z_top = box["z"] + lz / 2.0
    corners_w = box_top_corners(box["x"], box["y"], z_top, box["yaw"], lx, ly)
    pixels = []
    for xw, yw, zw in corners_w:
        p = project_point(xw, yw, zw, cam_pos)
        if p is None:
            return None
        pixels.append(p)
    return pixels


# ── Grabber node ───────────────────────────────────────────────────────────

class Grabber(Node):
    def __init__(self):
        super().__init__("dataset_grabber")
        self.latest_left = None
        self.latest_right = None
        self._left_stamp = 0
        self._right_stamp = 0
        self.create_subscription(
            Image, "/camera_left/image_raw", self._cb_left, 10
        )
        self.create_subscription(
            Image, "/camera_right/image_raw", self._cb_right, 10
        )

    def _cb_left(self, msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )
        if msg.encoding.lower() == "bgr8":
            arr = arr[:, :, ::-1]
        self.latest_left = arr.copy()
        self._left_stamp += 1

    def _cb_right(self, msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )
        if msg.encoding.lower() == "bgr8":
            arr = arr[:, :, ::-1]
        self.latest_right = arr.copy()
        self._right_stamp += 1

    def capture_fresh_pair(self, timeout=3.0):
        """Wait for a new frame from each camera after now, return pair."""
        start_l = self._left_stamp
        start_r = self._right_stamp
        t0 = time.time()
        while (self._left_stamp == start_l or self._right_stamp == start_r) \
                and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
        return self.latest_left, self.latest_right


# ── Scene randomization ────────────────────────────────────────────────────

def randomize_scene(rng):
    """Place 1-6 random boxes on table. Return list of box dicts."""
    n = rng.randint(1, 6)
    chosen = rng.sample(BOX_NAMES, n)
    placed = []  # (x, y, r)
    active = []

    for name in chosen:
        lx, ly, lz = BOX_DIMS[name]
        r = math.hypot(lx, ly) / 2.0 + 0.05
        for _ in range(200):
            x = rng.uniform(X_MIN, X_MAX)
            y = rng.uniform(Y_MIN, Y_MAX)
            yaw = rng.uniform(-math.pi, math.pi)
            if all(math.hypot(x - px, y - py) >= r + pr
                   for px, py, pr in placed):
                z = TABLE_Z + lz / 2.0
                gz_set_pose(name, x, y, z, yaw)
                active.append({
                    "name": name, "x": x, "y": y, "z": z,
                    "yaw": yaw, "dim": BOX_DIMS[name],
                })
                placed.append((x, y, r))
                break

    # Park unused
    for name in BOX_NAMES:
        if name not in [a["name"] for a in active]:
            lx, ly, lz = BOX_DIMS[name]
            gz_set_pose(name, PARK_X, PARK_Y, TABLE_Z + lz / 2.0)

    return active


# ── Label writing + preview drawing ────────────────────────────────────────

def write_labels(active_boxes, cam_pos, path):
    """Write YOLO-OBB label file (normalized, 4 corners)."""
    lines = []
    for box in active_boxes:
        pixels = box_obb_pixels(box, cam_pos)
        if pixels is None:
            continue
        # Accept if at least the center is roughly in the frame
        u_c = sum(p[0] for p in pixels) / 4.0
        v_c = sum(p[1] for p in pixels) / 4.0
        if not (0 <= u_c <= IMG_W and 0 <= v_c <= IMG_H):
            continue
        parts = ["0"]
        for u, v in pixels:
            un = max(0.0, min(1.0, u / IMG_W))
            vn = max(0.0, min(1.0, v / IMG_H))
            parts.append(f"{un:.6f}")
            parts.append(f"{vn:.6f}")
        lines.append(" ".join(parts))
    with open(path, "w") as f:
        if lines:
            f.write("\n".join(lines) + "\n")


def render_preview(img, active_boxes, cam_pos, path):
    """Save image with OBB overlays to verify labels visually."""
    pil = PilImage.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(pil)
    for box in active_boxes:
        pixels = box_obb_pixels(box, cam_pos)
        if pixels is None:
            continue
        poly = [(max(0, min(IMG_W - 1, int(u))),
                 max(0, min(IMG_H - 1, int(v))))
                for u, v in pixels]
        draw.polygon(poly, outline=(0, 255, 0), width=2)
        cx = sum(p[0] for p in poly) / 4
        cy = sum(p[1] for p in poly) / 4
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(255, 0, 0))
    pil.save(path)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    grabber = Grabber()

    for split in ("train", "val"):
        os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    rng = random.Random(42)
    n_train = int(N_SCENES * TRAIN_RATIO)

    print(f"Warming up — waiting for first camera frames…", flush=True)
    for _ in range(60):
        rclpy.spin_once(grabber, timeout_sec=0.1)
        if grabber.latest_left is not None and grabber.latest_right is not None:
            break
    if grabber.latest_left is None or grabber.latest_right is None:
        print("ERROR: no camera frames received", flush=True)
        return 1
    print("Cameras ready.", flush=True)

    t0 = time.time()

    for i in range(N_SCENES):
        active = randomize_scene(rng)
        time.sleep(0.25)  # settle physics

        left, right = grabber.capture_fresh_pair(timeout=3.0)
        if left is None or right is None:
            print(f"[{i}] CAPTURE FAILED, skipping", flush=True)
            continue

        split = "train" if i < n_train else "val"
        base = f"scene_{i:04d}"

        # LEFT
        PilImage.fromarray(left).save(
            f"{OUTPUT_DIR}/images/{split}/{base}_left.png"
        )
        write_labels(
            active, CAM_LEFT,
            f"{OUTPUT_DIR}/labels/{split}/{base}_left.txt",
        )
        # RIGHT
        PilImage.fromarray(right).save(
            f"{OUTPUT_DIR}/images/{split}/{base}_right.png"
        )
        write_labels(
            active, CAM_RIGHT,
            f"{OUTPUT_DIR}/labels/{split}/{base}_right.txt",
        )

        # Preview at selected scenes
        if i in PREVIEW_SCENES:
            render_preview(
                left, active, CAM_LEFT,
                f"{PREVIEW_DIR}/preview_{i:04d}_left.png",
            )
            render_preview(
                right, active, CAM_RIGHT,
                f"{PREVIEW_DIR}/preview_{i:04d}_right.png",
            )

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta_min = (N_SCENES - i - 1) / rate / 60.0
            print(
                f"[{i + 1}/{N_SCENES}] n_boxes={len(active)} "
                f"rate={rate:.2f} sc/s  ETA={eta_min:.1f} min",
                flush=True,
            )

    # Write dataset.yaml
    with open(f"{OUTPUT_DIR}/dataset.yaml", "w") as f:
        f.write(
            f"path: {OUTPUT_DIR}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: 1\n"
            f"names: ['box']\n"
        )

    total = time.time() - t0
    print(
        f"\nDONE: {N_SCENES} scenes in {total / 60:.1f} min "
        f"({N_SCENES / total:.2f} sc/s)",
        flush=True,
    )
    grabber.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
