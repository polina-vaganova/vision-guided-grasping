#!/usr/bin/env python3
"""
pose_estimator.py — 2D bounding-box → 3D grasp-pose estimator.

Subscribes:
  /detections/boxes   (pick_place_msgs/BoxDetectionArray)
  /camera/camera_info (sensor_msgs/CameraInfo)

Publishes:
  /grasp_poses  (geometry_msgs/PoseArray)    — all detected objects in world frame
  /grasp_pose   (geometry_msgs/PoseStamped)  — highest-confidence detection

Algorithm (pinhole camera, camera looking straight down):
  1. Back-project pixel centre (u, v) to world XY:
       D = camera_z − z_object_centre    (depth from camera to object plane)
       x_w = cam_x + x_sign * (u − cx) / fx * D
       y_w = cam_y + y_sign * (v − cy) / fy * D

  2. Estimate box height from projected footprint (no size class needed):
       fw_m = w_px / fx * D_table    fh_m = h_px / fy * D_table
       h_box ≈ min(fw_m, fh_m) * 0.55   (empirical ratio: height ≈ 55% of shorter side)
       Clamped to [0.02, 0.15] m.

  3. Refine depth: z_centre = table_z + h_box / 2 → recompute D and XY.

  4. Grasp Z = table_z + h_box + gripper_standoff
     Grasp yaw = yaw_sign * angle_from_YOLO

Sign-convention parameters (x_sign, y_sign, yaw_sign are ±1):
  For the Gazebo Harmonic overhead camera as configured in sim.launch.py
  (optical axis pointing in −Z_world, +u direction → +X_world):
    x_sign = +1.0, y_sign = −1.0, yaw_sign = +1.0

  If objects appear mirrored along one world axis in RViz, negate the
  corresponding sign parameter in pose_estimator_params.yaml.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Quaternion

from pick_place_msgs.msg import BoxDetection, BoxDetectionArray


# ── Box height estimation ──────────────────────────────────────────────────
# For boxes of arbitrary size, height is estimated from the shorter footprint
# dimension measured in the image.  Empirically: h ≈ 0.55 × min(l, w).

_HEIGHT_RATIO  = 0.55   # h_box ≈ ratio × min(footprint_length, footprint_width)
_HEIGHT_MIN_M  = 0.02   # clamp lower bound (m)
_HEIGHT_MAX_M  = 0.15   # clamp upper bound (m)


def _estimate_height(fw_m: float, fh_m: float) -> float:
    """Estimate box height from its two footprint dimensions (metres)."""
    h = min(fw_m, fh_m) * _HEIGHT_RATIO
    return max(_HEIGHT_MIN_M, min(_HEIGHT_MAX_M, h))


def _yaw_to_quat(yaw: float) -> Quaternion:
    """Pure yaw rotation around world Z axis → quaternion (x, y, z, w)."""
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


# ── Node ──────────────────────────────────────────────────────────────────

class PoseEstimatorNode(Node):

    def __init__(self):
        super().__init__("pose_estimator")

        # ── Scene geometry ─────────────────────────────────────────────────
        self.declare_parameter("camera_x",         0.0)   # world X of camera
        self.declare_parameter("camera_y",         0.0)   # world Y of camera
        self.declare_parameter("camera_z",         1.90)  # world Z of camera
        self.declare_parameter("table_z",          0.780) # world Z of table surface
        self.declare_parameter("gripper_standoff", 0.12)  # m above box top
        self.declare_parameter("grasp_frame",      "world")

        # ── Projection sign parameters ─────────────────────────────────────
        self.declare_parameter("x_sign",   1.0)   # +u → +x_world  → +1
        self.declare_parameter("y_sign",  -1.0)   # +v → −y_world  → −1
        self.declare_parameter("yaw_sign", 1.0)   # image angle = world yaw → +1

        # ── Table workspace bounds (Gazebo world frame) ────────────────────
        # Reject detections outside this rectangle to avoid unreachable poses
        self.declare_parameter("table_x_min", -0.35)
        self.declare_parameter("table_x_max",  0.35)
        self.declare_parameter("table_y_min", -0.20)
        self.declare_parameter("table_y_max",  0.20)

        self._cam_x   = self.get_parameter("camera_x").value
        self._cam_y   = self.get_parameter("camera_y").value
        self._cam_z   = self.get_parameter("camera_z").value
        self._tbl_z   = self.get_parameter("table_z").value
        self._standoff = self.get_parameter("gripper_standoff").value
        self._frame   = self.get_parameter("grasp_frame").value
        self._x_sign  = self.get_parameter("x_sign").value
        self._y_sign  = self.get_parameter("y_sign").value
        self._yaw_sign = self.get_parameter("yaw_sign").value
        self._x_min = self.get_parameter("table_x_min").value
        self._x_max = self.get_parameter("table_x_max").value
        self._y_min = self.get_parameter("table_y_min").value
        self._y_max = self.get_parameter("table_y_max").value

        # Camera intrinsics — populated on first CameraInfo message
        self._fx = self._fy = self._cx = self._cy = None

        # ── Subscribers ────────────────────────────────────────────────────
        self.create_subscription(
            CameraInfo,
            "/camera/camera_info",
            self._camera_info_cb,
            QoSPresetProfiles.SENSOR_DATA.value,
        )
        self.create_subscription(
            BoxDetectionArray,
            "/detections/boxes",
            self._detections_cb,
            10,
        )

        # ── Publishers ─────────────────────────────────────────────────────
        self._pub_array = self.create_publisher(PoseArray,   "/grasp_poses", 10)
        self._pub_best  = self.create_publisher(PoseStamped, "/grasp_pose",  10)

        self.get_logger().info(
            f"pose_estimator ready  "
            f"camera=({self._cam_x}, {self._cam_y}, {self._cam_z})  "
            f"table_z={self._tbl_z}  standoff={self._standoff}  "
            f"signs: x={self._x_sign:+.0f} y={self._y_sign:+.0f} "
            f"yaw={self._yaw_sign:+.0f}"
        )

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        """Latch camera intrinsics from the first CameraInfo message."""
        if self._fx is None:
            K = msg.k  # 9-element row-major intrinsic matrix K
            self._fx = K[0]; self._cx = K[2]
            self._fy = K[4]; self._cy = K[5]
            self.get_logger().info(
                f"Intrinsics loaded: "
                f"fx={self._fx:.2f} fy={self._fy:.2f} "
                f"cx={self._cx:.1f} cy={self._cy:.1f}"
            )

    def _detections_cb(self, msg: BoxDetectionArray) -> None:
        """Convert each bounding-box detection to a 3D grasp pose."""
        if self._fx is None:
            self.get_logger().warn(
                "CameraInfo not received yet — skipping frame",
                throttle_duration_sec=5.0,
            )
            return

        if not msg.detections:
            return

        poses: list[Pose] = []
        best_pose = None
        best_conf = -1.0

        # Drop low-confidence detections — they are mostly false positives on
        # an empty table (shadows, table seam, tray edge).
        MIN_CONF = 0.70
        for det in msg.detections:
            if det.confidence < MIN_CONF:
                continue
            pose = self._bbox_to_pose(det)
            if pose is None:
                continue
            x, y = pose.position.x, pose.position.y
            if not (self._x_min <= x <= self._x_max and
                    self._y_min <= y <= self._y_max):
                self.get_logger().debug(
                    f"Rejected out-of-bounds detection: ({x:+.3f}, {y:+.3f})",
                    throttle_duration_sec=2.0,
                )
                continue
            poses.append(pose)
            if det.confidence > best_conf:
                best_conf = det.confidence
                best_pose = pose

        # Publish PoseArray with all grasp candidates
        pa = PoseArray()
        pa.header.stamp    = msg.header.stamp
        pa.header.frame_id = self._frame
        pa.poses = poses
        self._pub_array.publish(pa)

        # Publish single best-confidence pose
        if best_pose is not None:
            ps = PoseStamped()
            ps.header.stamp    = msg.header.stamp
            ps.header.frame_id = self._frame
            ps.pose = best_pose
            self._pub_best.publish(ps)
            p = best_pose.position
            self.get_logger().info(
                f"Best grasp → x={p.x:+.3f}  y={p.y:+.3f}  z={p.z:.3f}  "
                f"conf={best_conf:.2f}"
            )

    # ── Geometry ───────────────────────────────────────────────────────────

    def _bbox_to_pose(self, det: BoxDetection):
        """
        Back-project one OBB detection to a 3D grasp Pose in world frame.

        Returns None only if intrinsics are missing (should not happen after
        the guard in _detections_cb).
        """
        u, v  = det.x_center, det.y_center
        w_px  = det.width
        h_px  = det.height
        angle = det.angle   # rad, normalised to (−π/2, π/2] by yolo_detector

        # ── 1. Footprint at table level (coarse D) ─────────────────────────
        D_table = self._cam_z - self._tbl_z          # 1.90 − 0.780 = 1.12 m
        fw_m    = w_px / self._fx * D_table
        fh_m    = h_px / self._fy * D_table

        # ── 2. Estimate box height from footprint dimensions ───────────────
        h_box = _estimate_height(fw_m, fh_m)

        # ── 3. Refined depth to object centre ─────────────────────────────
        # z_centre = table surface + half box height
        z_obj_centre = self._tbl_z + h_box / 2.0
        D = self._cam_z - z_obj_centre

        # ── 4. Back-project pixel → world XY ──────────────────────────────
        # Gazebo Harmonic overhead camera (pitch=π/2):
        #   image +v (row, downward) → world +X  → x_sign = -1 (flipped)
        #   image +u (col, rightward) → world -Y  → y_sign = -1
        x_world = self._cam_x + self._x_sign * (v - self._cy) / self._fy * D
        y_world = self._cam_y + self._y_sign * (u - self._cx) / self._fx * D

        # ── 5. Grasp height = top of box + gripper standoff ───────────────
        z_grasp = self._tbl_z + h_box + self._standoff

        # ── 6. Yaw from OBB angle ─────────────────────────────────────────
        yaw = self._yaw_sign * angle

        # ── 7. Assemble Pose ──────────────────────────────────────────────
        pose = Pose()
        pose.position.x  = x_world
        pose.position.y  = y_world
        pose.position.z  = z_grasp
        pose.orientation = _yaw_to_quat(yaw)

        self.get_logger().debug(
            f"  px=({u:.0f},{v:.0f})  "
            f"fw={fw_m:.3f}m fh={fh_m:.3f}m  h_est={h_box:.3f}m  D={D:.3f}m  "
            f"→ ({x_world:+.3f}, {y_world:+.3f}, {z_grasp:.3f})  "
            f"yaw={yaw:.3f} rad"
        )
        return pose


# ── Entry point ────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
