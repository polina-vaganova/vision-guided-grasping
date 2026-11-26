#!/usr/bin/env python3
"""
pose_estimator.py — STEREO 2D bbox → 3D grasp-pose estimator.

Subscribes:
  /detections/boxes_left   (pick_place_msgs/BoxDetectionArray)  — LEFT camera
  /detections/boxes_right  (pick_place_msgs/BoxDetectionArray)  — RIGHT camera
  /camera/camera_info      (sensor_msgs/CameraInfo)             — intrinsics (same K for both)

Publishes:
  /grasp_poses  (geometry_msgs/PoseArray)    — all detected objects (world frame)
  /grasp_pose   (geometry_msgs/PoseStamped)  — highest-confidence detection

Stereo algorithm:
  Baseline is along world X (left cam at -B/2, right cam at +B/2). Because
  the overhead cameras are pitched by π/2, image u ↔ world Y and image v ↔
  world X. So for a box at world X_obj, disparity appears in the V direction:

      Δv = v_left − v_right = −B · fy / D           (signed, typically < 0)
      D  = B · fy / |Δv|                            (depth from camera plane)
      z_top = camera_z − D                          (top surface of box)
      h_box = z_top − table_z                       (box height)

  XY is back-projected from LEFT camera (same as mono pipeline).

Matching:
  For each LEFT detection, find the RIGHT detection whose u (≡ world Y)
  differs by less than MATCH_U_TOL. If multiple candidates, pick the one
  minimising Δu. If no match: fall back to the mono-heuristic height
  estimation (bbox size → height).

Sign-convention parameters (x_sign, y_sign, yaw_sign) are same as mono version.
"""

import math
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Quaternion

from pick_place_msgs.msg import BoxDetection, BoxDetectionArray


# ── Fallback box-height estimation (when no stereo match) ───────────────────
_HEIGHT_RATIO  = 0.55
_HEIGHT_MIN_M  = 0.02
_HEIGHT_MAX_M  = 0.15


def _estimate_height_mono(fw_m: float, fh_m: float) -> float:
    h = min(fw_m, fh_m) * _HEIGHT_RATIO
    return max(_HEIGHT_MIN_M, min(_HEIGHT_MAX_M, h))


def _yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0; q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


# ── Stereo matching tolerances ───────────────────────────────────────────────
MATCH_U_TOL_PX     = 15.0   # max Δu between matched left/right detections
MATCH_TIMESTAMP_S  = 0.20   # max age of right detections to consider
RIGHT_BUFFER_SIZE  = 10     # how many recent right frames to keep
# Physical disparity range for boxes on our table:
#   B=0.40m, fy=554, D_box_top ∈ [1.27m (max box), 1.40m (table)]
#   → disparity ∈ [158, 175] px.  Allow ±30% margin for calibration slack.
DISPARITY_MIN_PX   = 130.0
DISPARITY_MAX_PX   = 220.0
# Box physical height range (for sanity-checking stereo result)
Z_TOP_MIN_M        = 0.80   # table + 0.02m
Z_TOP_MAX_M        = 0.93   # table + 0.15m


class PoseEstimatorNode(Node):

    def __init__(self):
        super().__init__("pose_estimator")

        # ── Scene geometry ─────────────────────────────────────────────────
        # LEFT camera pose in world (used for XY back-projection).
        self.declare_parameter("camera_x",  -0.20)
        self.declare_parameter("camera_y",   0.0)
        self.declare_parameter("camera_z",   2.20)
        # Stereo baseline: distance between left and right camera X positions.
        self.declare_parameter("stereo_baseline_m", 0.40)
        self.declare_parameter("table_z",           0.780)
        self.declare_parameter("gripper_standoff",  0.05)
        self.declare_parameter("grasp_frame",       "world")

        # Projection sign parameters
        self.declare_parameter("x_sign",    1.0)
        self.declare_parameter("y_sign",   -1.0)
        self.declare_parameter("yaw_sign",  1.0)

        # Table workspace bounds (world frame)
        self.declare_parameter("table_x_min", -0.35)
        self.declare_parameter("table_x_max",  0.35)
        self.declare_parameter("table_y_min", -0.20)
        self.declare_parameter("table_y_max",  0.20)

        # Minimum detection confidence
        self.declare_parameter("min_confidence", 0.70)

        self._cam_x    = self.get_parameter("camera_x").value
        self._cam_y    = self.get_parameter("camera_y").value
        self._cam_z    = self.get_parameter("camera_z").value
        self._baseline = self.get_parameter("stereo_baseline_m").value
        self._tbl_z    = self.get_parameter("table_z").value
        self._standoff = self.get_parameter("gripper_standoff").value
        self._frame    = self.get_parameter("grasp_frame").value
        self._x_sign   = self.get_parameter("x_sign").value
        self._y_sign   = self.get_parameter("y_sign").value
        self._yaw_sign = self.get_parameter("yaw_sign").value
        self._x_min = self.get_parameter("table_x_min").value
        self._x_max = self.get_parameter("table_x_max").value
        self._y_min = self.get_parameter("table_y_min").value
        self._y_max = self.get_parameter("table_y_max").value
        self._min_conf = self.get_parameter("min_confidence").value

        # Intrinsics (populated on first CameraInfo)
        self._fx = self._fy = self._cx = self._cy = None

        # Right-cam detection buffer for matching
        # deque of (timestamp_float, list[BoxDetection])
        self._right_buf = deque(maxlen=RIGHT_BUFFER_SIZE)

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            CameraInfo, "/camera/camera_info", self._camera_info_cb,
            QoSPresetProfiles.SENSOR_DATA.value,
        )
        self.create_subscription(
            BoxDetectionArray, "/detections/boxes_right",
            self._right_cb, 10,
        )
        self.create_subscription(
            BoxDetectionArray, "/detections/boxes_left",
            self._left_cb, 10,
        )

        # ── Publishers ────────────────────────────────────────────────────
        self._pub_array = self.create_publisher(PoseArray,   "/grasp_poses", 10)
        self._pub_best  = self.create_publisher(PoseStamped, "/grasp_pose",  10)

        # Stats
        self._count_stereo = 0
        self._count_mono_fallback = 0

        self.get_logger().info(
            f"pose_estimator (STEREO) ready  "
            f"left_cam=({self._cam_x}, {self._cam_y}, {self._cam_z})  "
            f"baseline={self._baseline} m  table_z={self._tbl_z}  "
            f"standoff={self._standoff}"
        )

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self._fx is None:
            K = msg.k
            self._fx = K[0]; self._cx = K[2]
            self._fy = K[4]; self._cy = K[5]
            self.get_logger().info(
                f"Intrinsics: fx={self._fx:.2f} fy={self._fy:.2f} "
                f"cx={self._cx:.1f} cy={self._cy:.1f}"
            )

    def _right_cb(self, msg: BoxDetectionArray) -> None:
        """Buffer right-camera detections for later stereo matching."""
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # Filter low-confidence and keep only dims needed for matching
        kept = [d for d in msg.detections if d.confidence >= self._min_conf]
        self._right_buf.append((t, kept))

    def _left_cb(self, msg: BoxDetectionArray) -> None:
        """Main driver: match each left detection to a right one, publish poses."""
        if self._fx is None:
            self.get_logger().warn(
                "CameraInfo not received yet — skipping frame",
                throttle_duration_sec=5.0,
            )
            return
        if not msg.detections:
            return

        t_left = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        right_dets = self._get_right_detections_near(t_left)

        poses: list[Pose] = []
        best_pose = None
        best_conf = -1.0

        for det in msg.detections:
            if det.confidence < self._min_conf:
                continue

            match = self._find_match(det, right_dets)
            pose = self._compute_pose(det, match)
            if pose is None:
                continue
            x, y = pose.position.x, pose.position.y
            if not (self._x_min <= x <= self._x_max and
                    self._y_min <= y <= self._y_max):
                continue

            poses.append(pose)
            if det.confidence > best_conf:
                best_conf = det.confidence
                best_pose = pose

        # Publish array + best pose
        pa = PoseArray()
        pa.header.stamp    = msg.header.stamp
        pa.header.frame_id = self._frame
        pa.poses = poses
        self._pub_array.publish(pa)

        if best_pose is not None:
            ps = PoseStamped()
            ps.header.stamp    = msg.header.stamp
            ps.header.frame_id = self._frame
            ps.pose = best_pose
            self._pub_best.publish(ps)
            p = best_pose.position
            self.get_logger().info(
                f"Best grasp → x={p.x:+.3f} y={p.y:+.3f} z={p.z:.3f}  "
                f"conf={best_conf:.2f}  "
                f"[stereo={self._count_stereo} mono_fallback={self._count_mono_fallback}]"
            )

    # ── Matching ───────────────────────────────────────────────────────────

    def _get_right_detections_near(self, t_left: float) -> list:
        """Return right detections from the frame closest in time to t_left."""
        if not self._right_buf:
            return []
        # Find the frame with minimum |Δt|
        best = min(self._right_buf, key=lambda x: abs(x[0] - t_left))
        if abs(best[0] - t_left) > MATCH_TIMESTAMP_S:
            return []
        return best[1]

    def _find_match(self, left_det: BoxDetection, right_dets: list):
        """Return the right BoxDetection best matching this left detection, or None."""
        if not right_dets:
            return None
        # Камера повёрнута yaw=π/2: baseline по X_world → теперь disparity по u (горизонталь).
        # Match по v (≡ world -Y, инвариантно при стереобаз. сдвиге).
        best = None
        best_dv = float("inf")
        for r in right_dets:
            dv = abs(r.y_center - left_det.y_center)
            du = r.x_center - left_det.x_center  # signed
            disp = abs(du)
            if dv > MATCH_U_TOL_PX:
                continue
            if not (DISPARITY_MIN_PX <= disp <= DISPARITY_MAX_PX):
                continue
            if dv < best_dv:
                best_dv = dv
                best = r
        return best

    # ── Geometry ───────────────────────────────────────────────────────────

    def _compute_pose(self, det_left: BoxDetection, det_right):
        """Back-project to world pose. Uses stereo disparity for Z if match available."""
        u, v = det_left.x_center, det_left.y_center
        w_px, h_px = det_left.width, det_left.height
        angle = det_left.angle

        stereo_used = False
        if det_right is not None:
            # ── Stereo depth attempt (camera rotated yaw=π/2: disparity по u) ─
            disparity = abs(u - det_right.x_center)
            D = self._baseline * self._fx / disparity
            z_top_raw = self._cam_z - D
            # Physics check: if the inferred top is outside reasonable box height,
            # reject the match and fall back to mono.
            if Z_TOP_MIN_M <= z_top_raw <= Z_TOP_MAX_M:
                z_top = z_top_raw
                h_box = z_top - self._tbl_z
                stereo_used = True
            else:
                self.get_logger().debug(
                    f"stereo rejected: z_top={z_top_raw:.3f} out of physical range",
                    throttle_duration_sec=2.0,
                )

        if not stereo_used:
            # ── Mono fallback: estimate height from bbox footprint ────────
            D_table = self._cam_z - self._tbl_z
            fw_m = w_px / self._fx * D_table
            fh_m = h_px / self._fy * D_table
            h_box = _estimate_height_mono(fw_m, fh_m)
            z_top = self._tbl_z + h_box

        # XY back-projection from LEFT camera (depth D to box centre)
        # Камера повёрнута на yaw=π/2 → u (горизонталь) → +X_world, v (вертикаль) → -Y_world
        D_center = self._cam_z - (self._tbl_z + h_box / 2.0)
        x_world = self._cam_x + self._x_sign * (u - self._cx) / self._fx * D_center
        y_world = self._cam_y + self._y_sign * (v - self._cy) / self._fy * D_center

        # Grasp Z = top of box + standoff
        z_grasp = z_top + self._standoff
        yaw = self._yaw_sign * angle

        pose = Pose()
        pose.position.x  = x_world
        pose.position.y  = y_world
        pose.position.z  = z_grasp
        pose.orientation = _yaw_to_quat(yaw)

        if stereo_used:
            self._count_stereo += 1
        else:
            self._count_mono_fallback += 1

        self.get_logger().debug(
            f"  {'STEREO' if stereo_used else 'MONO '}  "
            f"h_box={h_box:.3f}m  z_top={z_top:.3f}  "
            f"→ ({x_world:+.3f}, {y_world:+.3f}, {z_grasp:.3f})"
        )
        return pose


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
