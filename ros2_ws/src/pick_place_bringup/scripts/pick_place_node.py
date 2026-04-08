#!/usr/bin/env python3
"""
pick_place_node.py — UR5 automated pick & place using MoveGroup action client.

Uses moveit_msgs/action/MoveGroup directly (no moveit_py dependency).

Subscribes:
  /grasp_pose  (geometry_msgs/PoseStamped)  — target in Gazebo world frame
    Position:    computed by pose_estimator (includes gripper standoff)
    Orientation: pure Z-rotation encoding box yaw (qz, qw only)

Publishes:
  /pick_place/status  (std_msgs/String)  — current state and metrics

Services called:
  /attach_object  (std_srvs/SetBool)  — tell gz_attach_node to fix box to gripper
  /detach_object  (std_srvs/SetBool)  — tell gz_attach_node to release box
  Both services are OPTIONAL: the node warns if unavailable but continues.

Coordinate transform — Gazebo world → MoveIt 2 planning frame:
  The UR5 URDF root ('world' link) is at robot spawn position in Gazebo.
  robot_state_publisher treats the URDF 'world' as origin (0,0,0).
  Therefore MoveIt 2 plans in a frame offset from Gazebo world by spawn position.
    p_moveit.x = p_gazebo.x − robot_spawn_x   (= p_gazebo.x,  spawn_x = 0)
    p_moveit.y = p_gazebo.y − robot_spawn_y   (= p_gazebo.y + 0.58)
    p_moveit.z = p_gazebo.z − robot_spawn_z   (= p_gazebo.z − 0.780)

Gripper orientation for top-down grasping:
  q_down = (x=1, y=0, z=0, w=0)  — 180° around world X  → tool Z points world −Z
  q_yaw  = (x=0, y=0, z=sin(θ/2), w=cos(θ/2))  — box yaw around tool Z
  q_total = q_down ⊗ q_yaw  (Hamilton product)
"""

import math
import queue
import threading
import time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_srvs.srv import SetBool

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    CollisionObject,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    OrientationConstraint,
    PlanningOptions,
    PositionConstraint,
    RobotState,
    WorkspaceParameters,
)
from shape_msgs.msg import SolidPrimitive


# ── Constants ───────────────────────────────────────────────────────────────

_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# "Ready" pose — arm raised, within table workspace
_HOME_JOINTS = [0.0, -1.9, 1.8, -1.5708, -1.5708, 0.0]

# Joint position bounds enforced as path_constraints on every plan request.
# Goal: cut off "wrap-around" IK branches (shoulder flip, elbow-down, etc.)
# so OMPL can't wander through the back of the robot.
# (center, half_range) per UR5 joint, in radians.
_JOINT_BOUNDS = {
    "shoulder_pan_joint":  (0.0,    1.5708),   # ±90°  — face the table
    "shoulder_lift_joint": (-1.57,  1.5708),   # -π…0  — keep lift "down"
    "elbow_joint":         (1.2,    1.5708),   # 0…~π  — elbow up
    "wrist_1_joint":       (-1.57,  3.1416),   # full
    "wrist_2_joint":       (-1.57,  3.1416),
    "wrist_3_joint":       (0.0,    3.1416),
}


# ── Math helpers ─────────────────────────────────────────────────────────────

def _qmul(q1: Tuple, q2: Tuple) -> Tuple:
    """Hamilton product of two quaternions in (x, y, z, w) form."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _down_yaw_quat(yaw: float,
                   qd: Tuple = (1.0, 0.0, 0.0, 0.0)) -> Tuple:
    """Quaternion for gripper pointing straight down and rotated by yaw."""
    q_yaw = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
    return _qmul(qd, q_yaw)


# ── ROS 2 node ───────────────────────────────────────────────────────────────

class PickPlaceNode(Node):

    def __init__(self):
        super().__init__("pick_place_node")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("robot_spawn_x",    0.0)
        self.declare_parameter("robot_spawn_y",   -0.58)
        self.declare_parameter("robot_spawn_z",    0.780)
        self.declare_parameter("pre_grasp_height", 0.15)
        self.declare_parameter("place_x",          0.63)
        self.declare_parameter("place_y",         -0.22)
        self.declare_parameter("place_z",          0.900)
        self.declare_parameter("planning_group",   "ur_manipulator")
        self.declare_parameter("eef_link",         "tool0")
        self.declare_parameter("planning_time",    5.0)
        self.declare_parameter("max_velocity",     0.3)
        self.declare_parameter("max_acceleration", 0.3)
        self.declare_parameter("qd_x", 1.0)
        self.declare_parameter("qd_y", 0.0)
        self.declare_parameter("qd_z", 0.0)
        self.declare_parameter("qd_w", 0.0)
        self.declare_parameter("max_consecutive_failures", 5)

        def gp(name):
            return self.get_parameter(name).value

        self._spawn     = (gp("robot_spawn_x"), gp("robot_spawn_y"), gp("robot_spawn_z"))
        self._pre_h     = gp("pre_grasp_height")
        self._place_gz  = (gp("place_x"), gp("place_y"), gp("place_z"))
        self._group     = gp("planning_group")
        self._eef       = gp("eef_link")
        self._plan_t    = gp("planning_time")
        self._max_vel   = gp("max_velocity")
        self._max_acc   = gp("max_acceleration")
        self._qd        = (gp("qd_x"), gp("qd_y"), gp("qd_z"), gp("qd_w"))
        self._max_consec_fail = int(gp("max_consecutive_failures"))

        # ── Internal state ─────────────────────────────────────────────────
        self._q: queue.Queue[PoseStamped] = queue.Queue(maxsize=1)
        self._total   = 0
        self._success = 0
        self._consec_fail = 0

        # ── Publishers / subscribers ───────────────────────────────────────
        self._status_pub = self.create_publisher(String, "/pick_place/status", 10)
        self.create_subscription(PoseStamped, "/grasp_pose", self._grasp_cb, 10)
        self._co_pub = self.create_publisher(CollisionObject, "/collision_object", 10)

        # ── MoveGroup action client ──────────────────────────────────────
        self._move_client = ActionClient(self, MoveGroup, "move_action")

        # ── Optional service clients (attach / detach) ─────────────────────
        self._attach_cli = self.create_client(SetBool, "/attach_object")
        self._detach_cli = self.create_client(SetBool, "/detach_object")

        self.get_logger().info(
            f"pick_place_node ready — spawn={self._spawn} "
            f"group={self._group} eef={self._eef}"
        )

    def _publish_table_collision(self) -> None:
        """Add the table surface as a collision object so MoveIt avoids it."""
        # Table surface in MoveIt (world) frame:
        #   Gazebo table top z = 0.780 m  →  MoveIt z = 0.780 − spawn_z = 0.0
        #   We place a thin slab just below z=0 to block downward IK solutions.
        co = CollisionObject()
        co.header.frame_id = "world"
        co.id = "table"
        co.operation = CollisionObject.ADD

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [2.0, 2.0, 0.05]   # 2×2 m slab, 5 cm thick
        co.primitives.append(box)

        pose = _make_pose(0.0, 0.2, -0.025, 0.0, 0.0, 0.0, 1.0)  # 2.5 cm below table top
        co.primitive_poses.append(pose)

        self._co_pub.publish(co)
        self.get_logger().info("Published table collision object to planning scene")

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _grasp_cb(self, msg: PoseStamped) -> None:
        if self._q.full():
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
        self._q.put_nowait(msg)

    # ── Public API ───────────────────────────────────────────────────────

    def gz_to_moveit(self, x: float, y: float, z: float) -> Tuple:
        sx, sy, sz = self._spawn
        return x - sx, y - sy, z - sz

    def publish(self, text: str) -> None:
        self._status_pub.publish(String(data=text))
        self.get_logger().info(f"[pick_place] {text}")

    def call_service(self, attach: bool) -> bool:
        cli  = self._attach_cli  if attach else self._detach_cli
        name = "/attach_object" if attach else "/detach_object"
        if not cli.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn(f"{name} not available — skipping")
            return False
        req = SetBool.Request()
        req.data = True
        fut = cli.call_async(req)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 6.0:
            time.sleep(0.05)
        if fut.done() and fut.result().success:
            self.get_logger().info(f"{name} OK")
            return True
        self.get_logger().warn(f"{name} failed or timed out")
        return False

    def next_pose(self, timeout: float = 120.0) -> Optional[PoseStamped]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── MoveGroup helpers ────────────────────────────────────────────────

    def _build_pose_goal(self, x: float, y: float, z: float,
                         qx: float, qy: float, qz: float, qw: float) -> MoveGroup.Goal:
        """Build a MoveGroup goal for a Cartesian pose target."""
        goal = MoveGroup.Goal()
        req = goal.request

        req.group_name = self._group
        req.num_planning_attempts = 5
        req.allowed_planning_time = self._plan_t
        req.max_velocity_scaling_factor = self._max_vel
        req.max_acceleration_scaling_factor = self._max_acc

        # Start state: current (empty = use current)
        req.start_state = RobotState()
        req.start_state.is_diff = True

        # Goal constraints: position + orientation on eef
        constraints = Constraints()

        # Position constraint
        pc = PositionConstraint()
        pc.header.frame_id = "world"
        pc.link_name = self._eef
        pc.target_point_offset.x = 0.0
        pc.target_point_offset.y = 0.0
        pc.target_point_offset.z = 0.0
        # Tolerance: small sphere around target
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.SPHERE
        prim.dimensions = [0.005]  # 5mm tolerance
        pc.constraint_region.primitives.append(prim)
        pc.constraint_region.primitive_poses.append(
            _make_pose(x, y, z, 0.0, 0.0, 0.0, 1.0)
        )
        pc.weight = 1.0
        constraints.position_constraints.append(pc)

        # Orientation constraint
        oc = OrientationConstraint()
        oc.header.frame_id = "world"
        oc.link_name = self._eef
        oc.orientation.x = qx
        oc.orientation.y = qy
        oc.orientation.z = qz
        oc.orientation.w = qw
        oc.absolute_x_axis_tolerance = 0.05  # ~3 degrees
        oc.absolute_y_axis_tolerance = 0.05
        oc.absolute_z_axis_tolerance = 0.05
        oc.weight = 1.0
        constraints.orientation_constraints.append(oc)

        req.goal_constraints.append(constraints)

        # Planning options
        goal.planning_options.plan_only = False  # plan + execute
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        return goal

    def _build_joint_goal(self, joint_values: list) -> MoveGroup.Goal:
        """Build a MoveGroup goal for a joint-space target."""
        goal = MoveGroup.Goal()
        req = goal.request

        req.group_name = self._group
        req.num_planning_attempts = 5
        req.allowed_planning_time = self._plan_t
        req.max_velocity_scaling_factor = self._max_vel
        req.max_acceleration_scaling_factor = self._max_acc

        req.start_state = RobotState()
        req.start_state.is_diff = True

        constraints = Constraints()
        for name, val in zip(_JOINT_NAMES, joint_values):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = val
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        req.goal_constraints.append(constraints)

        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        return goal

    def _wait_future(self, future, timeout: float) -> bool:
        """Poll a future until done (spin thread handles callbacks)."""
        t0 = time.time()
        while not future.done() and time.time() - t0 < timeout:
            time.sleep(0.05)
        return future.done()

    def send_goal(self, goal: MoveGroup.Goal, label: str = "") -> bool:
        """Send a MoveGroup goal and block until result. Returns True on success."""
        if not self._move_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("MoveGroup action server not available")
            return False

        self.get_logger().info(f"Sending goal: {label}")
        future = self._move_client.send_goal_async(goal)

        # Wait for goal acceptance (spin thread processes callbacks)
        if not self._wait_future(future, 10.0):
            self.get_logger().error(f"Goal send timed out: {label}")
            return False

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f"Goal rejected: {label}")
            return False

        # Wait for result
        result_future = goal_handle.get_result_async()
        if not self._wait_future(result_future, 60.0):
            self.get_logger().error(f"Execution timed out: {label}")
            return False

        result = result_future.result()
        error_code = result.result.error_code.val
        if error_code == 1:  # MoveItErrorCodes.SUCCESS
            self.get_logger().info(f"Goal succeeded: {label}")
            return True
        else:
            self.get_logger().warn(f"Goal failed (error_code={error_code}): {label}")
            return False

    def move_to_pose(self, x, y, z, qx, qy, qz, qw, label="") -> bool:
        goal = self._build_pose_goal(x, y, z, qx, qy, qz, qw)
        return self.send_goal(goal, label)

    def move_to_joints(self, joint_values, label="") -> bool:
        goal = self._build_joint_goal(joint_values)
        return self.send_goal(goal, label)

    def go_home(self) -> bool:
        return self.move_to_joints(_HOME_JOINTS, "home")


def _make_pose(x, y, z, qx, qy, qz, qw):
    from geometry_msgs.msg import Pose
    p = Pose()
    p.position.x = float(x)
    p.position.y = float(y)
    p.position.z = float(z)
    p.orientation.x = float(qx)
    p.orientation.y = float(qy)
    p.orientation.z = float(qz)
    p.orientation.w = float(qw)
    return p


# ── Main pick & place loop ────────────────────────────────────────────────────

def pick_place_loop(node: PickPlaceNode) -> None:
    """
    Blocking loop: wait for grasp pose → execute pick & place → repeat.
    Runs in the main thread; ROS callbacks run in a separate spin thread.
    """
    node.publish("IDLE — waiting for first grasp pose")

    while rclpy.ok():
        grasp_ps = node.next_pose(timeout=60.0)
        if grasp_ps is None:
            node.get_logger().info("No grasp pose in 60 s — still waiting")
            continue

        node._total += 1
        t_start = time.time()
        node.publish(f"BUSY — attempt #{node._total}")
        ok = False

        try:
            # ── Extract grasp info ─────────────────────────────────────────
            gx_gz = grasp_ps.pose.position.x
            gy_gz = grasp_ps.pose.position.y
            gz_gz = grasp_ps.pose.position.z

            qz_raw = grasp_ps.pose.orientation.z
            qw_raw = grasp_ps.pose.orientation.w
            yaw = 2.0 * math.atan2(qz_raw, qw_raw)

            # Convert to MoveIt 2 planning frame
            gx, gy, gz = node.gz_to_moveit(gx_gz, gy_gz, gz_gz)
            node.get_logger().info(
                f"Grasp: Gazebo({gx_gz:.3f}, {gy_gz:.3f}, {gz_gz:.3f}) "
                f"→ MoveIt2({gx:.3f}, {gy:.3f}, {gz:.3f})  yaw={yaw:.3f} rad"
            )

            qx, qy, qz_q, qw = _down_yaw_quat(yaw, node._qd)

            # Place position — 2×3 grid in tray (must match gz_attach_node).
            # Up to 6 cells, then wraps. Coordinates in Gazebo world frame.
            TRAY_GRID = [
                (0.58, -0.30),
                (0.58, -0.18),
                (0.58, -0.06),
                (0.70, -0.24),
            ]
            idx = node._success % len(TRAY_GRID)
            cell_x, cell_y = TRAY_GRID[idx]
            place_gz = (cell_x, cell_y, node._place_gz[2])
            px, py, pz = node.gz_to_moveit(*place_gz)
            pqx, pqy, pqz, pqw = _down_yaw_quat(0.0, node._qd)

            # ── Step 1: Pre-grasp ────────────────────────────────────────
            node.publish("1/8  moving to pre-grasp")
            if not node.move_to_pose(gx, gy, gz + node._pre_h,
                                     qx, qy, qz_q, qw, "pre-grasp"):
                raise RuntimeError("pre-grasp motion failed")

            # ── Step 2: Approach ─────────────────────────────────────────
            node.publish("2/8  approach — moving to grasp pose")
            if not node.move_to_pose(gx, gy, gz,
                                     qx, qy, qz_q, qw, "approach"):
                raise RuntimeError("approach motion failed")

            # ── Step 3: Attach ───────────────────────────────────────────
            node.publish("3/8  attaching object")
            if not node.call_service(True):
                raise RuntimeError("attach failed — no box at grasp pose")
            time.sleep(0.3)

            # ── Step 4: Lift ─────────────────────────────────────────────
            node.publish("4/8  lifting")
            if not node.move_to_pose(gx, gy, gz + node._pre_h,
                                     qx, qy, qz_q, qw, "lift"):
                raise RuntimeError("lift motion failed")

            # ── Step 5: Transport ────────────────────────────────────────
            node.publish("5/8  transporting to tray")
            if not node.move_to_pose(px, py, pz + node._pre_h,
                                     pqx, pqy, pqz, pqw, "transport"):
                raise RuntimeError("transport motion failed")

            # ── Step 6: Place ────────────────────────────────────────────
            node.publish("6/8  placing")
            if not node.move_to_pose(px, py, pz,
                                     pqx, pqy, pqz, pqw, "place"):
                raise RuntimeError("place motion failed")

            # ── Step 7: Detach ───────────────────────────────────────────
            node.publish("7/8  detaching object")
            node.call_service(False)
            time.sleep(1.0)  # let box fall into tray

            # ── Step 8: Retreat ──────────────────────────────────────────
            node.publish("8/8  retreating from tray")
            node.move_to_pose(px, py, pz + node._pre_h,
                              pqx, pqy, pqz, pqw, "retreat")

            ok = True

        except RuntimeError as err:
            node.get_logger().error(f"Pick & place aborted: {err}")

        finally:
            # Always release the object before going home so the box is not
            # dragged to a random position if the cycle failed mid-way.
            node.call_service(False)
            node.publish("returning home")
            node.go_home()

        # ── Metrics ────────────────────────────────────────────────────────
        if ok:
            node._success += 1
            node._consec_fail = 0
        else:
            node._consec_fail += 1

        elapsed = time.time() - t_start
        rate    = node._success / node._total * 100.0
        node.publish(
            f"{'SUCCESS' if ok else 'FAILURE'} | "
            f"{node._success}/{node._total} = {rate:.0f}%  "
            f"cycle={elapsed:.1f}s"
        )

        # After a successful place, drain the queue so we wait for a *fresh*
        # detection. Otherwise YOLO's in-flight pose (still showing the box we
        # just removed) leaks in and triggers a phantom "carry the air" cycle.
        if ok:
            try:
                while True:
                    node._q.get_nowait()
            except queue.Empty:
                pass
            time.sleep(1.5)  # let YOLO refresh its detections

        # Stop after too many consecutive failures
        if node._consec_fail >= node._max_consec_fail:
            node.publish(
                f"STOPPED — {node._consec_fail} consecutive failures. "
                f"Final: {node._success}/{node._total} = {rate:.0f}%"
            )
            break

        time.sleep(1.0)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node = PickPlaceNode()

    # Spin ROS 2 callbacks in a background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Wait for move_group to initialise
    node.get_logger().info("Waiting 5 s for move_group to be ready …")
    time.sleep(5.0)

    node.get_logger().info("Waiting for MoveGroup action server …")
    if not node._move_client.wait_for_server(timeout_sec=30.0):
        node.get_logger().error("MoveGroup action server not available after 30 s")
        node.destroy_node()
        rclpy.shutdown()
        return
    node.get_logger().info("MoveGroup action server connected")

    # Add table as collision object so MoveIt won't plan through it
    time.sleep(1.0)  # let planning scene subscriber connect
    node._publish_table_collision()

    # Ensure gz_attach_node is in clean (detached) state from any previous run
    node.get_logger().info("Releasing any stale attachment from previous run …")
    node.call_service(False)

    # Flush stale grasp poses that accumulated during startup
    flushed = 0
    try:
        while True:
            node._q.get_nowait()
            flushed += 1
    except queue.Empty:
        pass
    node.get_logger().info(f"Flushed {flushed} stale grasp pose(s) — ready")

    try:
        pick_place_loop(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
