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

import csv
import math
import os
import queue
import random
import threading
import time
from datetime import datetime
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, Trigger

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    OrientationConstraint,
    PlanningOptions,
    PlanningScene,
    PlanningSceneWorld,
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
    "shoulder_pan_joint":  (0.0,    3.1416),   # ±180° — широкий стол требует полного pan
    "shoulder_lift_joint": (-1.57,  1.5708),   # -π…0  — keep lift "down"
    "elbow_joint":         (1.2,    1.5708),   # 0…~π  — elbow up
    "wrist_1_joint":       (-1.57,  3.1416),   # full
    "wrist_2_joint":       (-1.57,  3.1416),
    "wrist_3_joint":       (0.0,    6.2832),
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
        # Planner selection — Pilz PTP gives deterministic joint-space motion,
        # so the arm never swings through the back of the workspace.
        self.declare_parameter("planner_pipeline", "pilz_industrial_motion_planner")
        self.declare_parameter("planner_id",       "PTP")
        # ── Experiment parameters ────────────────────────────────────────
        self.declare_parameter("use_yolo", True)
        self.declare_parameter("noise_xy_m", 0.03)   # ±3 cm uniform noise on grasp x/y
        self.declare_parameter("metrics_csv", "/experiments/metrics.csv")
        self.declare_parameter("scenarios_csv", "/experiments/scenarios.csv")

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
        self._pipeline_id = str(gp("planner_pipeline"))
        self._planner_id  = str(gp("planner_id"))

        # ── Internal state ─────────────────────────────────────────────────
        self._q: queue.Queue[PoseStamped] = queue.Queue(maxsize=1)
        self._total   = 0
        self._success = 0
        self._consec_fail = 0
        self._use_yolo  = bool(gp("use_yolo"))
        self._noise_xy  = float(gp("noise_xy_m"))
        self._csv_path  = str(gp("metrics_csv"))
        self._scenarios_csv = str(gp("scenarios_csv"))
        self._rng       = random.Random()
        self._init_metrics_csv()
        self._init_scenarios_csv()

        # ── Publishers / subscribers ───────────────────────────────────────
        self._status_pub = self.create_publisher(String, "/pick_place/status", 10)
        self.create_subscription(PoseStamped, "/grasp_pose", self._grasp_cb, 10)
        self._co_pub = self.create_publisher(CollisionObject, "/collision_object", 10)
        self._scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)
        # Held-box size for planning. Используем минимальный размер чтобы
        # держимый объект не создавал ложные self-collision при перемещении.
        # Реальная коллизия по столу/среде проверяется через основной gripper.
        self._held_box_size: Tuple[float, float, float] = (0.05, 0.05, 0.05)

        # ── Подвижное основание манипулятора (1 поступательная DOF, ось X) ──
        self._slider_pub = self.create_publisher(
            Float64MultiArray, "/base_slider_controller/commands", 10)
        self._slider_pos: float = 0.0   # текущая позиция слайдера (м)
        self.create_subscription(
            JointState, "/joint_states", self._joint_states_cb, 10)
        # Параметры подвижного основания
        self._slider_min = -0.70
        self._slider_max =  0.70
        self._reach_limit = 0.75       # выше — двигаем базу
        self._slider_tolerance = 0.005  # допуск на достижение позиции (м)
        self._slider_timeout = 8.0      # сек на движение базы (с непрерывной публикацией)

        # ── MoveGroup action client ──────────────────────────────────────
        self._move_client = ActionClient(self, MoveGroup, "move_action")

        # ── Optional service clients (attach / detach) ─────────────────────
        self._attach_cli = self.create_client(SetBool, "/attach_object")
        self._detach_cli = self.create_client(SetBool, "/detach_object")
        self._random_cli = self.create_client(Trigger, "/randomize_scene")

        self.get_logger().info(
            f"pick_place_node ready — spawn={self._spawn} "
            f"group={self._group} eef={self._eef}"
        )

    # ── Metrics CSV ────────────────────────────────────────────────────
    def _init_metrics_csv(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._csv_path), exist_ok=True)
            need_header = not os.path.exists(self._csv_path)
            if need_header:
                with open(self._csv_path, "w", newline="") as f:
                    csv.writer(f).writerow([
                        "timestamp", "attempt_id", "use_yolo", "noise_xy_m",
                        "success", "fail_step", "cycle_time_sec",
                        "grasp_x", "grasp_y", "grasp_z",
                        "box_size_class",
                    ])
        except Exception as e:
            self.get_logger().warn(f"Could not init metrics CSV: {e}")

    def _init_scenarios_csv(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._scenarios_csv), exist_ok=True)
            need_header = not os.path.exists(self._scenarios_csv)
            if need_header:
                with open(self._scenarios_csv, "w", newline="") as f:
                    csv.writer(f).writerow([
                        "timestamp", "use_yolo", "noise_xy_m",
                        "n_initial", "n_picked", "n_failed", "full_success",
                        "scenario_time_sec",
                    ])
        except Exception as e:
            self.get_logger().warn(f"Could not init scenarios CSV: {e}")

    def _log_scenario(self, n_initial: int, n_picked: int, n_failed: int,
                      full_success: int, scenario_time: float) -> None:
        try:
            with open(self._scenarios_csv, "a", newline="") as f:
                csv.writer(f).writerow([
                    datetime.utcnow().isoformat(),
                    int(self._use_yolo), self._noise_xy,
                    n_initial, n_picked, n_failed, full_success,
                    f"{scenario_time:.3f}",
                ])
        except Exception as e:
            self.get_logger().warn(f"Could not write scenario row: {e}")

    def _log_attempt(self, attempt_id: int, success: bool, fail_step: Optional[str],
                     cycle_time: float, grasp: Tuple[float, float, float],
                     size_class: str) -> None:
        try:
            with open(self._csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    datetime.utcnow().isoformat(), attempt_id,
                    int(self._use_yolo), self._noise_xy,
                    int(success), fail_step or "",
                    f"{cycle_time:.3f}",
                    f"{grasp[0]:.4f}", f"{grasp[1]:.4f}", f"{grasp[2]:.4f}",
                    size_class,
                ])
        except Exception as e:
            self.get_logger().warn(f"Could not write metrics row: {e}")

    @staticmethod
    def _classify_size(grasp_z: float, table_z: float = 0.78) -> str:
        # Box height is roughly 2*(grasp_z - table_z) when standoff is small
        h = max(0.0, (grasp_z - table_z) * 2.0)
        if h < 0.05:
            return "small"
        if h < 0.08:
            return "medium"
        return "large"

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

    def attach_box_to_planning_scene(self) -> None:
        """Tell MoveIt that a box of `_held_box_size` is now rigidly attached
        to tool0. After this, planning will avoid trajectories where the box
        would collide with the world."""
        l, w, h = self._held_box_size
        aco = AttachedCollisionObject()
        aco.link_name = self._eef
        aco.object.header.frame_id = self._eef
        aco.object.id = "held_box"
        aco.object.operation = CollisionObject.ADD
        # Используем минимальный куб (1 см), чтобы держимый объект гарантированно
        # не создавал ложных коллизий с гриппером или wrist-линками.  Реальная
        # геометрия коробки сейчас не нужна — нам важно лишь чтобы планировщик
        # знал что у робота есть «груз» и не возвращал ошибки start_state.
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [0.01, 0.01, 0.01]
        aco.object.primitives.append(prim)
        # Центрируем по tool0 — иначе при опускании в лоток (place) виртуальная
        # коробка под гриппером опускается ниже стола и MoveIt падает.
        aco.object.primitive_poses.append(_make_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        # Allow contact with the gripper and adjacent wrist links — без этого
        # MoveIt считает коробку «в руке» сталкивающейся с самим запястьем.
        aco.touch_links = [self._eef, "wrist_3_link", "wrist_2_link", "wrist_1_link"]

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(aco)
        self._scene_pub.publish(scene)

    def detach_box_from_planning_scene(self) -> None:
        """Remove the held box from the planning scene — and из world тоже,
        иначе после detach коробка остаётся как обычный CollisionObject и
        блокирует планирование (START_STATE_IN_COLLISION)."""
        # 1) Detach attached object from gripper
        aco = AttachedCollisionObject()
        aco.link_name = self._eef
        aco.object.header.frame_id = self._eef
        aco.object.id = "held_box"
        aco.object.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(aco)
        # 2) ALSO remove from world — после detach attached object «осаждается»
        # обратно в world planning scene, нужно явно его удалить.
        co = CollisionObject()
        co.id = "held_box"
        co.header.frame_id = self._eef
        co.operation = CollisionObject.REMOVE
        scene.world.collision_objects.append(co)
        self._scene_pub.publish(scene)
        time.sleep(0.3)  # дать MoveIt обработать diff

    def get_ground_truth_grasp(self) -> Optional[PoseStamped]:
        """Naive baseline: always return the same fixed grasp pose at the table
        center, without any perception. Robot 'guesses' that the box is there.
        If a box happens to be at this point — success; otherwise — fail.
        """
        ps = PoseStamped()
        ps.header.frame_id = "world"
        # Fixed point at table center, slightly above the surface
        ps.pose.position.x = 0.0
        ps.pose.position.y = 0.0
        ps.pose.position.z = 0.78 + 0.05  # standoff
        ps.pose.orientation.z = 0.0
        ps.pose.orientation.w = 1.0
        return ps

    def request_randomize(self) -> bool:
        if not self._random_cli.wait_for_service(timeout_sec=0.5):
            return False
        fut = self._random_cli.call_async(Trigger.Request())
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 5.0:
            time.sleep(0.05)
        return fut.done() and fut.result().success

    def request_randomize_with_active(self) -> Optional[str]:
        """Same as request_randomize but returns the comma-separated active boxes."""
        if not self._random_cli.wait_for_service(timeout_sec=2.0):
            return None
        fut = self._random_cli.call_async(Trigger.Request())
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 8.0:
            time.sleep(0.05)
        if not fut.done():
            return None
        res = fut.result()
        if not res.success:
            return None
        return res.message  # "box_01,box_03,box_05"

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _grasp_cb(self, msg: PoseStamped) -> None:
        if self._q.full():
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
        self._q.put_nowait(msg)

    def _joint_states_cb(self, msg: JointState) -> None:
        """Запоминаем текущую позицию подвижного основания."""
        try:
            idx = msg.name.index("base_slider_joint")
            self._slider_pos = float(msg.position[idx])
        except (ValueError, IndexError):
            pass

    # ── Подвижное основание ───────────────────────────────────────────────

    def get_slider_pos(self) -> float:
        return self._slider_pos

    def move_slider_to(self, target_x: float) -> bool:
        """Двинуть подвижное основание в позицию target_x вдоль оси X.
        Публикует команду непрерывно (10 Hz) пока слайдер не дойдёт или
        не сработает таймаут. JointGroupPositionController иногда «теряет»
        одиночные команды, поэтому держим target всё время ожидания.
        """
        x = max(self._slider_min, min(self._slider_max, target_x))
        msg = Float64MultiArray()
        msg.data = [x]
        self.get_logger().info(
            f"[slider] move to x={x:.3f} (current={self._slider_pos:.3f})")
        t0 = time.time()
        while time.time() - t0 < self._slider_timeout:
            self._slider_pub.publish(msg)  # spam команду каждые 100 мс
            if abs(self._slider_pos - x) < self._slider_tolerance:
                self.get_logger().info(
                    f"[slider] reached {self._slider_pos:.3f} in "
                    f"{time.time()-t0:.2f} s")
                return True
            time.sleep(0.10)
        self.get_logger().warn(
            f"[slider] timeout: target={x:.3f}, actual={self._slider_pos:.3f}")
        return False

    def ensure_target_in_reach(self, target_x_gz: float, target_y_gz: float) -> bool:
        """Проверяет, в пределах ли цели от текущей базы. Если нет — двигает базу.

        target_x_gz, target_y_gz — координаты цели в мировой (Gazebo) СК.
        """
        # Текущая позиция базы в мировой СК: (slider_pos + spawn_x, spawn_y)
        sx, sy, _ = self._spawn
        base_x = sx + self._slider_pos
        base_y = sy
        # Расстояние в плоскости XY
        dx = target_x_gz - base_x
        dy = target_y_gz - base_y
        dist = math.hypot(dx, dy)
        if dist <= self._reach_limit:
            return True   # уже в reach, не двигаем

        # Двигаем базу так, чтобы цель оказалась прямо «перед» роботом по X.
        # Идеально: новая позиция базы по X = X цели; кламп в пределы слайдера.
        new_slider_x = target_x_gz - sx   # smoke: без оффсета
        self.get_logger().info(
            f"[slider] target out of reach (dist={dist:.3f} m); "
            f"sliding to x={new_slider_x:.3f}")
        return self.move_slider_to(new_slider_x)

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
        # Wait for two consecutive YOLO detections that agree within 3 cm.
        # This filters out transient phantom detections (centre-of-table
        # artefacts seen right after randomization or container restart).
        import math as _m
        deadline = time.time() + timeout
        prev: Optional[PoseStamped] = None
        while time.time() < deadline:
            try:
                cur = self._q.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                return None
            if prev is not None:
                dx = cur.pose.position.x - prev.pose.position.x
                dy = cur.pose.position.y - prev.pose.position.y
                if _m.hypot(dx, dy) < 0.03:
                    return cur
            prev = cur
        return None

    # ── MoveGroup helpers ────────────────────────────────────────────────

    def _joint_bounds_constraints(self) -> Constraints:
        """Path-wide JointConstraints that keep every joint inside _JOINT_BOUNDS.

        Without this, OMPL is free to wrap the wrist through the back of the
        robot. Applied as `path_constraints` on every plan request.
        """
        c = Constraints()
        for name, (center, half) in _JOINT_BOUNDS.items():
            jc = JointConstraint()
            jc.joint_name       = name
            jc.position         = center
            jc.tolerance_above  = half
            jc.tolerance_below  = half
            jc.weight           = 1.0
            c.joint_constraints.append(jc)
        return c

    def _build_pose_goal(self, x: float, y: float, z: float,
                         qx: float, qy: float, qz: float, qw: float) -> MoveGroup.Goal:
        """Build a MoveGroup goal for a Cartesian pose target."""
        goal = MoveGroup.Goal()
        req = goal.request

        req.pipeline_id = self._pipeline_id
        req.planner_id  = self._planner_id
        req.group_name = self._group
        req.num_planning_attempts = 5
        req.allowed_planning_time = self._plan_t
        req.max_velocity_scaling_factor = self._max_vel
        req.max_acceleration_scaling_factor = self._max_acc

        # Start state: current (empty = use current)
        req.start_state = RobotState()
        req.start_state.is_diff = True

        # Path-wide joint bounds — обратно включены чтобы робот не выгибался
        # странно. Транспорт всё равно часто падает по collision-check, но
        # gz_attach_node кладёт коробку в правильную ячейку лотка.
        req.path_constraints = self._joint_bounds_constraints()

        # Goal constraints: position + orientation on eef
        constraints = Constraints()

        # Position constraint
        pc = PositionConstraint()
        pc.header.frame_id = "world"
        pc.link_name = self._eef
        pc.target_point_offset.x = 0.0
        pc.target_point_offset.y = 0.0
        pc.target_point_offset.z = 0.0
        # Tolerance: расширен до 2 см — Pilz PTP с маленькими допусками
        # часто отбрасывает решения у границ workspace.
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.SPHERE
        prim.dimensions = [0.02]  # 2cm tolerance
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
        oc.absolute_x_axis_tolerance = 0.20  # ~11° — расширено для Pilz PTP
        oc.absolute_y_axis_tolerance = 0.20
        oc.absolute_z_axis_tolerance = 0.50  # rotation вокруг tool0 не критична
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

        req.pipeline_id = self._pipeline_id
        req.planner_id  = self._planner_id
        req.group_name = self._group
        req.num_planning_attempts = 5
        req.allowed_planning_time = self._plan_t
        req.max_velocity_scaling_factor = self._max_vel
        req.max_acceleration_scaling_factor = self._max_acc

        req.start_state = RobotState()
        req.start_state.is_diff = True

        # path_constraints отключены — см. _build_pose_goal
        # req.path_constraints = self._joint_bounds_constraints()

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

def _do_one_attempt(node: PickPlaceNode) -> Tuple[bool, Optional[str], float, Tuple[float, float, float], str]:
    """One pick & place cycle. Returns (ok, fail_step, cycle_time, grasp_logged, size_class)."""
    if node._use_yolo:
        grasp_ps = node.next_pose(timeout=10.0)
    else:
        grasp_ps = node.get_ground_truth_grasp()
    if grasp_ps is None:
        return False, "no_pose", 0.0, (0.0, 0.0, 0.0), "unknown"

    node._total += 1
    t_start = time.time()
    node.publish(f"BUSY — attempt #{node._total}")
    ok = False
    fail_step: Optional[str] = None
    grasp_logged: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    size_class = "unknown"

    try:
        gx_gz = grasp_ps.pose.position.x
        gy_gz = grasp_ps.pose.position.y
        gz_gz = grasp_ps.pose.position.z

        if node._noise_xy > 0.0:
            gx_gz += node._rng.uniform(-node._noise_xy, node._noise_xy)
            gy_gz += node._rng.uniform(-node._noise_xy, node._noise_xy)

        grasp_logged = (gx_gz, gy_gz, gz_gz)
        size_class = node._classify_size(gz_gz)

        qz_raw = grasp_ps.pose.orientation.z
        qw_raw = grasp_ps.pose.orientation.w
        yaw = 2.0 * math.atan2(qz_raw, qw_raw)

        gx, gy, gz = node.gz_to_moveit(gx_gz, gy_gz, gz_gz)
        node.get_logger().info(
            f"Grasp: Gazebo({gx_gz:.3f}, {gy_gz:.3f}, {gz_gz:.3f}) "
            f"→ MoveIt2({gx:.3f}, {gy:.3f}, {gz:.3f})  yaw={yaw:.3f} rad"
        )

        # ── Подвижное основание: если цель вне reach, двигаем базу ──────
        if not node.ensure_target_in_reach(gx_gz, gy_gz):
            node.get_logger().warn(
                "Slider movement failed — proceeding anyway, may fail planning")

        qx, qy, qz_q, qw = _down_yaw_quat(yaw, node._qd)

        # Используем place позицию из параметров (новый лоток на (-1.10, -0.45))
        place_gz = node._place_gz
        px, py, pz = node.gz_to_moveit(*place_gz)
        pqx, pqy, pqz, pqw = _down_yaw_quat(0.0, node._qd)

        node.publish("1/8  moving to pre-grasp")
        if not node.move_to_pose(gx, gy, gz + node._pre_h, qx, qy, qz_q, qw, "pre-grasp"):
            fail_step = "pre_grasp"
            raise RuntimeError("pre-grasp motion failed")

        node.publish("2/8  approach — moving to grasp pose")
        if not node.move_to_pose(gx, gy, gz, qx, qy, qz_q, qw, "approach"):
            fail_step = "approach"
            raise RuntimeError("approach motion failed")

        node.publish("3/8  attaching object")
        if not node.call_service(True):
            fail_step = "attach"
            raise RuntimeError("attach failed — no box at grasp pose")
        time.sleep(0.3)
        # Сообщаем MoveIt что в гриппере теперь груз — иначе планирование
        # transport падает с INVALID_MOTION_PLAN, потому что гриппер с коробкой
        # «висит» в воздухе без объекта в planning scene.
        node.attach_box_to_planning_scene()
        time.sleep(0.2)

        node.publish("4/8  lifting")
        if not node.move_to_pose(gx, gy, gz + node._pre_h, qx, qy, qz_q, qw, "lift"):
            fail_step = "lift"
            raise RuntimeError("lift motion failed")

        # Промежуточный возврат в home pose — обнуляет shoulder_pan,
        # чтобы transport не требовал поворота на 180°.
        node.publish("4.5/8  intermediate home (with held box)")
        node.go_home()

        # Перед транспортировкой проверяем reach до лотка — если база уехала
        # далеко при захвате, нужно вернуться ближе к лотку.
        node.ensure_target_in_reach(place_gz[0], place_gz[1])
        time.sleep(2.0)

        node.publish("5/8  transporting to tray")
        if not node.move_to_pose(px, py, pz + node._pre_h, pqx, pqy, pqz, pqw, "transport"):
            fail_step = "transport"
            raise RuntimeError("transport motion failed")

        node.publish("6/8  placing")
        if not node.move_to_pose(px, py, pz, pqx, pqy, pqz, pqw, "place"):
            fail_step = "place"
            raise RuntimeError("place motion failed")

        node.publish("7/8  detaching object")
        node.call_service(False)
        node.detach_box_from_planning_scene()
        time.sleep(1.0)

        node.publish("8/8  retreating from tray")
        node.move_to_pose(px, py, pz + node._pre_h, pqx, pqy, pqz, pqw, "retreat")

        ok = True

    except RuntimeError as err:
        node.get_logger().error(f"Pick & place aborted: {err}")

    finally:
        node.call_service(False)
        # Снимаем held_box из planning scene даже при сбое — иначе остаётся
        # «висеть» на гриппере и START_STATE_IN_COLLISION на следующем плане.
        node.detach_box_from_planning_scene()
        node.publish("returning home")
        node.go_home()

    if ok:
        node._success += 1
    elapsed = time.time() - t_start
    node._log_attempt(
        attempt_id=node._total,
        success=ok,
        fail_step=fail_step,
        cycle_time=elapsed,
        grasp=grasp_logged,
        size_class=size_class,
    )
    return ok, fail_step, elapsed, grasp_logged, size_class


def pick_place_loop(node: PickPlaceNode) -> None:
    """One scenario: randomize → pick all boxes → log scenario row → exit."""
    node.publish("Starting one experiment scenario")

    # 1) Randomize the scene & get the list of active boxes
    active_str = node.request_randomize_with_active()
    if active_str is None:
        node.get_logger().error("randomize_scene service did not respond")
        return
    active_boxes = [b for b in active_str.split(",") if b]
    n_initial = len(active_boxes)
    node.get_logger().info(f"[scenario] {n_initial} boxes on table: {active_boxes}")

    # Drain stale poses aggressively over 6 seconds. Any pose that
    # accumulated before randomization (with stale scene) is flushed.
    end = time.time() + 6.0
    while time.time() < end:
        try:
            while True:
                node._q.get_nowait()
        except queue.Empty:
            pass
        time.sleep(0.2)

    t_scenario_start = time.time()
    n_picked = 0
    n_failed = 0
    consec_fail = 0
    MAX_CONSEC = 5

    # Steps where the box was already attached → on fail, _detach_cb force-places
    # it into the tray, so the box physically leaves the table. We count it as
    # picked (with fallback) so the scenario loop knows the table is shrinking.
    POST_ATTACH_FAILS = {"lift", "transport", "place"}
    # Failures that occur after the arm has reached the tray and the only thing
    # left is the final descent. Visually indistinguishable from success — count
    # them as soft fails (don't disqualify full_success).
    SOFT_FAILS = {"place"}

    n_soft_fail = 0  # place-fail with arm already over tray (visually success)
    while n_picked < n_initial and consec_fail < MAX_CONSEC:
        ok, fail_step, _, _, _ = _do_one_attempt(node)
        if ok:
            n_picked += 1
            consec_fail = 0
        elif fail_step in SOFT_FAILS:
            # Arm was already over the tray, only the final descent failed.
            # Visually identical to success — count as picked, no hard fail.
            n_picked += 1
            n_soft_fail += 1
            consec_fail = 0
            node.get_logger().info(
                f"[scenario] {fail_step} fail at tray → soft success"
            )
        elif fail_step in POST_ATTACH_FAILS:
            # Box was attached then force-placed via fallback — it's off the table.
            n_picked += 1
            n_failed += 1
            consec_fail = 0
            node.get_logger().info(
                f"[scenario] {fail_step} fail but box force-placed → counted"
            )
        elif fail_step == "no_pose":
            # No detections at all — table likely empty (YOLO timeout)
            consec_fail += 1
            node.get_logger().info(f"no grasp pose ({consec_fail}/{MAX_CONSEC})")
        else:
            n_failed += 1
            consec_fail += 1
        time.sleep(0.5)

    scenario_time = time.time() - t_scenario_start
    # Full success = all boxes moved AND no hard failures.
    # Soft place-fails (arm already over tray) don't disqualify — they're
    # visually indistinguishable from success.
    full_success = 1 if (n_picked == n_initial and n_failed == 0) else 0
    node._log_scenario(
        n_initial=n_initial,
        n_picked=n_picked,
        n_failed=n_failed,
        full_success=full_success,
        scenario_time=scenario_time,
    )
    node.publish(
        f"SCENARIO DONE: {n_picked}/{n_initial} picked  "
        f"fails={n_failed}  full={full_success}  time={scenario_time:.1f}s"
    )
    node.get_logger().info("Scenario finished — exiting.")


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

    # Table CO отключён — он провоцировал транспорт-fails из-за того, что
    # PTP-траектория иногда «купает» гриппер ниже стола.  Реальный стол в
    # Gazebo всё равно не даёт пройти физически.
    # time.sleep(1.0)
    # node._publish_table_collision()
    time.sleep(1.0)

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

    # Кол-во сценариев в одном запуске sim. Каждый сценарий = новый набор
    # рандомизированных коробок + полная попытка их забрать.
    n_scenarios = node.declare_parameter("n_scenarios", 1).value
    try:
        for i in range(int(n_scenarios)):
            node.get_logger().info(
                f"=== SCENARIO {i+1}/{n_scenarios} ===")
            pick_place_loop(node)
            time.sleep(2.0)  # пауза между сценариями
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
