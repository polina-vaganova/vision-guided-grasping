"""
sim.launch.py — Main launch file for UR5 pick & place simulation.

Launch sequence:
  1. Generate SDF from ur5_gz.urdf.xacro (pre-converts URDF→SDF to avoid
     Gazebo Harmonic's URDF-to-SDF naming bug with 'world' parent joints)
  2. Gazebo Harmonic  with pick_and_place_env.sdf (table, boxes, camera, tray)
  3. robot_state_publisher  with ur5_gz.urdf.xacro (gz_ros2_control hardware)
  4. Spawn UR5 into Gazebo  at (0, -0.58, 0.780) — top of robot_mount
     Uses pre-converted SDF file (-file) instead of URDF topic (-topic)
  5. ros2_control controllers: joint_state_broadcaster + joint_trajectory_controller
  6. ros_gz_bridge  for /clock and /camera/image_raw
  7. camera_info_pub  — publishes /camera/camera_info synced with image timestamps
  8. static_transform_publisher  — world → camera_link TF
  9. yolo_detector    — YOLOv8-OBB inference on /camera/image_raw
                        → /detections/boxes (BoxDetectionArray)
                        → /detections/image (annotated frame)
 10. pose_estimator   — 2D bbox → 3D grasp pose (pinhole back-projection)
                        → /grasp_poses (PoseArray)   — all detections
                        → /grasp_pose  (PoseStamped) — best detection
 11. MoveIt 2 move_group + RViz  (delayed 15 s to let controllers activate)
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

# Resolved at import time — safe because the package is already installed
_bringup_share = get_package_share_directory("pick_place_bringup")
_xacro_path    = os.path.join(_bringup_share, "urdf", "ur5_gz.urdf.xacro")
_controllers   = os.path.join(_bringup_share, "config", "ur5_controllers.yaml")
_SPAWN_SDF     = "/tmp/ur5_spawn.sdf"


def generate_launch_description():
    # ── args ──────────────────────────────────────────────────────────────
    ur_type        = LaunchConfiguration("ur_type",        default="ur5")
    launch_rviz    = LaunchConfiguration("launch_rviz",    default="true")
    run_pick_place = LaunchConfiguration("run_pick_place", default="true")
    world_file     = LaunchConfiguration("world_file",
                                         default="/worlds/pick_and_place_env.sdf")

    # ── URDF for robot_state_publisher ────────────────────────────────────
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([
            FindPackageShare("pick_place_bringup"),
            "urdf", "ur5_gz.urdf.xacro",
        ]),
        " ur_type:=", ur_type,
        " simulation_controllers:=",
        PathJoinSubstitution([
            FindPackageShare("pick_place_bringup"),
            "config", "ur5_controllers.yaml",
        ]),
    ])
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    # ── 0. Pre-generate SDF ───────────────────────────────────────────────
    # ros_gz_sim 'create -topic' triggers a Gazebo Harmonic URDF→SDF conversion
    # bug that creates duplicate SDF entities for the world-parent fixed joint,
    # causing the spawn to fail.  Workaround: convert to SDF ourselves first
    # using 'gz sdf -p', then spawn with '-file'.
    generate_sdf = ExecuteProcess(
        cmd=[
            "bash", "-c",
            f"source /opt/ros/jazzy/setup.bash && "
            f"source /ros2_ws/install/setup.bash && "
            f"xacro '{_xacro_path}' ur_type:=ur5 "
            f"simulation_controllers:='{_controllers}' "
            f"> /tmp/ur5_spawn.urdf && "
            f"gz sdf -p /tmp/ur5_spawn.urdf > '{_SPAWN_SDF}' && "
            f"echo '[sim.launch] SDF written to {_SPAWN_SDF}'"
        ],
        output="screen",
    )

    # ── 1. Gazebo (server only, no GUI) ───────────────────────────────────
    gz_sim = ExecuteProcess(
        cmd=["gz", "sim", "-s", "-r", world_file],
        additional_env={
            "GZ_SIM_RESOURCE_PATH": "/worlds/models:/opt/ros/jazzy/share",
        },
        output="screen",
    )

    # ── 2. robot_state_publisher ───────────────────────────────────────────
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": True}],
    )

    # ── 3. Spawn UR5 in Gazebo from pre-converted SDF ────────────────────
    # Using -file avoids the URDF→SDF conversion bug in ros_gz_sim.
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name",  "ur",
            "-file",  _SPAWN_SDF,
            "-x", "0.0", "-y", "-0.58", "-z", "0.780",
            "-R", "0.0", "-P", "0.0",  "-Y", "0.0",
        ],
        output="screen",
    )

    # ── 4. ros2_control controllers ────────────────────────────────────────
    load_jsb = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "/controller_manager"],
        output="screen",
    )
    load_jtc = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "-c", "/controller_manager"],
        output="screen",
    )
    load_slider = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["base_slider_controller", "-c", "/controller_manager"],
        output="screen",
    )

    jsb_after_spawn = RegisterEventHandler(
        OnProcessExit(target_action=spawn_robot, on_exit=[load_jsb])
    )
    jtc_after_jsb = RegisterEventHandler(
        OnProcessExit(target_action=load_jsb, on_exit=[load_jtc])
    )
    slider_after_jtc = RegisterEventHandler(
        OnProcessExit(target_action=load_jtc, on_exit=[load_slider])
    )

    # ── 5. ros_gz_bridge ──────────────────────────────────────────────────
    # camera_info is NOT bridged here — the GZ topic name is the full scoped
    # sensor path (/world/.../sensor/camera/camera_info), not /camera/camera_info.
    # Instead, camera_info_pub.py reads calibration YAML and re-publishes it
    # synced with each incoming image frame.
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/camera_left/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera_right/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    # ── 5b. Camera TF: world → camera_link ────────────────────────────────
    # Stereo setup — we treat camera_link as the LEFT stereo camera
    # (where yolo_detector and pose_estimator operate). SDF pose:
    # left: (-0.20, 0, 2.20), pitch = π/2 (looking down).
    camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_link_tf",
        arguments=["--x", "-0.40", "--y", "0", "--z", "2.50",
                   "--yaw", "1.5708", "--pitch", "1.5708", "--roll", "0",
                   "--frame-id", "world", "--child-frame-id", "camera_link"],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    # ── 5c. Camera info publisher ─────────────────────────────────────────
    camera_info_pub = Node(
        package="pick_place_bringup",
        executable="camera_info_pub",
        name="camera_info_pub",
        parameters=[{"use_sim_time": True, "frame_id": "camera_link"}],
        output="screen",
    )

    # ── 6. YOLO detector ──────────────────────────────────────────────────
    # Subscribes /camera/image_raw, publishes /detections/boxes and
    # /detections/image.  Started after bridge so the image topic exists.
    yolo_params = os.path.join(_bringup_share, "config", "yolo_params.yaml")
    yolo_detector = Node(
        package="pick_place_bringup",
        executable="yolo_detector",
        name="yolo_detector",
        parameters=[yolo_params, {"use_sim_time": True}],
        output="screen",
    )

    # ── 6b. Pose estimator ────────────────────────────────────────────────
    # Subscribes /detections/boxes + /camera/camera_info.
    # Publishes /grasp_poses (PoseArray) and /grasp_pose (PoseStamped)
    # in world frame — ready for MoveIt 2 pick & place.
    pose_est_params = os.path.join(_bringup_share, "config", "pose_estimator_params.yaml")
    pose_estimator = Node(
        package="pick_place_bringup",
        executable="pose_estimator",
        name="pose_estimator",
        parameters=[pose_est_params, {"use_sim_time": True}],
        output="screen",
    )

    # ── 6c. Pick & place nodes (delayed — MoveIt must be ready) ──────────
    pp_params = os.path.join(_bringup_share, "config", "pick_place_params.yaml")

    pick_place_node = Node(
        package="pick_place_bringup",
        executable="pick_place_node",
        name="pick_place_node",
        parameters=[pp_params, {"use_sim_time": True}],
        condition=IfCondition(run_pick_place),
        output="screen",
    )

    gz_attach_node = Node(
        package="pick_place_bringup",
        executable="gz_attach_node",
        name="gz_attach_node",
        parameters=[pp_params, {"use_sim_time": True}],
        condition=IfCondition(run_pick_place),
        output="screen",
    )

    # ── 7. MoveIt 2 (delayed — controllers must be active first) ──────────
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("pick_place_bringup"),
                "launch", "moveit.launch.py",
            ])
        ]),
        launch_arguments={
            "ur_type":       ur_type,
            "use_sim_time":  "true",
            "launch_rviz":   launch_rviz,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument("ur_type",         default_value="ur5"),
        DeclareLaunchArgument("launch_rviz",     default_value="true"),
        DeclareLaunchArgument("run_pick_place",  default_value="true"),
        DeclareLaunchArgument("world_file",
                              default_value="/worlds/pick_and_place_env.sdf"),

        # Generate SDF immediately at launch start
        generate_sdf,
        gz_sim,
        robot_state_publisher,
        bridge,
        camera_tf,
        camera_info_pub,
        yolo_detector,
        pose_estimator,
        # Give both SDF generation and Gazebo init time before spawning.
        # SDF gen typically takes < 5 s; Gazebo init ~3 s.
        TimerAction(period=10.0, actions=[spawn_robot]),
        jsb_after_spawn,
        jtc_after_jsb,
        slider_after_jtc,
        # Give controllers time to activate before MoveIt
        TimerAction(period=20.0, actions=[moveit_launch]),
        # gz_attach_node randomizes box positions at startup (3s delay),
        # then handles attach/detach via teleportation during pick & place
        TimerAction(period=16.0, actions=[gz_attach_node]),
        TimerAction(period=30.0, actions=[pick_place_node]),
    ])
