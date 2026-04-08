"""
moveit.launch.py — Standalone MoveIt 2 launch for UR5 with gz_ros2_control.

Uses our custom ur5_gz.urdf.xacro (with GazeboSimSystem hardware interface)
and ur_moveit_config for SRDF, kinematics, planning pipeline configs.

Call this AFTER sim.launch.py has started controllers (or via sim.launch.py TimerAction).
"""
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    ur_type     = LaunchConfiguration("ur_type",     default="ur5")
    launch_rviz = LaunchConfiguration("launch_rviz", default="true")
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    # Resolve package paths at Python evaluation time (not launch substitutions)
    bringup_share = get_package_share_directory("pick_place_bringup")
    urdf_xacro    = os.path.join(bringup_share, "urdf", "ur5_gz.urdf.xacro")

    # Build MoveIt config:
    #   URDF  — our custom xacro with gz_ros2_control hardware interface
    #   SRDF  — from ur_moveit_config (arm group, end-effectors, etc.)
    #   rest  — kinematics, planning pipelines from ur_moveit_config
    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur", package_name="ur_moveit_config")
        .robot_description(
            file_path=urdf_xacro,
            mappings={"ur_type": "ur5"},
        )
        .robot_description_semantic(
            Path("srdf") / "ur.srdf.xacro",
            {"name": "ur"},   # must match robot name in URDF (<robot name="ur">)
        )
        .to_moveit_configs()
    )

    # Wrap XML strings as explicit str type to prevent launch YAML parser from
    # choking on XML content (evaluate_parameters.py tries yaml.safe_load on everything)
    moveit_dict = moveit_config.to_dict()
    for key in ("robot_description", "robot_description_semantic"):
        if key in moveit_dict and isinstance(moveit_dict[key], str):
            moveit_dict[key] = ParameterValue(moveit_dict[key], value_type=str)

    # Override MoveIt controller config to use joint_trajectory_controller
    # (scaled_joint_trajectory_controller is not available in Gazebo simulation)
    import yaml
    custom_controllers_path = os.path.join(bringup_share, "config", "moveit_controllers.yaml")
    with open(custom_controllers_path) as f:
        ctrl_params = yaml.safe_load(f)

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_dict,
            ctrl_params,
            {"use_sim_time": use_sim_time},
        ],
    )

    rviz_config = PathJoinSubstitution([
        FindPackageShare("ur_moveit_config"),
        "config", "moveit.rviz",
    ])

    # Build RViz parameter dicts with same XML-as-str wrapping
    rd_dict  = moveit_config.robot_description
    rds_dict = moveit_config.robot_description_semantic
    for d in (rd_dict, rds_dict):
        for key in ("robot_description", "robot_description_semantic"):
            if key in d and isinstance(d[key], str):
                d[key] = ParameterValue(d[key], value_type=str)

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        condition=IfCondition(launch_rviz),
        arguments=["-d", rviz_config],
        parameters=[
            rd_dict,
            rds_dict,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {"use_sim_time": use_sim_time},
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument("ur_type",      default_value="ur5"),
        DeclareLaunchArgument("launch_rviz",  default_value="true"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        move_group,
        rviz,
    ])
