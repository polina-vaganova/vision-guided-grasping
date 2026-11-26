#!/bin/bash
set -e

# ---- Virtual display ----
Xvnc :1 -geometry 1920x1080 -depth 24 -SecurityTypes None &
sleep 2
websockify --web=/usr/share/novnc 8080 localhost:5901 &

export DISPLAY=:1
export LIBGL_ALWAYS_SOFTWARE=1
export GZ_RENDERING_ENGINE=ogre
export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH:-/worlds/models}"
# Expose ROS 2 plugin libraries to Gazebo (needed for gz_ros2_control)
export GZ_SIM_SYSTEM_PLUGIN_PATH="/opt/ros/jazzy/lib${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"

# ---- Source ROS 2 ----
source /opt/ros/jazzy/setup.bash
[ -f /ros2_ws/install/setup.bash ] && source /ros2_ws/install/setup.bash

echo ""
echo "================================================"
echo "  noVNC:  http://localhost:8080/vnc.html"
echo "  Shell:  docker exec -it ros2-gazebo bash"
echo "================================================"
echo ""
echo "Usage inside container:"
echo "  source /opt/ros/jazzy/setup.bash"
echo "  source /ros2_ws/install/setup.bash"
echo "  ros2 launch pick_place_bringup sim.launch.py"
echo ""

# Keep container alive
tail -f /dev/null
