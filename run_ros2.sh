#!/bin/bash
# Full clean restart: remove container, recreate, fix deps, launch sim + GUI.
# Access: http://localhost:8082/vnc.html
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "[1/5] Removing old container…"
docker rm -f ros2-gazebo 2>/dev/null || true

echo "[2/5] Starting fresh container…"
docker run -d --name ros2-gazebo \
  -p 8082:8080 \
  -v "$ROOT/worlds:/worlds" \
  -v "$ROOT/ros2_ws:/ros2_ws" \
  -e GZ_SIM_RESOURCE_PATH=/worlds/models \
  gazebo-ros2-moveit > /dev/null

sleep 2

echo "[3/5] Patching deps + UR joint limits…"
docker exec ros2-gazebo bash -c "
  pip install 'numpy>=2' 'opencv-python>=4.10' --break-system-packages 2>&1 | tail -1
  sed -i 's/max_acceleration: 5.0/max_acceleration: 15.0/g' \
    /opt/ros/jazzy/share/ur_moveit_config/config/joint_limits.yaml
"

echo "[4/5] Launching ROS 2 + Gazebo + MoveIt 2…"
docker exec -d ros2-gazebo bash -c '
  source /opt/ros/jazzy/setup.bash &&
  source /ros2_ws/install/setup.bash &&
  ros2 launch pick_place_bringup sim.launch.py launch_rviz:=false > /tmp/sim.log 2>&1
'
sleep 4

echo "[5/5] Starting Gazebo GUI client…"
docker exec -d ros2-gazebo bash -c 'export DISPLAY=:1 && gz sim -g'

echo ""
echo "Container started. Wait ~25s for full load."
echo "noVNC: http://localhost:8082/vnc.html"
echo "Logs:  docker exec ros2-gazebo tail -f /tmp/sim.log"
