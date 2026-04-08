#!/bin/bash

# Start TigerVNC server (supports dynamic resizing)
Xvnc :1 -geometry 2560x1600 -depth 24 -SecurityTypes None &
sleep 2

# Start noVNC (browser-based VNC client)
websockify --web=/usr/share/novnc 8080 localhost:5901 &

# Start Gazebo
export DISPLAY=:1
export LIBGL_ALWAYS_SOFTWARE=1
export GZ_RENDERING_ENGINE=ogre
WORLD_FILE="${GZ_WORLD:-empty.sdf}"
gz sim -r "$WORLD_FILE"
