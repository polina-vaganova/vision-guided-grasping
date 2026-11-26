#!/bin/bash
CONTAINER=$(docker ps --filter ancestor=gazebo-harmonic -q | head -1)
if [ -z "$CONTAINER" ]; then
  echo "Gazebo контейнер не запущен!"
  exit 1
fi
NAME="${1:-my_world}"
docker exec "$CONTAINER" save_world "$NAME"
