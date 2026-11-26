#!/bin/bash
# Stop existing container
docker ps --filter ancestor=gazebo-harmonic -q | xargs -r docker stop 2>/dev/null

WORLD_FILE="${1:-empty.sdf}"

# If a local file is given, validate and use it from the mounted volume
if [ -f "gazebo_projects/$WORLD_FILE" ]; then
  if grep -q "\\\\'" "gazebo_projects/$WORLD_FILE"; then
    echo "ОШИБКА: файл gazebo_projects/$WORLD_FILE повреждён (экранированные кавычки). Пересохраните его."
    exit 1
  fi
  WORLD_FILE="/root/gazebo_projects/$WORLD_FILE"
fi

docker run --rm -d \
  -p 8080:8080 \
  -v "$(pwd)/gazebo_projects:/root/gazebo_projects" \
  -e GZ_WORLD="$WORLD_FILE" \
  gazebo-harmonic

echo "Gazebo запущен с миром: $WORLD_FILE"
echo "Откройте: http://localhost:8080/vnc.html?resize=remote&autoconnect=true"
