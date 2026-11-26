#!/bin/bash
# Save current Gazebo world to /root/gazebo_projects/
FILENAME="${1:-my_world}.sdf"
gz service -s /world/empty/generate_world_sdf \
  --reqtype gz.msgs.SdfGeneratorConfig \
  --reptype gz.msgs.StringMsg \
  --timeout 5000 --req '' \
  | sed -n "s/^data: \"//;s/\"$//;s/\\\\n/\n/g;s/\\\\'/'/g;p" \
  > "/root/gazebo_projects/$FILENAME"
echo "Saved to /root/gazebo_projects/$FILENAME"
