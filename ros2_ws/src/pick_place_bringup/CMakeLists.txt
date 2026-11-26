cmake_minimum_required(VERSION 3.8)
project(pick_place_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config urdf
  DESTINATION share/${PROJECT_NAME}
)

# Install Python scripts as ROS 2 executables
install(PROGRAMS
  scripts/camera_info_pub.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME camera_info_pub
)

install(PROGRAMS
  scripts/yolo_detector.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME yolo_detector
)

install(PROGRAMS
  scripts/pose_estimator.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME pose_estimator
)

install(PROGRAMS
  scripts/pick_place_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME pick_place_node
)

install(PROGRAMS
  scripts/gz_attach_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME gz_attach_node
)

ament_package()
