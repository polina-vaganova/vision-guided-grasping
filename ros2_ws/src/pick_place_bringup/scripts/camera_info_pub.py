#!/usr/bin/env python3
"""
camera_info_pub.py
Publishes sensor_msgs/CameraInfo on /camera/camera_info in sync with
/camera/image_raw.  Reads intrinsics from camera_info.yaml (installed
alongside the package under config/).

Parameters
----------
camera_info_path : str
    Absolute path to camera_info.yaml.
frame_id : str  (default: "camera_link")
    TF frame embedded in the header of every CameraInfo message.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import CameraInfo, Image
from ament_index_python.packages import get_package_share_directory
import yaml
import os


class CameraInfoPublisher(Node):
    def __init__(self):
        super().__init__("camera_info_pub")

        # ── Parameters ────────────────────────────────────────────────────
        default_yaml = os.path.join(
            get_package_share_directory("pick_place_bringup"),
            "config", "camera_info.yaml",
        )
        self.declare_parameter("camera_info_path", default_yaml)
        self.declare_parameter("frame_id", "camera_link")

        yaml_path = self.get_parameter("camera_info_path").get_parameter_value().string_value
        frame_id  = self.get_parameter("frame_id").get_parameter_value().string_value

        # ── Load YAML ─────────────────────────────────────────────────────
        with open(yaml_path, "r") as f:
            cal = yaml.safe_load(f)

        self._msg = CameraInfo()
        self._msg.header.frame_id = frame_id
        self._msg.width  = cal["image_width"]
        self._msg.height = cal["image_height"]
        self._msg.distortion_model = cal["distortion_model"]
        self._msg.d = cal["distortion_coefficients"]["data"]
        self._msg.k = cal["camera_matrix"]["data"]
        self._msg.r = cal["rectification_matrix"]["data"]
        self._msg.p = cal["projection_matrix"]["data"]

        # ── Pub / Sub ─────────────────────────────────────────────────────
        qos = QoSPresetProfiles.SENSOR_DATA.value
        self._pub = self.create_publisher(CameraInfo, "/camera/camera_info", qos)
        self.create_subscription(Image, "/camera_left/image_raw", self._image_cb, qos)

        self.get_logger().info(
            f"camera_info_pub ready — frame_id={frame_id}, yaml={yaml_path}"
        )

    def _image_cb(self, img_msg: Image):
        """Publish CameraInfo with the same timestamp as the incoming image."""
        self._msg.header.stamp = img_msg.header.stamp
        self._pub.publish(self._msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraInfoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
