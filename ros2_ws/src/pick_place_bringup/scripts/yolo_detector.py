#!/usr/bin/env python3
"""
yolo_detector.py — YOLOv8-OBB detection node for UR5 pick-and-place.

Subscribes
----------
/camera/image_raw   (sensor_msgs/Image)  — raw RGB frame from overhead camera

Publishes
---------
/detections/boxes   (pick_place_msgs/BoxDetectionArray)  — OBB results
/detections/image   (sensor_msgs/Image)                  — annotated frame

Parameters
----------
model_path   : str   — path to .pt file or ultralytics model name
                       default: 'yolov8n-obb.pt'  (auto-downloaded on first run)
conf         : float — confidence threshold, default 0.25
device       : str   — 'cpu' or 'cuda:0', default 'cpu'
frame_id     : str   — TF frame for detection header, default 'camera_link'

Notes on the OBB angle convention
----------------------------------
ultralytics YOLOv8-OBB stores the rotation angle θ in radians in result.obb.xywhr[:, 4].
The angle is defined as the rotation (counter-clockwise) from the image +x axis to the
box's long axis, in the range (−π/2, π/2].  This matches OpenCV minAreaRect output after
the standard conversion:  angle_rad = -angle_deg * π / 180.
The 3D yaw passed to MoveIt is this same value (rotation around the Z axis of the camera
frame, which points down, so the sign is consistent with world-frame yaw when the camera
is mounted eye-to-hand looking straight down).
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

from sensor_msgs.msg import Image

from pick_place_msgs.msg import BoxDetection, BoxDetectionArray


# ── Lightweight ROS Image ↔ numpy conversion (no cv_bridge needed) ────────
# cv_bridge is compiled against numpy 1.x while ultralytics requires 2.x;
# bypassing it removes the version conflict entirely.

_ENCODING_CHANNELS = {
    "rgb8": (3, np.uint8,  False),
    "bgr8": (3, np.uint8,  True),   # flip to RGB for YOLO
    "rgba8": (4, np.uint8, False),
    "bgra8": (4, np.uint8, True),
    "mono8": (1, np.uint8, False),
}


def _imgmsg_to_rgb(msg: Image) -> np.ndarray:
    """Convert sensor_msgs/Image to an HxWx3 uint8 RGB numpy array."""
    enc = msg.encoding.lower()
    if enc not in _ENCODING_CHANNELS:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")
    channels, dtype, bgr = _ENCODING_CHANNELS[enc]
    arr = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels)
    if channels == 4:
        arr = arr[:, :, :3]  # drop alpha
    if bgr:
        arr = arr[:, :, ::-1].copy()  # BGR → RGB
    if channels == 1:
        arr = np.repeat(arr, 3, axis=2)  # mono → RGB
    return arr


def _rgb_to_imgmsg(arr: np.ndarray, stamp, frame_id: str) -> Image:
    """Convert HxWx3 uint8 RGB numpy array to sensor_msgs/Image."""
    msg = Image()
    msg.header.stamp    = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = "rgb8"
    msg.step = msg.width * 3
    msg.data = arr.tobytes()
    return msg


class YoloDetector(Node):

    def __init__(self):
        super().__init__("yolo_detector")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("model_path", "yolov8n-obb.pt")
        self.declare_parameter("conf",       0.25)
        self.declare_parameter("device",     "cpu")
        self.declare_parameter("frame_id",   "camera_link")

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        conf       = self.get_parameter("conf").get_parameter_value().double_value
        device     = self.get_parameter("device").get_parameter_value().string_value
        self._frame_id = self.get_parameter("frame_id").get_parameter_value().string_value

        # ── Load YOLO model ───────────────────────────────────────────────
        # Import here so the node can start (and report the error) even if
        # ultralytics is not installed yet.
        try:
            from ultralytics import YOLO
            import torch
        except ImportError as exc:
            self.get_logger().fatal(
                f"Cannot import ultralytics / torch: {exc}\n"
                "Install with:  pip install ultralytics"
            )
            raise SystemExit(1) from exc

        self.get_logger().info(f"Loading YOLO model: {model_path}  device={device}")
        self._model = YOLO(model_path)
        self._conf  = conf
        self._device = device

        # Warm-up: compile with torch.compile if PyTorch >= 2.0 for faster inference
        try:
            if hasattr(torch, "compile"):
                self.get_logger().info("torch.compile available — export model for speed-up")
                self._model.model = torch.compile(self._model.model, mode="reduce-overhead")
        except Exception as e:
            self.get_logger().warning(f"torch.compile skipped: {e}")

        # Determine whether the loaded model supports OBB output
        # (True for *-obb.pt models).  For plain detect models we fall back
        # to axis-aligned bboxes and estimate angle via minAreaRect.
        self._is_obb = hasattr(self._model.model, "task") and \
                       getattr(self._model.model, "task", "") == "obb"
        if not self._is_obb:
            # Also check model names attribute
            try:
                self._is_obb = "obb" in model_path.lower()
            except Exception:
                pass
        self.get_logger().info(f"OBB mode: {self._is_obb}")

        # ── Publishers: per-camera detection streams ─────────────────────
        qos_sensor = QoSPresetProfiles.SENSOR_DATA.value
        self._pub_boxes_left = self.create_publisher(
            BoxDetectionArray, "/detections/boxes_left", 10
        )
        self._pub_boxes_right = self.create_publisher(
            BoxDetectionArray, "/detections/boxes_right", 10
        )
        # Legacy topic — keep for backwards compatibility (= left stream).
        self._pub_boxes = self._pub_boxes_left
        self._pub_image = self.create_publisher(
            Image, "/detections/image", qos_sensor
        )

        # ── Subscribers: left + right stereo cameras ─────────────────────
        self.create_subscription(
            Image, "/camera_left/image_raw",
            lambda m: self._image_cb(m, "left"), qos_sensor,
        )
        self.create_subscription(
            Image, "/camera_right/image_raw",
            lambda m: self._image_cb(m, "right"), qos_sensor,
        )

        self.get_logger().info(
            f"yolo_detector ready (stereo)  conf={conf}  model={model_path}"
        )

    # ──────────────────────────────────────────────────────────────────────
    def _image_cb(self, msg: Image, cam: str = "left"):
        """Run YOLO inference on incoming frame and publish results."""
        # Convert ROS Image → RGB numpy array (no cv_bridge — avoids numpy version conflict)
        try:
            frame = _imgmsg_to_rgb(msg)
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        # Convert to grayscale (model trained on B&W images)
        gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        frame_bw = np.stack([gray, gray, gray], axis=-1)

        # Run inference (verbose=False suppresses per-frame stdout)
        results = self._model(
            frame_bw,
            conf=self._conf,
            device=self._device,
            verbose=False,
        )

        # ── Build BoxDetectionArray ────────────────────────────────────────
        det_array = BoxDetectionArray()
        det_array.header.stamp    = msg.header.stamp
        det_array.header.frame_id = self._frame_id

        annotated = results[0].plot()  # HxWx3 RGB annotated frame

        for result in results:
            if self._is_obb and result.obb is not None and len(result.obb) > 0:
                detections = self._parse_obb(result)
            else:
                detections = self._parse_boxes(result, frame)
            det_array.detections.extend(detections)

        pub = self._pub_boxes_left if cam == "left" else self._pub_boxes_right
        pub.publish(det_array)

        # Publish annotated image (left camera only — avoids UI flicker)
        if cam == "left":
            try:
                ann_msg = _rgb_to_imgmsg(annotated, msg.header.stamp, self._frame_id)
                self._pub_image.publish(ann_msg)
            except Exception as e:
                self.get_logger().error(f"annotated image publish error: {e}")

    # ──────────────────────────────────────────────────────────────────────
    def _parse_obb(self, result) -> list:
        """Extract detections from an OBB result (yolov8-obb model)."""
        detections = []
        names = result.names  # {class_id: label_str}

        # result.obb.xywhr: Tensor[N, 5] — cx, cy, w, h, angle_rad
        xywhr  = result.obb.xywhr.cpu().numpy()
        confs  = result.obb.conf.cpu().numpy()
        clsids = result.obb.cls.cpu().numpy().astype(int)

        for (cx, cy, w, h, angle_rad), conf, cls_id in zip(xywhr, confs, clsids):
            d = BoxDetection()
            d.x_center   = float(cx)
            d.y_center   = float(cy)
            # Ensure w >= h so that 'width' is always the longer dimension
            if w < h:
                w, h = h, w
                # Rotate angle by 90° when swapping axes
                angle_rad = angle_rad + math.pi / 2
            d.width      = float(w)
            d.height     = float(h)
            # Normalise angle to (−π/2, π/2]
            d.angle      = float(_normalise_angle(angle_rad))
            d.confidence = float(conf)
            d.label      = names.get(cls_id, str(cls_id))
            detections.append(d)

        return detections

    # ──────────────────────────────────────────────────────────────────────
    def _parse_boxes(self, result, frame) -> list:
        """
        Fallback for regular (non-OBB) YOLO models.
        Uses each detection's bounding box crop + OpenCV minAreaRect to
        estimate the orientation angle of the largest contour.
        """
        import cv2

        detections = []
        names = result.names

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        # frame is RGB; convert to gray via numpy (no cv2.COLOR_BGR2GRAY)
        gray = (0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] +
                0.114 * frame[:, :, 2]).astype(np.uint8)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        boxes_xywh = result.boxes.xywh.cpu().numpy()    # [cx, cy, w, h]
        confs      = result.boxes.conf.cpu().numpy()
        clsids     = result.boxes.cls.cpu().numpy().astype(int)
        h_img, w_img = frame.shape[:2]

        for (cx, cy, bw, bh), conf, cls_id in zip(boxes_xywh, confs, clsids):
            d = BoxDetection()
            d.x_center   = float(cx)
            d.y_center   = float(cy)
            d.width      = float(bw)
            d.height     = float(bh)
            d.confidence = float(conf)
            d.label      = names.get(cls_id, str(cls_id))

            # Estimate angle from contour inside the bounding box crop
            x1 = max(0, int(cx - bw / 2))
            y1 = max(0, int(cy - bh / 2))
            x2 = min(w_img, int(cx + bw / 2))
            y2 = min(h_img, int(cy + bh / 2))
            crop_thresh = thresh[y1:y2, x1:x2]

            angle_rad = 0.0
            contours, _ = cv2.findContours(
                crop_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 10:
                    # minAreaRect returns angle in degrees in OpenCV convention
                    rect = cv2.minAreaRect(largest)
                    angle_deg = rect[2]          # (-90, 0] degrees
                    angle_rad = -angle_deg * math.pi / 180.0
                    # Ensure the long axis is 'width'
                    rw, rh = rect[1]
                    if rw < rh:
                        angle_rad = _normalise_angle(angle_rad + math.pi / 2)

            d.angle = float(_normalise_angle(angle_rad))
            detections.append(d)

        return detections


# ── Helpers ───────────────────────────────────────────────────────────────

def _normalise_angle(angle_rad: float) -> float:
    """Wrap angle to (−π/2, π/2] — symmetric range for a box long axis."""
    while angle_rad >  math.pi / 2:
        angle_rad -= math.pi
    while angle_rad <= -math.pi / 2:
        angle_rad += math.pi
    return angle_rad


# ── Entry point ───────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
