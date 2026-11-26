# Single oriented bounding box detection from YOLOv8-OBB.
#
# Pixel coordinates follow image convention: origin top-left,
# x-axis right (columns), y-axis down (rows).
# Angle follows OpenCV minAreaRect convention: rotation of the
# long axis from the x-axis, in radians, range (−π/2 … π/2].

float32 x_center    # box centre, image columns [px]
float32 y_center    # box centre, image rows    [px]
float32 width       # OBB extent along long axis  [px]
float32 height      # OBB extent along short axis [px]
float32 angle       # yaw of long axis [rad], (−π/2 … π/2]
float32 confidence  # YOLO confidence score [0..1]
string  label       # class label, e.g. 'box_small'
