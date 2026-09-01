import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from ultralytics import YOLO


class YoloDetectorNode(Node):
    """Runs YOLO on incoming frames and publishes the pixel centroid of the
    chosen target weed. Using stock YOLO for now (per instructions) - swap
    the model_path param for a fine-tuned weed model later, nothing else
    in this node needs to change."""

    def __init__(self):
        super().__init__('yolo_detector_node')

        self.declare_parameter('model_path', '/home/sunoj-orin/devansh/camera/yolo26l.engine')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('img_size', 640)
        # Class id(s) to treat as "weed" target. Empty list = accept all detections
        # for now since we're using stock YOLO, not a weed-specific model yet.
        self.declare_parameter('target_class_ids', [])

        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.img_size = self.get_parameter('img_size').value
        self.target_class_ids = self.get_parameter('target_class_ids').value

        self.bridge = CvBridge()
        self.model = YOLO(model_path)

        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.target_pub = self.create_publisher(PointStamped, '/weed_target', 10)
        self.debug_pub = self.create_publisher(Image, '/weed_detector/image_annotated', 1)

        self.get_logger().info('YOLO detector node started (model=%s)' % model_path)

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]
        img_center = (w / 2.0, h / 2.0)

        results = self.model(frame, imgsz=self.img_size, conf=self.conf_threshold, verbose=False)

        best_box = None
        best_dist = None
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if self.target_class_ids and cls_id not in self.target_class_ids:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dist = (cx - img_center[0]) ** 2 + (cy - img_center[1]) ** 2
            # Pick the detection nearest the image center - simplest useful
            # "next target" heuristic; swap for your own priority logic later.
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_box = (cx, cy)

        if best_box is not None:
            pt = PointStamped()
            pt.header.stamp = self.get_clock().now().to_msg()
            pt.header.frame_id = 'camera'
            pt.point.x = float(best_box[0])
            pt.point.y = float(best_box[1])
            pt.point.z = 0.0
            self.target_pub.publish(pt)

        annotated = results[0].plot()
        debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
