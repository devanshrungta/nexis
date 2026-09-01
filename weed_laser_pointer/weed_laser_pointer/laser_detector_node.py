import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge


class LaserDetectorNode(Node):
    """Finds the laser dot in the camera frame and publishes its pixel
    position. This is the sensor that closes the feedback loop - the
    controller compares this against /weed_target instead of trusting
    open-loop angle math.

    Detection strategy: threshold for very bright + saturated pixels in the
    laser's color range, then take the centroid of the largest blob.
    Tune hsv_lower/hsv_upper for your actual laser color and ambient light -
    defaults below are a starting point for a red laser."""

    def __init__(self):
        super().__init__('laser_detector_node')

        self.declare_parameter('hsv_lower', [0, 100, 200])
        self.declare_parameter('hsv_upper', [10, 255, 255])
        # Some red lasers also show up near the wrap-around of the hue circle
        self.declare_parameter('hsv_lower2', [170, 100, 200])
        self.declare_parameter('hsv_upper2', [180, 255, 255])
        self.declare_parameter('min_blob_area', 4)

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.laser_pub = self.create_publisher(PointStamped, '/laser_position', 10)
        self.debug_pub = self.create_publisher(Image, '/laser_detector/image_debug', 1)

        self.get_logger().info('Laser detector node started')

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower1 = np.array(self.get_parameter('hsv_lower').value)
        upper1 = np.array(self.get_parameter('hsv_upper').value)
        lower2 = np.array(self.get_parameter('hsv_lower2').value)
        upper2 = np.array(self.get_parameter('hsv_upper2').value)

        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = self.get_parameter('min_blob_area').value
        best = None
        best_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > best_area and area >= min_area:
                best_area = area
                best = c

        if best is not None:
            M = cv2.moments(best)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                pt = PointStamped()
                pt.header.stamp = self.get_clock().now().to_msg()
                pt.header.frame_id = 'camera'
                pt.point.x = float(cx)
                pt.point.y = float(cy)
                pt.point.z = 0.0
                self.laser_pub.publish(pt)

        debug_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LaserDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
