import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32


class ServoControllerNode(Node):
    """Closed-loop visual servoing, 1 DOF (rotation only).

    Continuously drives the laser dot (/laser_position) toward the current
    weed target (/weed_target) by nudging the single rotation servo. Only
    the horizontal (x) pixel error is used, since the translation axis
    isn't implemented yet - vertical error is ignored for now.

    Publishes the commanded angle (degrees) as Float32 on /servo_cmd for
    servo_driver_node to actually drive. The driver hard-clamps this to
    the calibrated MIN/MAX limits regardless of what's published here.
    """

    def __init__(self):
        super().__init__('servo_controller_node')

        self.declare_parameter('kp', 0.02)              # deg per pixel of x-error - TUNE THIS
        self.declare_parameter('pixel_tolerance', 6.0)   # "on target" threshold, x-axis only
        self.declare_parameter('max_step_deg', 2.0)      # per-cycle slew limit
        self.declare_parameter('angle_min_deg', 0.0)
        self.declare_parameter('angle_max_deg', 180.0)   # set to your calibrated real swing
        self.declare_parameter('target_timeout_s', 1.0)
        self.declare_parameter('control_rate_hz', 30.0)
        self.declare_parameter('invert', False)          # flip if it diverges instead of converges

        self.weed_target = None
        self.weed_target_time = 0.0
        self.laser_pos = None
        self.laser_pos_time = 0.0

        self.angle_deg = (self.get_parameter('angle_min_deg').value
                           + self.get_parameter('angle_max_deg').value) / 2.0

        self.create_subscription(PointStamped, '/weed_target', self.weed_cb, 10)
        self.create_subscription(PointStamped, '/laser_position', self.laser_cb, 10)
        self.cmd_pub = self.create_publisher(Float32, '/servo_cmd', 10)

        rate = self.get_parameter('control_rate_hz').value
        self.timer = self.create_timer(1.0 / rate, self.control_loop)

        self.get_logger().info('Servo controller node started (1 DOF - rotation only)')

    def weed_cb(self, msg: PointStamped):
        self.weed_target = (msg.point.x, msg.point.y)
        self.weed_target_time = time.time()

    def laser_cb(self, msg: PointStamped):
        self.laser_pos = (msg.point.x, msg.point.y)
        self.laser_pos_time = time.time()

    def control_loop(self):
        now = time.time()
        timeout = self.get_parameter('target_timeout_s').value

        target_fresh = self.weed_target is not None and (now - self.weed_target_time) < timeout
        laser_fresh = self.laser_pos is not None and (now - self.laser_pos_time) < timeout

        if not (target_fresh and laser_fresh):
            # No weed to aim at, or we've lost sight of the laser dot -
            # hold position rather than guessing.
            return

        # Only horizontal error matters for a single rotation axis.
        err_x = self.weed_target[0] - self.laser_pos[0]

        if abs(err_x) <= self.get_parameter('pixel_tolerance').value:
            self.get_logger().debug('On target (x err=%.1fpx)' % err_x)
            return  # locked on - hold current angle, don't dither

        kp = self.get_parameter('kp').value
        max_step = self.get_parameter('max_step_deg').value
        step = max(-max_step, min(max_step, kp * err_x))

        if self.get_parameter('invert').value:
            step = -step

        angle_min = self.get_parameter('angle_min_deg').value
        angle_max = self.get_parameter('angle_max_deg').value
        self.angle_deg = max(angle_min, min(angle_max, self.angle_deg + step))

        msg = Float32()
        msg.data = self.angle_deg
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ServoControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
