import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import Jetson.GPIO as GPIO


class ServoDriverNode(Node):
    """Subscribes to /servo_cmd (Float32, degrees) and drives a single
    rotation servo via Jetson.GPIO software PWM.

    min_us / max_us / max_angle come from your manual calibration (jog to
    each physical limit, note the pulse width) - update the defaults below
    to match what you locked in. The clamp here is the last line of
    defense: no matter what any upstream node publishes, the servo
    physically cannot be commanded past these limits.
    """

    def __init__(self):
        super().__init__('servo_driver_node')

        self.declare_parameter('pin', 32)          # BOARD numbering, hardware PWM capable
        self.declare_parameter('pwm_freq_hz', 50)

        # Set these to the values you found with the calibration script.
        self.declare_parameter('min_us', 500.0)
        self.declare_parameter('max_us', 2500.0)
        self.declare_parameter('max_angle', 180.0)  # replace with your measured real swing

        # Only re-send a duty cycle if the change is meaningful - reduces
        # needless PWM churn that was contributing to jitter/buzz at rest.
        self.declare_parameter('min_change_deg', 0.5)

        self.pin = self.get_parameter('pin').value
        freq = self.get_parameter('pwm_freq_hz').value

        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setup(self.pin, GPIO.OUT)

        self.pwm = GPIO.PWM(self.pin, freq)
        self.pwm.start(0)

        self.last_angle = None

        self.create_subscription(Float32, '/servo_cmd', self.cmd_callback, 10)
        self.get_logger().info('Servo driver node started (pin=%d, 1 DOF)' % self.pin)

    def angle_to_duty(self, angle):
        min_us = self.get_parameter('min_us').value
        max_us = self.get_parameter('max_us').value
        max_angle = self.get_parameter('max_angle').value

        angle = max(0.0, min(max_angle, angle))  # hard clamp - critical, this points a laser
        pulse_us = min_us + (angle / max_angle) * (max_us - min_us)
        return (pulse_us / 20000.0) * 100.0  # 20000us = period at 50Hz

    def cmd_callback(self, msg: Float32):
        angle_deg = msg.data
        min_change = self.get_parameter('min_change_deg').value

        if self.last_angle is None or abs(angle_deg - self.last_angle) >= min_change:
            duty = self.angle_to_duty(angle_deg)
            self.pwm.ChangeDutyCycle(duty)
            self.last_angle = angle_deg

    def destroy_node(self):
        self.pwm.stop()
        GPIO.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ServoDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
