from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='weed_laser_pointer',
            executable='camera_node',
            name='camera_node',
            output='screen',
        ),
        Node(
            package='weed_laser_pointer',
            executable='yolo_detector_node',
            name='yolo_detector_node',
            output='screen',
            parameters=[{'model_path': '/home/sunoj-orin/devansh/camera/yolo26l.engine'}],
        ),
        Node(
            package='weed_laser_pointer',
            executable='laser_detector_node',
            name='laser_detector_node',
            output='screen',
        ),
        Node(
            package='weed_laser_pointer',
            executable='servo_controller_node',
            name='servo_controller_node',
            output='screen',
        ),
        Node(
            package='weed_laser_pointer',
            executable='servo_driver_node',
            name='servo_driver_node',
            output='screen',
            parameters=[{
                'pin': 32,
                'min_us': 500.0,   # set to your calibrated MIN
                'max_us': 2500.0,  # set to your calibrated MAX
                'max_angle': 180.0,
            }],
        ),
    ])
