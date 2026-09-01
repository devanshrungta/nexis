from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'weed_laser_pointer'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='devansh',
    maintainer_email='you@example.com',
    description='YOLO weed detection + closed-loop pan/tilt laser pointing on Jetson',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = weed_laser_pointer.camera_node:main',
            'yolo_detector_node = weed_laser_pointer.yolo_detector_node:main',
            'laser_detector_node = weed_laser_pointer.laser_detector_node:main',
            'servo_controller_node = weed_laser_pointer.servo_controller_node:main',
            'servo_driver_node = weed_laser_pointer.servo_driver_node:main',
        ],
    },
)
