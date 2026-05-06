from setuptools import find_packages, setup
from glob import glob

package_name = 'vision_grasp'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/description', glob('description/*.urdf')),
        ('share/' + package_name + '/description/meshes', glob('description/meshes/*.STL')),
        ('share/' + package_name + '/config', glob('config/*.yaml') + glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lming',
    maintainer_email='dev@example.com',
    description='ROS 2 visual grasping system with simulation support',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = vision_grasp.camera_node:main',
            'vision_node = vision_grasp.vision_node:main',
            'tf_transformer_node = vision_grasp.tf_transformer_node:main',
            'ik_solver_node = vision_grasp.ik_solver_node:main',
            'arm_driver_node = vision_grasp.arm_driver_node:main',
            'trajectory_node = vision_grasp.trajectory_node:main',
        ],
    },
)
