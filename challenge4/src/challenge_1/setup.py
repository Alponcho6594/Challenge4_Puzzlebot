from setuptools import find_packages, setup

package_name = 'challenge_1'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/car_launch.py']),
        ('share/' + package_name + '/config', ['config/params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='stephs',
    maintainer_email='stephs@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_car = challenge_1.path_car:main',
            'odometry_car = challenge_1.odometry_car:main',
            'controller_car = challenge_1.controller_car:main',
            'vision_car = challenge_1.vision_car:main',
            'line_car = challenge_1.line_car:main'
        ],
    },
)
