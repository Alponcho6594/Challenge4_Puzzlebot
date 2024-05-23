import os 
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    config = os.path.join(

        get_package_share_directory('challenge_1'),
        'config',
        'params.yaml'
    )

    control_node = Node(

        package = 'challenge_1',
        executable = 'line_car',
        output = 'screen'
    )

    vision_node = Node(

        package = 'challenge_1',
        executable = 'vision_car',
        output = 'screen'
    )

    rqt_graph_node = Node(

        package = 'rqt_graph',
        executable = 'rqt_graph',
    )

    return LaunchDescription([control_node, vision_node, rqt_graph_node])