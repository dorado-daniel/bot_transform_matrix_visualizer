from roboticstoolbox.models.URDF import Puma560


def load_robot():
    robot = Puma560()
    robot.q = robot.qz
    return robot
