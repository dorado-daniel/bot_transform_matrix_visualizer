from robot_kinematics_visualizer.load_robot import load_robot
from robot_kinematics_visualizer.pyvista_viewer import show_robot_viewer


robot = load_robot()
show_robot_viewer(robot)
