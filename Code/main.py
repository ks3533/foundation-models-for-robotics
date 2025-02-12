from math import atan2, pi
from time import sleep
from typing import Sequence

import numpy
import numpy as np

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from scipy.spatial.transform import Rotation

import threading


class Controller:
    def __init__(self):
        self.max_velocity = 0.2
        self.min_velocity = 0.02

        self.max_angle_velocity = 0.1
        self.min_angle_velocity = 0.01

        self.movement = np.zeros(7)

        self.simulation_is_running = True

        self.simulation = threading.Thread(target=self._simulate)
        options = {
            "env_name": "PickPlaceMilk",
            "robots": "Panda",
            "controller_configs": refactor_composite_controller_config(
                suite.load_part_controller_config(default_controller="OSC_POSE"), "Panda", ["right"]
            )

        }
        self.env = suite.make(
            **options,
            has_renderer=True,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            horizon=100000,
            control_freq=20,
        )
        self.env.reset()
        self.env.viewer.set_camera(camera_id=2)

    def _simulate(self):
        while self.simulation_is_running:
            self.env.step(self.movement)
            self.env.render()
        # when the simulation is finished
        self.env.close_renderer()
        self.env.reset()

    def start(self):
        self.simulation_is_running = True
        self.simulation.start()

    def stop(self):
        self.simulation_is_running = False
        self.simulation.join()

    def move(self, x, y, z):
        while np.max(abs(np.array([x, y, z]) - self.env.observation_spec()["robot0_eef_pos"])) > 0.02:
            vector = np.array([x, y, z]) - self.env.observation_spec()["robot0_eef_pos"]
            distance = np.linalg.norm(vector)
            velocity = 0
            if distance > 0.02:
                velocity = self.max_velocity
            elif distance > 0.002:
                velocity = self.min_velocity
            velocities = vector / distance * velocity
            self.movement[:3] = velocities
        self.movement = np.zeros(7)
        print(f"moved to {x}, {y}, {z}")

    def open_gripper(self):
        self.movement[6] = -1
        while (self.env.observation_spec()["robot0_gripper_qpos"][0] < 0.0395 or
               self.env.observation_spec()["robot0_gripper_qpos"][1] > -0.0395):
            pass
        self.movement = np.zeros(7)
        print("opened gripper")

    def close_gripper(self):
        self.movement[6] = 1
        while np.max(self.env.observation_spec()["robot0_gripper_qvel"]) < 0.01:
            pass
        while np.max(abs(self.env.observation_spec()["robot0_gripper_qvel"])) > 0.01:
            pass
        self.movement = np.zeros(7)
        print("closed gripper")

    def rotate_gripper_abs(self, end_rotation: Sequence[int]) -> None:
        """Rotates the gripper to an absolute end rotation"""
        # if end_rotation is quaternion, convert it to euler
        if len(end_rotation) == 4:
            end_rotation = quat_to_euler(end_rotation)
        # if it is not yet euler, it was neither quaternion nor euler, throw error
        if len(end_rotation) != 3:
            raise ValueError(f'"values" must either be euler rotation (3 values) '
                             f'or quaternion (4 values), not {len(end_rotation)} values')
        # while the angle differences are bigger than the tolerance
        while np.max(abs(subtract_angles(end_rotation, quat_to_euler(
                controller.env.observation_spec()["robot0_eef_quat"]))
                         )) > 0.5:
            vector = subtract_angles(end_rotation, quat_to_euler(controller.env.observation_spec()["robot0_eef_quat"]))
            angle_velocities = []
            for rotation in vector:
                angle_velocity = 0
                if abs(rotation) > 5:
                    angle_velocity = self.max_angle_velocity
                elif abs(rotation) > 2:
                    angle_velocity = self.min_angle_velocity
                if rotation < 0:
                    angle_velocity = -angle_velocity
                angle_velocities.append(angle_velocity)

            self.movement[3:6] = angle_velocities

            # break if the angle difference is nearly zero for all three axes
            if all(v == 0 for v in angle_velocities):
                break

        self.movement = np.zeros(7)
        print(f'rotated gripper to {", ".join([str(rot) for rot in end_rotation])}')

    def rotate_axis(self, end_rotation: Sequence[int], axis: int) -> None:
        """Rotates around one axis using only quaternions"""

        if len(end_rotation) == 3:
            end_rotation = Rotation.from_euler('xyz', end_rotation, degrees=True).as_quat()
        elif len(end_rotation) != 4:
            raise ValueError(f'"values" must be either Euler rotation (3 values) '
                             f'or quaternion (4 values), not {len(end_rotation)} values')

        while True:
            current_quat = self.env.observation_spec()["robot0_eef_quat"]

            # calculate the rotation required to get from the current rotation to the target rotation and convert it to a vector
            rotation_diff = Rotation.from_quat(end_rotation) * Rotation.from_quat(current_quat).inv()
            rotvec = rotation_diff.as_rotvec()

            # break if the distance to the target rotation is less than 0.1
            if np.linalg.norm(rotvec) < 0.1:
                break

            # Set rotation_speed to a value between max and min speed, matching the rotation-distance-vector
            rotation_speed = np.clip(rotvec, -self.max_angle_velocity, self.max_angle_velocity)
            if abs(rotation_speed[axis]) < self.min_angle_velocity:
                break

            axis_vector = np.zeros(3)
            axis_vector[axis] = rotation_speed[axis]

            self.movement[3:6] = axis_vector

        self.movement = np.zeros(7)
        print("done")

    def resolve_object_from_name(self, name: str) -> dict[str, list]:
        """Finds position and rotation of the object with the given name"""
        observation = self.env.observation_spec()
        return {"pos": observation[f"{name.capitalize()}_pos"], "quat": observation[f"{name.capitalize()}_quat"]}

    def pick_object(self, name: str) -> None:
        """Opens gripper, moves gripper to the object with the given name, then closes gripper"""
        controller.open_gripper()
        # controller.rotate_gripper_abs([90, 90, 0])
        controller.move(*(self.resolve_object_from_name(name)["pos"] + [0, 0, 0.1]))
        # controller.rotate_gripper_abs([90, 90, quat_to_euler(self.resolve_object_from_name(name)["quat"])[2] % 90])
        controller.move(*(self.resolve_object_from_name(name)["pos"] + [0, 0, 0]))
        controller.close_gripper()
        print(f'picked object "{name}"')

    def place_object(self, name: str) -> None:
        """Opens gripper and places object on ground"""
        initial_time = self.env.timestep
        prior_time = initial_time
        prior_pos = np.array(self.resolve_object_from_name(name)["pos"])
        self.movement[2] = -self.max_velocity

        while self.env.timestep == prior_time or \
                abs((prior_pos - self.resolve_object_from_name(name)["pos"])[2]) > 0.0001:
            if self.env.timestep == prior_time:
                sleep(1 / 80)
                continue
            prior_time = self.env.timestep
            prior_pos = np.array(self.resolve_object_from_name(name)["pos"])
        self.movement = np.zeros(7)
        self.open_gripper()
        print(f'placed object "{name}"')

    def match_orientation_object(self, name: str) -> None:
        """Try to match the orientation of object with given name and rotate gripper accordingly"""
        self.rotate_gripper_abs([90, 90, 0])
        obj = self.resolve_object_from_name(name)
        obj_rot = quat_to_euler(obj["quat"])
        robot_pos = self.env.robots[0].base_pos
        robot_rot = quat_to_euler(controller.env.observation_spec()["robot0_eef_quat"])
        direction_vector = (np.array(obj["pos"]) - robot_pos)[0:2]
        angle_to_robot = atan2(direction_vector[1], direction_vector[0]) / pi * 180 + 90
        best_angle = find_best_angle(obj_rot[2], angle_to_robot)
        print(f"{angle_to_robot}° to robot (robot: {robot_rot}, obj: {obj_rot}) -> best angle: {best_angle}°")
        self.rotate_axis([90, 90, best_angle], 2)
        print(angle_to_robot)


def find_best_angle(obj_rot, angle_to_robot):
    """Findet den Wert aus obj['pos'] und seinen ±90°/±180° Variationen, der angle_to_robot am nächsten ist."""
    possible_angles = [
        obj_rot,
        obj_rot + 90,
        obj_rot - 90,
        obj_rot + 180,
        obj_rot - 180
    ]
    return min(possible_angles, key=lambda x: subtract_angles([angle_to_robot+180], [x]))


def quat_to_euler(quat: Sequence[int]) -> numpy.ndarray:
    """Takes a quaternion and returns it as an euler angle"""
    return Rotation.from_quat(quat).as_euler('xyz', degrees=True)


def subtract_angles(angles1, angles2):
    """Subtracts two angles in degrees from -180 and 180 and returns a result in the same range"""
    return (np.array(angles1) - angles2 + 180) % 360 - 180


if __name__ == "__main__":
    controller = Controller()
    controller.start()

    controller.match_orientation_object("Milk")
    controller.pick_object("Milk")
    controller.move(*(controller.env.observation_spec()["robot0_eef_pos"] + [0, 0, 0.2]))
    controller.move(*(controller.env.target_bin_placements[0] + [0, 0, 0.2]))
    controller.place_object("Milk")
    controller.stop()
    print("finished simulation")
    pass
