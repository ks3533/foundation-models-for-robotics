from math import atan2, pi
from time import sleep
from typing import Sequence

import numpy
import numpy as np

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config
from scipy.spatial.transform import Rotation
import robocasa  # needed for the environments, doesn't find them otherwise
from robocasa.utils.object_utils import compute_rel_transform

import threading


# TODO force kitchen variant


class Controller:
    def __init__(self):
        self.max_velocity = 0.2
        self.min_velocity = 0.02

        self.max_angle_velocity = 0.1
        self.min_angle_velocity = 0.01

        options = {
            "env_name": "MicrowaveThawing",
            "robots": "PandaOmron",
            "controller_configs": load_composite_controller_config(robot="PandaOmron"),

        }
        self.env = suite.make(
            **options,
            has_renderer=True,
            has_offscreen_renderer=False,
            render_camera=None,
            ignore_done=True,
            use_camera_obs=False,
            control_freq=80,
            renderer="mjviewer"
        )

        self.env.reset()

        self.action_dim = self.env.action_spec[0].shape[0]

        self.movement = np.zeros(self.action_dim)

        self.simulation_is_running = True

        self.simulation = threading.Thread(target=self._simulate)

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
        print(f"Started moving to {x, y, z}")
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
            if self.env.timestep % 20 == 0:
                print(f"Currently at {self.env.observation_spec()['robot0_eef_pos']}")
        self.movement = np.zeros(self.action_dim)
        print(f"moved to {x}, {y}, {z}")

    def move_relative_to_robot(self, x, y, z):
        print(f"Started moving to {x, y, z}")
        robot_orientation = self.env.robots[0].base_ori
        object_orientation = numpy.identity(3)
        while np.max(  # compute the maximum difference of start and goal
                abs(
                    compute_rel_transform(
                        np.zeros(3),
                        robot_orientation,
                        np.array([x, y, z]) - self.env.observation_spec()["robot0_eef_pos"],
                        object_orientation
                    )[0]
                )
        ) > 0.02:
            # print(f'original {np.array([x, y, z]) - self.env.observation_spec()["robot0_eef_pos"]} now {self.env.observation_spec()["obj_to_robot0_eef_pos"]}')
            vector = compute_rel_transform(
                        np.zeros(3),
                        robot_orientation,
                        np.array([x, y, z]) - self.env.observation_spec()["robot0_eef_pos"],
                        object_orientation
                    )[0]
            distance = np.linalg.norm(vector)
            velocity = 0
            if distance > 0.02:
                velocity = self.max_velocity
            elif distance > 0.002:
                velocity = self.min_velocity
            velocities = vector / distance * velocity
            self.movement[:3] = velocities
            if self.env.timestep % 20 == 0:
                print(f"Currently at {self.env.observation_spec()['robot0_eef_pos']}")
        self.movement = np.zeros(self.action_dim)
        print(f"moved to {x}, {y}, {z}")

    def open_gripper(self):
        self.movement[6] = -1
        while (self.env.observation_spec()["robot0_gripper_qpos"][0] < 0.0395 or
               self.env.observation_spec()["robot0_gripper_qpos"][1] > -0.0395):
            pass
        self.movement = np.zeros(self.action_dim)
        print("opened gripper")

    def close_gripper(self):
        self.movement[6] = 1
        while np.max(self.env.observation_spec()["robot0_gripper_qvel"]) < 0.01:
            pass
        while np.max(abs(self.env.observation_spec()["robot0_gripper_qvel"])) > 0.01:
            pass
        self.movement = np.zeros(self.action_dim)
        print("closed gripper")

    def get_eef_pos(self):
        return self.env.observation_spec()["robot0_eef_pos"]

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
                self.env.observation_spec()["robot0_eef_quat"]))
                         )) > 0.5:
            vector = subtract_angles(end_rotation, quat_to_euler(self.env.observation_spec()["robot0_eef_quat"]))
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

        self.movement = np.zeros(self.action_dim)
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

        self.movement = np.zeros(self.action_dim)
        print("done")

    def resolve_object_from_name(self, object_name: str) -> dict[str, list]:
        """Finds position and rotation of the object with the given name"""
        observation = self.env.observation_spec()
        print(f'Resolved "{object_name}" to ("pos": {observation[f"{object_name}_pos"]} "quat": {observation[f"{object_name}_quat"]})')
        return {"pos": observation[f"{object_name}_pos"], "quat": observation[f"{object_name}_quat"]}

    def pick_object(self, object_name: str) -> None:
        """Opens gripper, moves gripper to the object with the given name, then closes gripper"""
        self.open_gripper()
        # self.rotate_gripper_abs([90, 90, 0])
        self.move_relative_to_robot(*(self.resolve_object_from_name(object_name)["pos"] + [0, 0, 0.1]))
        # self.rotate_gripper_abs([90, 90, quat_to_euler(self.resolve_object_from_name(object_name)["quat"])[2] % 90])
        self.move_relative_to_robot(*(self.resolve_object_from_name(object_name)["pos"] + [0, 0, 0]))
        self.close_gripper()
        print(f'picked object "{object_name}"')

    def place_object_from_direction(self, object_name: str, destination: Sequence[int], direction: str):
        # todo fix, use robot orientation to get the direction vector, then move along every axis except for the given vector
        robot_orientation = self.env.robots[0].base_ori
        # compute_rel_transform(
        #     np.zeros(3),
        #     robot_orientation,
        #     np.array(destination) - self.env.observation_spec()["robot0_eef_pos"],
        #     np.identity(3)
        # )[0]
        match direction:
            case "front":
                pass
            case "left":
                pass
            case "right":
                pass
            case "up":
                pass
        self.move()
        self.place_object(object_name)

    def grip_object_with_rotation_offset(self, object_name: str, rotation_offset: int):
        """Opens gripper, moves gripper to the object with the given name, then closes gripper"""
        self.open_gripper()
        self.move_relative_to_robot(*(self.resolve_object_from_name(object_name)["pos"] + [0, 0, 0.1]))
        self.match_orientation_with_offset(object_name, rotation_offset)
        self.move_relative_to_robot(*(self.resolve_object_from_name(object_name)["pos"] + [0, 0, 0]))
        self.close_gripper()
        print(f'picked object "{object_name}"')
        pass

    def press_button(self):
        pass

    def open_door(self):
        pass

    def close_door(self):
        pass

    def place_object(self, object_name: str) -> None:
        """Opens gripper and places object on ground"""
        initial_time = self.env.timestep
        prior_time = initial_time
        prior_pos = np.array(self.resolve_object_from_name(object_name)["pos"])
        self.movement[2] = -self.max_velocity

        while self.env.timestep == prior_time or \
                abs((prior_pos - self.resolve_object_from_name(object_name)["pos"])[2]) > 0.0001:
            if self.env.timestep == prior_time:
                sleep(1 / 80)
                continue
            prior_time = self.env.timestep
            prior_pos = np.array(self.resolve_object_from_name(object_name)["pos"])
        self.movement = np.zeros(self.action_dim)
        self.open_gripper()
        print(f'placed object "{object_name}"')

    def match_orientation_with_offset(self, object_name: str, offset: int) -> None:
        """Try to match the orientation of object with given name (and offset) and rotate gripper accordingly"""
        # TODO fix
        obj = self.resolve_object_from_name(object_name)
        obj_rot = quat_to_euler(obj["quat"])
        robot_pos = self.env.robots[0].base_pos
        # robot_orientation = self.env.robots[0].base_ori
        direction_vector = (np.array(obj["pos"]) - robot_pos)
        angle_to_robot = atan2(direction_vector[1], direction_vector[0]) / pi * 180 + 90
        # rotation_matrix = compute_rel_transform(
        #     np.zeros(3),
        #     robot_orientation,
        #     direction_vector,
        #     np.identity(3)
        # )[0]
        self.rotate_axis([0, 0, (obj_rot[2] % 90)+offset], 2)
        print(angle_to_robot)


# def find_best_angle(obj_rot, angle_to_robot):
#     """Findet den Wert aus obj['pos'] und seinen ±90°/±180° Variationen, der angle_to_robot am nächsten ist."""
#     possible_angles = [
#         obj_rot,
#         obj_rot + 90,
#         obj_rot - 90,
#         obj_rot + 180,
#         obj_rot - 180
#     ]
#     return min(possible_angles, key=lambda x: subtract_angles([angle_to_robot+180], [x]))


def quat_to_euler(quat: Sequence[int]) -> numpy.ndarray:
    """Takes a quaternion and returns it as an euler angle"""
    return Rotation.from_quat(quat).as_euler('xyz', degrees=True)


def subtract_angles(angles1, angles2):
    """Subtracts two angles in degrees from -180 and 180 and returns a result in the same range"""
    return (np.array(angles1) - angles2 + 180) % 360 - 180


if __name__ == "__main__":
    controller = Controller()
    controller.start()

    controller.grip_object_with_rotation_offset("obj", 0)
    controller.move(*(controller.get_eef_pos() + [0, 0, 0.2]))
    # controller.move(*(controller.env.target_bin_placements[0] + [0, 0, 0.2]))
    controller.place_object("obj")
    controller.stop()
    print("finished simulation")
    pass
