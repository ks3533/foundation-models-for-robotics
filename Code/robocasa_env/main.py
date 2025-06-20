from math import atan2, pi
from time import sleep
import threading

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

import mujoco

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config

# noinspection PyUnresolvedReferences
import robocasa  # needed for the environments, doesn't find them otherwise
from robocasa.utils.object_utils import compute_rel_transform


class Controller:
    def __init__(self):
        self.max_velocity = 0.3
        self.min_velocity = 0.03

        self.max_angle_velocity = 0.1
        self.min_angle_velocity = 0.01

        options = {
            "env_name": "MicrowaveThawing",
            "robots": "PandaOmron",
            "controller_configs": load_composite_controller_config(robot="PandaOmron"),
            "layout_ids": [0],  # change for different kitchen layout
            "style_ids": [0]  # change for different kitchen style
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

        # array that contains all movement velocities that are to be executed in the next timestep
        # consists of xyz-velocities [0:3], xyz-rotation [3:6], gripper [6] and unknown [7:12]
        self.movement = np.zeros(self.action_dim)

        self.simulation_is_running = False

        self.simulation = threading.Thread(target=self._simulate)

        self.env.viewer.set_camera(camera_id=2)

    def _simulate(self) -> None:
        while self.simulation_is_running and not self.check_successful():
            self.env.step(self.movement)
            self.env.render()
        # when the simulation is finished
        self.env.close_renderer()
        self.env.reset()

    def start(self) -> None:
        self.simulation_is_running = True
        self.simulation.start()

    def stop(self) -> None:
        self.simulation_is_running = False
        self.simulation.join()

    # maybe add to available commands
    def check_gripping_object(self) -> bool:
        return sum(abs(self.env.observation_spec()["robot0_gripper_qpos"])) > 0.0011

    # maybe add to available commands
    def check_successful(self):
        # noinspection PyProtectedMember
        return self.env._check_success()

    def transform_to_robot_frame(self, coordinates: Sequence[int], orientation=np.identity(3)) \
            -> (np.ndarray, np.ndarray):
        """transforms coordinates and orientations as rotation matrices into the robot frame"""
        return compute_rel_transform(self.env.robots[0].base_pos, self.env.robots[0].base_ori, coordinates, orientation)

    def get_eef_pos(self) -> np.ndarray:
        """returns the eef pos relative to the robot frame"""
        return self.transform_to_robot_frame(self.env.observation_spec()["robot0_eef_pos"])[0]

    def get_eef_rot(self) -> np.ndarray:
        """returns the eef rotation relative to the robot frame"""
        return Rotation.from_matrix(
            self.transform_to_robot_frame(
                self.env.observation_spec()["robot0_eef_pos"],
                Rotation.from_quat(self.env.observation_spec()["robot0_eef_quat"]).as_matrix())[1]
        ).as_euler("xyz") / pi * 180

    def resolve_object_from_name(self, object_name: str) -> dict[str, list]:
        """finds position and rotation of the object with the given name and returns it in the robot frame"""
        observation = self.env.observation_spec()
        print(f'Resolved "{object_name}" '
              f'to ("pos": {observation[f"{object_name}_pos"]} "quat": {observation[f"{object_name}_quat"]})')
        result = self.transform_to_robot_frame(
            observation[f"{object_name}_pos"],
            Rotation.from_quat(observation[f"{object_name}_quat"]).as_matrix()
        )
        return {"pos": result[0], "rot": result[1]}

    def open_gripper(self) -> None:
        """opens gripper"""
        self.movement[6] = -1
        while (self.env.observation_spec()["robot0_gripper_qpos"][0] < 0.0395 and
               self.env.observation_spec()["robot0_gripper_qpos"][1] > -0.0395):
            pass
        self.movement[6] = 0
        print("opened gripper")

    def close_gripper(self) -> None:
        """closes gripper"""
        self.movement[6] = 1
        while np.max(self.env.observation_spec()["robot0_gripper_qvel"]) < 0.01:
            pass
        while np.max(abs(self.env.observation_spec()["robot0_gripper_qvel"])) > 0.01:
            pass
        # self.movement[6] = 0
        print("closed gripper")
        # return self.check_gripping_object()

    def move_abs(self, x, y, z) -> bool:
        """move to given coordinates relative to the robot frame"""
        print(f"Started moving to relative coordinates {x, y, z}")
        prior_vector = None
        prior_timestep = self.env.timestep
        while np.max(  # compute the maximum difference of start and goal
                abs(np.array([x, y, z]) - self.get_eef_pos())
        ) > 0.02:
            vector = np.array([x, y, z]) - self.get_eef_pos()
            distance = np.linalg.norm(vector)
            velocity = 0
            if distance > 0.02:
                velocity = self.max_velocity
            elif distance > 0.002:
                velocity = self.min_velocity
            velocities = vector / distance * velocity
            self.movement[:3] = velocities
            # if self.env.timestep % 80 == 0:
            #     if prior_vector is not None and sum(abs(prior_vector - vector)) < 0.1 \
            #             and self.env.timestep != prior_timestep:
            #         return False
            #     else:
            #         prior_vector = vector
            #         prior_timestep = self.env.timestep
            # if self.env.timestep % 500 == 0:
            #     print(f"Currently at relative coordinates {self.get_eef_pos()}")
        self.movement[:3] = np.zeros(3)
        print(f"moved to relative coordinates {x}, {y}, {z}")
        return True

    # TODO add to available commands
    def rotate_gripper_abs(self, end_rotation: Sequence[int]) -> None:
        """Rotates the gripper to an absolute end rotation relative to the robot frame"""
        # if end_rotation is quaternion, convert it to euler
        if len(end_rotation) == 4:
            end_rotation = quat_to_euler(end_rotation)
        # if it is not yet euler, it was neither quaternion nor euler, throw error
        if len(end_rotation) != 3:
            raise ValueError(f'"values" must either be euler rotation (3 values) '
                             f'or quaternion (4 values), not {len(end_rotation)} values')
        # while the angle differences are bigger than the tolerance
        while np.max(abs(subtract_angles(end_rotation, self.get_eef_rot()))) > 1:
            vector = subtract_angles(end_rotation, self.get_eef_rot())
            angle_velocities = []
            for rotation in vector:
                angle_velocity = 0
                if abs(rotation) > 5:
                    angle_velocity = self.max_angle_velocity
                elif abs(rotation) > 1:
                    angle_velocity = self.min_angle_velocity
                if rotation < 0:
                    angle_velocity = -angle_velocity
                angle_velocities.append(angle_velocity)

            self.movement[3:6] = angle_velocities

            # break if the angle difference is nearly zero for all three axes
            if all(v == 0 for v in angle_velocities):
                break

        self.movement[3:6] = np.zeros(3)
        print(f'rotated gripper to {", ".join([str(rot) for rot in end_rotation])}')

    # TODO add to available commands
    def rotate_axis(self, end_rotation: Sequence[int], axis: int) -> None:
        # maybe needs fixing because of relative coordinates
        """Rotates around one axis using only quaternions"""

        if len(end_rotation) == 3:
            end_rotation = Rotation.from_euler('xyz', end_rotation, degrees=True).as_quat(False)
        elif len(end_rotation) != 4:
            raise ValueError(f'"values" must be either Euler rotation (3 values) '
                             f'or quaternion (4 values), not {len(end_rotation)} values')

        while True:
            current_quat = self.env.observation_spec()["robot0_eef_quat"]
            # current_quat = Rotation.from_matrix(np.dot(Rotation.from_quat(self.env.observation_spec()["robot0_eef_quat"]).as_matrix(), self.env.robots[0].base_ori)).as_quat(False)

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

        self.movement[3:6] = np.zeros(3)
        print("done")

    # currently unused
    # maybe add to available commands
    def pick_object(self, object_name: str) -> bool:
        # maybe needs fixing because of relative coordinates
        """Opens gripper, moves gripper to the object with the given name, then closes gripper"""
        self.open_gripper()
        # self.rotate_gripper_abs([90, 90, 0])
        if not self.move_abs(*(self.resolve_object_from_name(object_name)["pos"] + [0, 0, 0.1])):
            return False
        # self.rotate_gripper_abs([90, 90, quat_to_euler(self.resolve_object_from_name(object_name)["quat"])[2] % 90])
        if not self.move_abs(*(self.resolve_object_from_name(object_name)["pos"] + [0, 0, 0])):
            return False
        self.close_gripper()
        print(f'picked object "{object_name}"')
        return True

    def approach_destination_from_direction(self, destination: Sequence[float], direction: str) -> bool:
        """move along every axis except for the given direction, then move to destination
        uses coordinates relative to the robot frame"""
        match direction:
            case "front":
                matrix = np.array([
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ])
            case "left":
                matrix = np.array([
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1]
                ])
            case "right":
                matrix = np.array([
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1]
                ])
            case "up":
                matrix = np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]
                ])
            case _:
                matrix = np.identity(3)
        if not self.move_abs(*(
                np.dot(matrix, destination)
                + np.dot(np.identity(3, dtype=int) - matrix, self.get_eef_pos())
        )) or not self.move_abs(*destination):
            return False
        return True

    # currently unused
    # maybe add to available commands
    def grip_object_with_rotation_offset(self, object_name: str, rotation_offset: int) -> bool:
        # maybe needs fixing because of relative coordinates
        """Opens gripper, moves gripper to the object with the given name, then closes gripper"""
        self.open_gripper()
        if not self.move_abs(*(self.resolve_object_from_name(object_name)["pos"] + [0, 0, 0.1])):
            return False
        self.match_orientation_with_offset(object_name, rotation_offset)
        if not self.move_abs(*(self.resolve_object_from_name(object_name)["pos"] + [0, 0, -0.01])):
            return False
        self.close_gripper()
        print(f'picked object "{object_name}"')
        return True

    def grip_object(self, object_name: str) -> bool:
        # maybe needs fixing because of relative coordinates
        """Opens gripper, moves gripper to the object with the given name, then closes gripper"""
        self.open_gripper()
        if not self.move_abs(*(self.resolve_object_from_name(object_name)["pos"] + [0, 0, -0.01])):
            return False
        self.close_gripper()
        print(f'picked object "{object_name}"')
        return True

    def grip_object_from_above(self, object_name: str) -> bool:
        # maybe needs fixing because of relative coordinates
        """Opens gripper, moves gripper to the object with the given name, then closes gripper"""
        self.rotate_gripper_abs([180, 0, 0])
        self.open_gripper()
        if (not self.move_abs(*(self.resolve_object_from_name(object_name)["pos"] + [0, 0, 0.15])) or
                not self.move_abs(*(self.resolve_object_from_name(object_name)["pos"] + [0, 0, -0.01]))):
            return False
        self.close_gripper()
        print(f'picked object "{object_name}"')
        return True

    def press_button(self) -> bool:
        """presses the button of the microwave"""
        button_pos_abs = self.env.sim.data.get_body_xpos(self.env.microwave.door_name) \
                         + np.dot(np.array([-0.22, -0.33, -0.105]), self.env.robots[0].base_ori.T)
        self.render_coordinate_frame(*button_pos_abs, None)
        button_pos_rel = self.transform_to_robot_frame(button_pos_abs)[0]
        self.rotate_axis([-180, -90, 0], 1)
        self.close_gripper()
        if not self.move_abs(*(self.get_eef_pos() + [-0.05, 0, 0])):
            return False
        self.approach_destination_from_direction(button_pos_rel, "front")
        self.movement[0] = self.max_velocity
        sleep(1)
        self.movement[0] = 0
        return True

    def open_door(self) -> bool:
        """opens the microwave door"""
        # alternatively self.env.sim.data.get_body_xpos(self.env.microwave.door_name)
        handle_pos_abs = self.env.microwave.pos + np.dot(np.array([-0.22, -0.18, 0]), self.env.robots[0].base_ori.T)
        # self.render_coordinate_frame(*handle_pos_abs, None)
        # controller.env.microwave.pos - controller.env.microwave.size*[-0.21, 0.5, 0]
        handle_pos_rel = self.transform_to_robot_frame(handle_pos_abs)[0]
        self.open_gripper()
        if not self.move_abs(*(self.get_eef_pos() + [-0.1, 0, 0])):
            return False
        self.rotate_axis([-180, -90, 0], 1)
        self.approach_destination_from_direction(handle_pos_rel, "front")
        self.close_gripper()
        joint_pos, _ = self.transform_to_robot_frame(self.env.sim.data.joint(self.env.microwave.joints[0]).xanchor)
        vector = (self.get_eef_pos() - joint_pos)[:2]
        vector = vector / np.linalg.norm(vector)
        while abs(vector[1]) > 0.15:
            tangential_vector = np.dot(vector, np.array([[0, -1], [1, 0]]))
            self.movement[0:2] = tangential_vector * self.max_velocity

            vector = (self.get_eef_pos() - joint_pos)[:2]
            vector = vector / np.linalg.norm(vector)
        self.movement[0:2] = np.zeros(2)
        self.open_gripper()
        if not (self.move_abs(*(self.get_eef_pos() + [-0.1, 0, 0]))
                and self.move_abs(*(self.get_eef_pos() + [0, -0.25, 0]))
                and self.move_abs(*(self.get_eef_pos() + [0.1, 0, 0]))):
            return False
        vector = (self.get_eef_pos() - joint_pos)[:2]
        vector = vector / np.linalg.norm(vector)
        # possibly in world coordinates, requires testing
        self.movement[1] = self.max_velocity
        while abs(vector[1]) > 0.2:
            vector = (self.get_eef_pos() - joint_pos)[:2]
            vector = vector / np.linalg.norm(vector)
        self.movement[1] = 0
        if not self.move_abs(*(self.get_eef_pos() + [0.15, -0.1, -0.26])):
            return False
        print("opened door")
        return True

    def close_door(self) -> bool:
        """closes the microwave door"""
        joint_pos, _ = self.transform_to_robot_frame(self.env.sim.data.joint(self.env.microwave.joints[0]).xanchor)
        microwave_pos = self.transform_to_robot_frame(self.env.microwave.pos)[0]
        if not (self.move_abs(*(microwave_pos + [-0.4, 0.1, -0.05]))
                and self.move_abs(*(microwave_pos + [-0.3, 0.1, -0.3]))):
            return False
        self.rotate_axis([-180, -90, 0], 1)
        self.approach_destination_from_direction(
            [joint_pos[0] - 0.15, joint_pos[1] + 0.2, microwave_pos[2] - 0.08], "up"
        )
        if not self.move_abs(joint_pos[0] - 0.15, microwave_pos[1], microwave_pos[2] - 0.05):
            return False
        self.movement[0] = self.max_velocity
        sleep(2)
        self.movement[0] = 0
        print("closed door")
        return True

    # currently unused in main routine
    def put_down_object_at_current_pos(self, object_name: str) -> None:
        # maybe needs fixing because of relative coordinates
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
        self.movement[2] = 0
        self.open_gripper()
        print(f'placed object "{object_name}"')

    def place_object_at_destination(self, object_name: str, destination_name: str = None, height_offset: int = 0.1,
                                    front_offset: int = -0.05) -> None:
        """Opens gripper and drops object on ground (optionally at destination)"""
        if destination_name is not None:
            dest_pos = self.resolve_object_from_name(destination_name)["pos"]
            self.approach_destination_from_direction(dest_pos + [front_offset, 0, height_offset], "front")
        self.open_gripper()

        print(f'placed object {object_name}{f" at {destination_name}" if destination_name is not None else ""}')

    # maybe add to available commands
    def match_orientation_with_offset(self, object_name: str, offset: int) -> None:
        # maybe needs fixing because of relative coordinates
        """Try to match the orientation of object with given name (and offset) and rotate gripper accordingly"""
        # maybe fix
        obj = self.resolve_object_from_name(object_name)
        obj_rot = Rotation.from_matrix(obj["rot"]).as_euler("xyz")
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
        self.rotate_axis([0, 0, (obj_rot[2] % 90) - 90 + offset], 2)
        print(angle_to_robot)

    # maybe add to available commands
    def render_coordinate_frame(self, x, y, z, rotation_matrix=None) -> None:
        # todo add rotation_matrix
        sleep(4)
        viewer = self.env.viewer.viewer
        shape = mujoco.mjtGeom.mjGEOM_ARROW
        size = np.array([0.01, 0.01, 0.1], dtype=np.float64)
        position = np.array([x, y, z], dtype=np.float64)
        axes = np.array([
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
        ], dtype=np.float64)
        colors = np.array([
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1]
        ], dtype=np.float32)
        for axis, color in zip(axes, colors):
            viewer.user_scn.ngeom += 1
            mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom - 1], shape,
                                size=size,
                                pos=position, mat=axis.flatten(),
                                rgba=color)


def quat_to_euler(quat: Sequence[int]) -> np.ndarray:
    """Takes a quaternion and returns it as an euler angle"""
    return Rotation.from_quat(quat).as_euler('xyz', degrees=True)


def subtract_angles(angles1, angles2):
    """Subtracts two angles in degrees from -180 and 180 and returns a result in the same range"""
    return (np.array(angles1) - angles2 + 180) % 360 - 180


if __name__ == "__main__":
    controller = Controller()
    try:
        controller.start()

        controller.open_door()
        # controller.close_door()  # for testing

        controller.approach_destination_from_direction(
            controller.resolve_object_from_name("obj")["pos"] + [0, 0, 0.15],
            "front"
        )
        controller.rotate_gripper_abs([180, 0, 0])
        # controller.grip_object_with_rotation_offset("obj", 0)
        controller.grip_object("obj")

        controller.move_abs(*(controller.get_eef_pos() + [0, 0, 0.1]))
        controller.move_abs(*(controller.get_eef_pos() * [1, 0, 1] + [0, controller.transform_to_robot_frame(
            controller.env.microwave.pos)[0][1], 0]))
        controller.move_abs(*(controller.get_eef_pos() + [-0.12, 0, 0.1]))
        controller.rotate_gripper_abs([180, -90, 0])
        controller.place_object_at_destination("obj", "container")

        controller.close_door()

        controller.press_button()

        # controller.stop()
        print("finished simulation")
    except KeyboardInterrupt:
        print("received KeyboardInterrupt, stopping simulation")
        controller.stop()
        exit()
