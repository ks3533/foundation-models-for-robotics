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
        while True:
            self.env.step(self.movement)
            self.env.render()

    def start(self):
        self.simulation.start()

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
        print("done")

    def open_gripper(self):
        self.movement[6] = -1
        while (self.env.observation_spec()["robot0_gripper_qpos"][0] < 0.0395 or
               self.env.observation_spec()["robot0_gripper_qpos"][1] > -0.0395):
            pass
        self.movement = np.zeros(7)
        print("done")

    def close_gripper(self):
        self.movement[6] = 1
        while np.max(self.env.observation_spec()["robot0_gripper_qvel"]) < 0.01:
            pass
        while np.max(abs(self.env.observation_spec()["robot0_gripper_qvel"])) > 0.01:
            pass
        self.movement = np.zeros(7)
        print("done")

    def rotate_gripper_abs(self, end_rotation: Sequence[int]) -> None:
        """Rotates the gripper to an absolute end rotation"""
        # TODO make execution more consistent
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
                elif abs(rotation) > 1:
                    angle_velocity = self.min_angle_velocity
                if rotation < 0:
                    angle_velocity = -angle_velocity
                angle_velocities.append(angle_velocity)

            self.movement[3:6] = angle_velocities

            #break if the angle difference is nearly zero for all three axes
            if all(v == 0 for v in angle_velocities):
                break

        self.movement = np.zeros(7)
        print("done")

    def rotate_z(self, goal_rotation):
        """rotates the gripper around the z-axis to the next angle at which it can grip the milk properly (i.e. the grippers are parallel to the sides of the carton)"""
        """is interrupted when the distance between the target and current angle increases
        it still happens frequently that the robot is turning in the wrong direction and i cant figure out why"""
        last_diff = None
        
        possible_rotations = [
        goal_rotation,
        goal_rotation + 90,
        goal_rotation - 90,
        goal_rotation + 180,
        goal_rotation - 180
    ]
        end_rotation = min(possible_rotations, key=lambda r: abs(quat_to_euler(controller.env.observation_spec()["robot0_eef_quat"])[2] - r))
        
        while True:
            current_diff = quat_to_euler(controller.env.observation_spec()["robot0_eef_quat"])[2] - end_rotation
            
            if abs(current_diff) <= 1: break
            
            if abs(current_diff) > 5:
                angle_velocity = self.max_angle_velocity
            elif abs(current_diff) > 1:
                angle_velocity = self.min_angle_velocity
            else:
                angle_velocity = 0  
                break
            
            if current_diff < 0:
                angle_velocity = -angle_velocity

            if last_diff is not None and abs(last_diff) < abs(current_diff):
                print("moving wrong direction")
                break
                
            last_diff = current_diff
            self.movement[5] = angle_velocity
            print(f"goal: {end_rotation}")
            print(f"current_diff: {current_diff} v: {angle_velocity}")
            print(quat_to_euler(controller.env.observation_spec()["robot0_eef_quat"]))

        self.movement = np.zeros(7)
        print("done")

def quat_to_euler(quat: Sequence[int]) -> numpy.ndarray:
    """Takes a quaternion and returns it as an euler angle"""
    return Rotation.from_quat(quat).as_euler('xyz', degrees=True)


def subtract_angles(angles1, angles2):
    """Subtracts two angles in degrees from -180 and 180 and returns a result in the same range"""
    return (np.array(angles1)-angles2+180) % 360 - 180


if __name__ == "__main__":
    controller = Controller()
    controller.start()

    controller.open_gripper()
    controller.rotate_gripper_abs([90, 90, 0])
    controller.move(*(controller.env.observation_spec()["Milk_pos"] + [0, 0, 0.1]))
    controller.rotate_z((quat_to_euler(controller.env.observation_spec()["Milk_quat"])[2]))
    # TODO rotate that the milk doesn't fall back down
    #controller.rotate_gripper_abs([180, 0, quat_to_euler(controller.env.observation_spec()["Milk_quat"])[2] % 90])
    #controller.move(*(controller.env.observation_spec()["Milk_pos"] + [0, 0, 0.1]))
    controller.move(*(controller.env.observation_spec()["Milk_pos"] + [0, 0, 0.03]))
    controller.close_gripper()
    controller.move(*(controller.env.observation_spec()["robot0_eef_pos"] + [0, 0, 0.1]))
    pass
