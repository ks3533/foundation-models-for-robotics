import numpy as np

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config

import threading


class Controller:
    def __init__(self):
        self.max_velocity = 0.2
        self.min_velocity = 0.02

        self.movement = np.zeros(7)

        self.simulation = threading.Thread(target=self._simulate)
        options = {
            "env_name": "PickPlaceMilk",
            "robots": "Panda",
            "controller_configs": refactor_composite_controller_config(
                suite.load_part_controller_config(default_controller="OSC_POSE"), "Panda", ["right", "left"]
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
        while np.max(abs(np.array([x, y, z])-self.env.observation_spec()["robot0_eef_pos"])) > 0.2:
            vector = np.array([x, y, z])-self.env.observation_spec()["robot0_eef_pos"]
            distance = np.linalg.norm(vector)
            velocity = 0
            if distance > 0.2:
                velocity = self.max_velocity
            elif distance > 0.02:
                velocity = self.min_velocity
            velocities = np.array(vector/distance*velocity)
            self.movement[:3] = velocities
        self.movement = np.zeros(7)
        print("done")


if __name__ == "__main__":
    controller = Controller()
    controller.start()
    controller.move(0.3, 0, 1)
