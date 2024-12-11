import numpy as np

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config


class Controller:
    def __init__(self):
        options = {
            "env_name": "PickPlaceMilk",
            "robots": "Panda",
            "controller_configs": refactor_composite_controller_config(
                suite.load_part_controller_config(default_controller="OSC_POSE"), "Panda", ["right", "left"]
            )

        }
        env = suite.make(
            **options,
            has_renderer=True,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            horizon=100000,
            control_freq=20,
        )
        env.reset()
        env.viewer.set_camera(camera_id=2)
        while True:
            env.step(np.zeros(7))
            env.render()


if __name__ == "__main__":
    controller = Controller()
