import json
from typing import Union

from Code.robocasa_env.main import Controller

high_level_function_subset = {
    "open_gripper",
    "close_gripper",
    "grip_object_from_above",  # alternative to grip_object and the nonexistent grip_object_from_direction
    "press_button",
    "open_door",
    "close_door",
    "place_object_at_destination"
}


def all_functions(controller: Controller):
    functions = [
        controller.get_eef_pos,
        controller.get_eef_rot,
        controller.resolve_object_from_name,
        controller.open_gripper,
        controller.close_gripper,
        controller.move_abs,
        controller.grip_object,
        controller.grip_object_from_above,
        controller.press_button,
        controller.open_door,
        controller.close_door,
        controller.place_object_at_destination,
        controller.approach_destination_from_direction,
        controller.put_down_object_at_current_pos
    ]
    return {controller_function.__name__: controller_function for controller_function in functions}, \
        json.loads(open("robot_api.json", "r").read())


# generates a set of available functions specific to the given controller instance from a set or list of function names
def available_function_generator(controller: Controller, available_functions: Union[list[str], set[str]]):
    functions, tools = all_functions(controller)
    return {name: func for name, func in functions.items() if name in available_functions}, \
        [tool for tool in tools if tool["function"]["name"] in available_functions]


def high_level_functions(controller: Controller):
    return available_function_generator(controller, high_level_function_subset)
