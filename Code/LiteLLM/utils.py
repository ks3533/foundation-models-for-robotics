from typing import Union

from Code.robocasa_env.main import Controller


def all_functions(controller: Controller):
    return {
        "get_eef_pos": controller.get_eef_pos,
        "get_eef_rot": controller.get_eef_rot,
        "resolve_object_from_name": controller.resolve_object_from_name,
        "open_gripper": controller.open_gripper,
        "close_gripper": controller.close_gripper,
        "move_abs": controller.move_abs
    }


# generates a set of available functions specific to the given controller instance from a set or list of function names
def available_function_generator(controller: Controller, available_functions: Union[list[str], set[str]]):
    # TODO limit the functions before returning
    return all_functions
