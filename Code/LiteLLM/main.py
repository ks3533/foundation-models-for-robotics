from os import environ
from typing import List
import argparse
from pathlib import Path

import litellm
import json
import logging
from sys import stdout

from Code.LiteLLM.utils import high_level_control_functions
from Code.LiteLLM.image_logger import ImageLogger
from Code.robocasa_env.main import Controller

cur_dir = Path(__file__).parent
logs_dir = cur_dir / "Logs"

environ["OPENAI_API_KEY"] = open(cur_dir / "API_KEY", mode="r").read()

parser = argparse.ArgumentParser(
                    prog='AutoBatchRobocasaLLM',
                    description='Executes the given batch with the current configuration')

parser.add_argument('-b', '--batch-name', type=str, default='')
parser.add_argument('-V', '--vision-enabled', action='store_true')
parser.add_argument('--vision-legacy', action='store_true')
parser.add_argument('-m', '--model', default='o4-mini')
parser.add_argument('-r', '--renderer', action='store_true')
parser.add_argument('-p', '--log-pictures', action='store_true')


args = parser.parse_args()

batch_name = args.batch_name  # leave empty to save in Logs directly
batch_path = logs_dir / batch_name
batch_path.mkdir(parents=True, exist_ok=True)
number_of_logs = sum(
    1 for f in batch_path.iterdir()
    if f.is_dir() and f.name.startswith("RobocasaLLM_")
)
log_path = batch_path / f"RobocasaLLM_{number_of_logs}"
log_path.mkdir(parents=True)

# file handler
file_handler = logging.FileHandler(log_path / 'RobocasaLLM.log', mode='w')

# console handler
console_handler = logging.StreamHandler(stdout)

# create logger
logger = logging.getLogger("RobocasaLLM")
logger.setLevel("INFO")
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False

vision_legacy = args.vision_legacy
vision_enabled = not vision_legacy and args.vision_enabled
# vision_send_picture_every_tool_call = vision_enabled and True
headless = not args.renderer

with open(log_path / "args.json", mode="w") as args_file:
    json.dump(vars(args), args_file)

controller = Controller(headless=headless)
controller.start()

image_logger = ImageLogger(controller, log_path)

# You may only use one function call per response and have to wait for it to finish that you can potentially react to
#  errors that arise during execution and are returned by the function. If multiple function calls are provided, only
#  the first one will be executed.
# Before attempting to call a function, send one message reasoning which step would be useful to achieve your goal.
#  Afterward, execute the next logical step.
system_prompt = {"role": "system", "content": "You may only use one function call per response and have to wait for "
                                              "it to finish that you can potentially react to errors that arise "
                                              "during execution and are returned by the function. If multiple "
                                              "function calls are provided, the execution will fail. "
                                              f"First,{' describe the image provided, then' if vision_legacy else ''} "
                                              "think of a plan on how to achieve the task and send a message "
                                              f"containing the {'image description and ' if vision_legacy else ''}"
                                              "plan. You may pause and think at any time."
                 }

# The microwave door is closed.
# You should regularly check if the object didn't fall down, as that may happen often.
user_prompt = {"role": "user", "content": [
        {
            "type": "text",
            "text": "You are a one-armed robot with a single gripper. Your objective is to thaw food in a microwave. "
                    "The object is called \"obj\" in the simulation, the microwave is called \"container\". "
                    "In the end, the food should be in the microwave, the microwave should be turned on "
                    "and you should be at least 25 cm away from the object."
                    f"{' The microwave door is closed.' if not vision_legacy and not vision_enabled else ''}"
        }
    ]
}
messages: List = [system_prompt, user_prompt]
if vision_legacy:
    image_logger.add_current_scene_to_message(messages[1])

logger.info(f"Using system prompt:\n{system_prompt['content']}\n")
logger.info(f"Using microwave prompt:\n{messages[1]['content'][0]['text']}")

available_functions, tools = high_level_control_functions(controller)
assert len(available_functions) == len(tools)

activate_tools = False
used_tool_calls = []
while not controller.check_successful() and len(messages) <= 12:  # fixed limit of ten messages + sys + user
    try:
        response = litellm.completion(
            model=args.model,
            messages=messages,
            tools=tools if activate_tools else None,
        )
    except Exception as e:
        logger.error(f"Execution failed and yielded following error:\n{e}")
        logger.info("ERROR")
        raise e
    # response.usage contains tokens
    logger.info(f"\nLLM Response:\n{response.choices[0].message.content}")
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # Note: the JSON response may not always be valid; be sure to handle errors
    messages.append(response_message)  # extend conversation with assistant's reply

    # Step 2: check if the model wanted to call a function
    if tool_calls:
        logger.info("LLM wants to execute tool calls")
        logger.info("\nTool calls:")
        for n, tool_call in enumerate(tool_calls[:]):  # create a copy of tool_calls
            name = tool_call.function.name
            f_args = json.loads(tool_call.function.arguments)
            color = "\033[36m"
            reset = "\033[0m"
            logger.info(f"{color + 'Will not be executed: ' if n > 0 else ''}"
                        f"{name}({', '.join([f'{arg}={val}' for arg, val in f_args.items()])}){reset if n > 0 else ''}")
            if n > 0:
                tool_calls.pop()  # pop one element for each element after the first one

        # Step 3: call the function

        # Step 4: send the info for each function call and function response to the model
        tool_call = tool_calls[0]
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        function_response = function_to_call(**function_args)
        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": str(function_response),
            }
        )  # extend conversation with function response

        used_tool_calls.append(tool_call)
    else:
        logger.info("LLM is reasoning")
        logger.info(f"\nLLM Reasoning:\n{response.choices[0].message.content}")
        activate_tools = True
    if vision_legacy or vision_enabled or args.log_pictures:
        image_logger.get_image()

if controller.check_successful():
    logger.info("Task accomplished successfully!")
    logger.info("SUCCESS")
else:
    logger.info("Task failed after ten messages...\n"
                "Current State:\n"
                f"Object inside the microwave: {controller.check_object_in_microwave()}\n"
                f"Microwave button was pressed: {controller.check_button_pressed()}\n"
                f"Gripper is at least 25cm away from the door: {controller.check_gripper_away_from_microwave()}\n")
    logger.info("Manually check if the procedure was correct:")
    for tool_call in used_tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        logger.info(f"{name}({', '.join([f'{arg}={val}' for arg, val in args.items()])})")
    logger.info("FAIL")
controller.stop()
