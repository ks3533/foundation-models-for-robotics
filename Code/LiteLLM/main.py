from os import listdir, environ, mkdir
import os
from typing import List

import litellm
import json
import logging
from sys import stdout

from Code.LiteLLM.utils import high_level_control_functions, get_image, add_image_to_message
from Code.robocasa_env.main import Controller

environ["OPENAI_API_KEY"] = open("API_KEY", mode="r").read()

batch_name = ''  # leave empty to save in Logs directly
number_of_logs = sum(1 for f in listdir('Logs'))  # if os.path.isfile(os.path.join('Logs', f))
number_of_images = 0
log_path = os.path.join("Logs", batch_name, f"RobocasaLLM_{number_of_logs}")
mkdir(log_path)

# file handler
file_handler = logging.FileHandler(os.path.join(log_path, 'RobocasaLLM.log'), mode='w')

# console handler
console_handler = logging.StreamHandler(stdout)

# create logger
logger = logging.getLogger("RobocasaLLM")
logger.setLevel("INFO")
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False

vision_enabled = True
# vision_send_picture_every_tool_call = vision_enabled and True
headless = True

controller = Controller(headless=headless)
controller.start()

# You may only use one function call per response and have to wait for it to finish that you can potentially react to
#  errors that arise during execution and are returned by the function. If multiple function calls are provided, only
#  the first one will be executed.
# Before attempting to call a function, send one message reasoning which step would be useful to achieve your goal.
#  Afterward, execute the next logical step.
system_prompt = {"role": "system", "content": "You may only use one function call per response and have to wait for "
                                              "it to finish that you can potentially react to errors that arise "
                                              "during execution and are returned by the function. If multiple "
                                              "function calls are provided, the execution will fail. "
                                              f"First,{' describe the image provided, then' if vision_enabled else ''} "
                                              "think of a plan on how to achieve the task and send a message "
                                              f"containing the {'image description and ' if vision_enabled else ''}"
                                              "plan. You may pause and think at any time."
                 }

# What's the current robot end effector position and rotation?
# What's the position of the object 'obj'?
# Also, can you close the gripper, then open it again?
# Can you move to the coordinates that are ten centimetres above the end effector? If so, please proceed to do so.
# Can you press the button of the microwave?
# The microwave door is closed.
# You should regularly check if the object didn't fall down, as that may happen often.
messages: List = [system_prompt, {"role": "user", "content": [
    {
        "type": "text",
        "text": "You are a one-armed robot with a single gripper. Your objective is to thaw food in a microwave. "
                "The object is called \"obj\" in the simulation, the microwave is called \"container\". "
                "In the end, the food should be in the microwave, the microwave should be turned on "
                "and you should be at least 25 cm away from the object."
    }
]}]
if vision_enabled:
    image = get_image(controller)
    add_image_to_message(messages[1], image)
    image.save(os.path.join(log_path, f"Image_{number_of_images}.jpg"), format="JPEG")
    number_of_images += 1

logger.info(f"Using system prompt:\n{system_prompt['content']}\n")
logger.info(f"Using microwave prompt:\n{messages[1]['content'][0]['text']}")

available_functions, tools = high_level_control_functions(controller)
assert len(available_functions) == len(tools)

activate_tools = False
while not controller.check_successful():
    try:
        response = litellm.completion(
            model="gpt-4o",
            messages=messages,
            tools=tools if activate_tools else None,
        )
    except Exception as e:
        logger.error(e)
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
        for n, tool_call in enumerate(tool_calls):
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            color = "\033[36m"
            reset = "\033[0m"
            logger.info(f"{color + 'Will not be executed: ' if n > 0 else ''}"
                        f"{name}({', '.join([f'{arg}={val}' for arg, val in args.items()])}){reset if n > 0 else ''}")

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
    else:
        logger.info("LLM is reasoning")
        logger.info(f"\nLLM Reasoning:\n{response.choices[0].message.content}")
        activate_tools = True


controller.stop()
logger.info("Task accomplished successfully!")
