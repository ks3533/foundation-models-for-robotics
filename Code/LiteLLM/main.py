import os
import litellm
import json
# import logging

from Code.LiteLLM.utils import high_level_control_functions
from Code.robocasa_env.main import Controller

# os.environ["LITELLM_LOG"] = "DEBUG"  # 👈 print DEBUG LOGS
os.environ["OPENAI_API_KEY"] = open("API_KEY", mode="r").read()

# logger = logging.getLogger("LiteLLM")
# number_of_logs = sum(1 for f in os.listdir('Logs'))  # if os.path.isfile(os.path.join('Logs', f))
# logging.basicConfig(level=logging.INFO, filename=f'Logs/LiteLLM_{number_of_logs}.log')

controller = Controller()
controller.start()

# You may only use one function call per response and have to wait for it to finish that you can potentially react to
#  errors that arise during execution and are returned by the function. If multiple function calls are provided, only
#  the first one will be executed.
# Before attempting to call a function, send one message reasoning which step would be useful to achieve your goal.
#  Afterward, execute the next logical step.
system_prompt = {"role": "system", "content": "You may only use one function call per response and have to wait for "
                                              "it to finish that you can potentially react to errors that arise "
                                              "during execution and are returned by the function. If multiple "
                                              "function calls are provided, only the first one will be executed. "
                                              "Before you start, think of a plan on how to achieve the task and send a "
                                              "message containing the plan. You may pause and think at any time."}

# What's the current robot end effector position and rotation?
# What's the position of the object 'obj'?
# Also, can you close the gripper, then open it again?
# Can you move to the coordinates that are ten centimetres above the end effector? If so, please proceed to do so.
# Can you press the button of the microwave?
messages = [system_prompt, {"role": "user", "content":
            "Your objective is to thaw food in a microwave. The microwave door is closed. "
                            "The object is called \"obj\" in the simulation, the microwave is called \"container\". "
                            "In the end, the food should be in the microwave, the microwave should be turned on "
                            "and you should be at least 25 cm away from the object. You should regularly check if the "
                            "object didn't fall down, as that may happen often."}]

available_functions, tools = high_level_control_functions(controller)
assert len(available_functions) == len(tools)

while not controller.check_successful():
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice='auto'
    )
    print("\nLLM Response:\n", response)
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # Note: the JSON response may not always be valid; be sure to handle errors
    messages.append(response_message)  # extend conversation with assistant's reply

    # Step 2: check if the model wanted to call a function
    if tool_calls:
        print("LLM wants to execute tool calls")
        print("\nTool calls:")
        for n, tool_call in enumerate(tool_calls):
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            color = "\033[36m"
            print(f"{color + 'Will not be executed: ' if n > 0 else ''}"
                  f"{name}({', '.join([f'{arg}={val}' for arg, val in args.items()])})\033[0m")

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
        print("LLM is reasoning")
        print("\nLLM Reasoning:\n", response.choices[0].message.content)


controller.stop()
