import os
import litellm
import json

from Code.LiteLLM.utils import all_functions
from Code.robocasa_env.main import Controller


os.environ["LITELLM_LOG"] = "DEBUG"  # 👈 print DEBUG LOGS
os.environ["OPENAI_API_KEY"] = open("API_KEY", mode="r").read()

controller = Controller()
controller.start()

system_prompt = {"role": "system", "content": "You may only use one function call per response and have to wait for "
                                              "it to finish that you can potentially react to errors that arise "
                                              "during execution and are returned by the function. If multiple "
                                              "function calls are provided, only the first one will be executed."}

# What's the current robot end effector position and rotation?
#  What's the position of the object 'obj'?
#  Also, can you close the gripper, then open it again?
#  Can you move to the coordinates that are ten centimetres above the end effector? If so, please proceed to do so.
#  Can you press the button of the microwave?
messages = [system_prompt, {"role": "user", "content": "Can you press the button of the microwave?"}]

available_functions, tools = all_functions(controller)
assert len(available_functions) == len(tools)

response = litellm.completion(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice='auto'
)
print("\nFirst LLM Response:\n", response)
response_message = response.choices[0].message
tool_calls = response_message.tool_calls

print("\nTool calls:")
for n, tool_call in enumerate(tool_calls):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    color = "\033[36m"
    print(f"{color+'Will not be executed: ' if n>0 else ''}"
          f"{name}({', '.join([f'{arg}={val}' for arg, val in args.items()])})\033[0m")

# Step 2: check if the model wanted to call a function
if tool_calls:
    # Step 3: call the function
    # Note: the JSON response may not always be valid; be sure to handle errors
    messages.append(response_message)  # extend conversation with assistant's reply

    # Step 4: send the info for each function call and function response to the model
    for tool_call in tool_calls:
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
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )  # get a new response from the model where it can see the function response
    print("\nSecond LLM response:\n", response)


controller.stop()
