import os
import litellm
import json
from Code.robocasa_env.main import Controller


os.environ["LITELLM_LOG"] = "DEBUG"  # 👈 print DEBUG LOGS
os.environ["OPENAI_API_KEY"] = open("API_KEY", mode="r").read()

controller = Controller()
controller.start()


# Step 1: send the conversation and available functions to the model

# What's the current robot end effector position and rotation?
#  What's the position of the object 'obj'?
#  Also, can you close the gripper, then open it again?
messages = [{"role": "user", "content": "Can you move to the coordinates that are ten centimetres above the end effector? If so, please procede to do so."}]
tools = json.loads(open("robot_api.json", "r").read())
response = litellm.completion(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto",  # auto is default, but we'll be explicit
)
print("\nFirst LLM Response:\n", response)
response_message = response.choices[0].message
tool_calls = response_message.tool_calls

print("\nLength of tool calls", len(tool_calls))

# Step 2: check if the model wanted to call a function
if tool_calls:
    # Step 3: call the function
    # Note: the JSON response may not always be valid; be sure to handle errors
    available_functions = {
        "get_eef_pos": controller.get_eef_pos,
        "get_eef_rot": controller.get_eef_rot,
        "resolve_object_from_name": controller.resolve_object_from_name,
        "open_gripper": controller.open_gripper,
        "close_gripper": controller.close_gripper,
        "move_abs": controller.move_abs
    }  # only one function in this example, but you can have multiple
    assert len(available_functions) == len(tools)
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
    second_response = litellm.completion(
        model="gpt-4o-mini",
        messages=messages,
    )  # get a new response from the model where it can see the function response
    print("\nSecond LLM response:\n", second_response)


controller.stop()
