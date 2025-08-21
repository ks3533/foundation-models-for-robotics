import litellm
from typing import Union

from PIL.Image import Image

from Code.LiteLLM.image_logger import ImageLogger


def get_scene_description(image_logger: ImageLogger, model: str):
    system_prompt = {"role": "system",
                     "content": "You are a chatbot that is meant to give scene descriptions with a given "
                                "image. You should describe the image provided in "
                                "detail, describing all objects that can be seen, where they are, how "
                                "the relations between the objects are and in which state they are "
                                "currently (e.g. open, closed). Do not focus on asthetics too much, take a more "
                                "functional and pragmatic approach instead. The state of containers is very important."
                     }
    user_prompt = {"role": "user", "content": [
        {
            "type": "text",
            "text": "Please describe the attached picture of the scene."
        }
    ]
                   }
    messages = [system_prompt, user_prompt]

    image_logger.add_current_scene_to_message(user_prompt)

    response = litellm.completion(
        model=model,
        messages=messages,
        tools=None,
    )

    return response.choices[0].message.content


def get_scene_description_json(image_logger: ImageLogger, model: str):
    system_prompt = {"role": "system",
                     "content": "You are a chatbot that is meant to give scene descriptions in JSON with a given "
                                "image. You should list all objects that can be seen, where they are, how "
                                "the relations between the objects are and in which state they are "
                                "currently (e.g. open, closed). The state of containers is very important, as well as "
                                "the relations."
                     }
    user_prompt = {"role": "user", "content": [
        {
            "type": "text",
            "text": "Please describe the attached picture of the scene."
        }
    ]
                   }
    messages = [system_prompt, user_prompt]

    image_logger.add_current_scene_to_message(user_prompt)

    response = litellm.completion(
        model=model,
        messages=messages,
        tools=None,
    )

    return response.choices[0].message.content


def get_scene_diff(image_logger: ImageLogger, previous_scene: Union[Image, str], model: str, mode="auto"):
    if mode == "auto":
        if isinstance(previous_scene, Image):
            mode = "image"
        elif isinstance(previous_scene, str):
            mode = "json"
        else:
            raise TypeError(
                f"previous_scene must be either Image or a json string, not {previous_scene.__class__.__name__}"
            )

    system_prompt = {"role": "system",
                     "content": "You are a chatbot that is meant to give scene difference descriptions in JSON with "
                                f"{'two given images' if mode == 'image' else 'a JSON scene description and a given image'}."
                                f"You should list all object differences that can be seen, how they moved, how "
                                "the relations between the objects changed and in which state they are now "
                                "(e.g. were opened, closed). The state change of containers is very important, as well "
                                "as the new relations. The first picture or the JSON scene description, depending on "
                                "which one is present, shall be treated as the original scene to create the difference from."
                     }
    user_prompt = {"role": "user", "content": [
        {
            "type": "text",
            "text": "Please create a scene difference description given th"
                    f"{'ese two images.' if mode == 'image' else 'is JSON scene description and this given image:'}"
        }
    ]
                   }

    if mode == "json":
        user_prompt["content"][0]["text"] += f"\n{previous_scene}"
    elif mode == "image":
        ImageLogger.add_image_to_message(user_prompt, previous_scene)
    image_logger.add_current_scene_to_message(user_prompt)

    messages = [system_prompt, user_prompt]

    response = litellm.completion(
        model=model,
        messages=messages,
        tools=None,
    )

    return response.choices[0].message.content
