import litellm

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

    image_logger.add_current_scene_to_message(messages[1])

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

    image_logger.add_current_scene_to_message(messages[1])

    response = litellm.completion(
        model=model,
        messages=messages,
        tools=None,
    )

    return response.choices[0].message.content
