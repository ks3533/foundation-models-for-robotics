from os import environ

from Code.LiteLLM.image_logger import ImageLogger
from Code.LiteLLM.scene_description import get_scene_description
from Code.robocasa_env.main import Controller

environ["OPENAI_API_KEY"] = open("API_KEY", mode="r").read()

controller = Controller()

image_logger = ImageLogger(controller, ".")

print(get_scene_description(image_logger, "o4-mini"))
