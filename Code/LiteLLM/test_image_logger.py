from time import sleep

import numpy as np
from PIL import Image

from Code.LiteLLM.image_logger import ImageLogger
from Code.robocasa_env.main import Controller

controller = Controller()
controller.start()

image_logger = ImageLogger(controller, ".")

image_logger.get_image().show()
sleep(5)
image_logger.get_image().show()
sleep(5)
image_logger.get_image().show()
sleep(5)
image_logger.get_image().show()
