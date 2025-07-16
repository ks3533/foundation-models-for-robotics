import base64
from io import BytesIO
from pathlib import Path
from typing import Union

from PIL import Image

from Code.robocasa_env.main import Controller


class ImageLogger:
    def __init__(self, controller: Controller, log_path: Union[str, Path]):
        self.controller = controller
        self.log_path = Path(log_path)
        self.number_of_images = 0

    def get_image(self):
        image = self.controller.get_vision_data()
        self.save_image(image)
        return Image.fromarray(image)

    @staticmethod
    def to_base64_image(image: Image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_base64_image(self):
        image = self.get_image()
        return ImageLogger.to_base64_image(image)

    @staticmethod
    def add_image_to_message(message: dict, image: Image):
        b64_image = ImageLogger.to_base64_image(image)
        message["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64_image}", "detail": "auto"
            }
        })

    def add_current_scene_to_message(self, message: dict):
        ImageLogger.add_image_to_message(message, self.get_base64_image())

    def save_image(self, image: Image):
        image.save(self.log_path / f"Image_{self.number_of_images}.jpg", format="JPEG")
        self.number_of_images += 1

