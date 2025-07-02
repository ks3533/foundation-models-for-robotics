import base64
import os
from io import BytesIO

from PIL import Image
import litellm

from Code.robocasa_env.main import Controller

os.environ["OPENAI_API_KEY"] = open("API_KEY", mode="r").read()

controller = Controller()
controller.start()

image = controller.get_vision_data()

buffered = BytesIO()
Image.fromarray(image).save(buffered, format="JPEG")
Image.fromarray(image).show()
b64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

messages = [{"role": "user", "content": [
                            {
                                "type": "text",
                                "text": "What’s in this image?"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}", "detail": "high"
                                }
                            }
                        ]}]


response = litellm.completion(
    model="gpt-4o-mini",
    messages=messages,
)
print("\nLLM Response:\n", response)
response_message = response.choices[0].message

controller.stop()
