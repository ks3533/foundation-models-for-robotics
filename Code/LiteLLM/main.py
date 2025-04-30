import os

import instructor
from litellm import completion
from pydantic import BaseModel

os.environ["LITELLM_LOG"] = "DEBUG"  # 👈 print DEBUG LOGS

client = instructor.from_litellm(completion)

# import dotenv
# dotenv.load_dotenv()


class UserDetail(BaseModel):
    name: str
    age: int


user = client.chat.completions.create(
    model="ollama/llama3.2:3b",
    api_base="http://localhost:11434",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract Jason is 25 years old"},
    ],
)

assert isinstance(user, UserDetail)
assert user.name == "Jason"
assert user.age == 25

print(f"user: {user}")
