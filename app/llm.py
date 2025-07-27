from litellm import completion
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "openrouter/google/gemini-2.5-flash")

response = completion(
  model=MODEL_NAME,
  messages = [{ "content": "Hello, how are you?","role": "user"}],
)