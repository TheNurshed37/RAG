import base64
from openai import OpenAI
from PIL import Image
from io import BytesIO

client = OpenAI(
    api_key="GEMINI_API_KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

response = client.images.generate(
    model="imagen-3.0-generate-002",
    prompt="A solitary, luminous red plastic bag sits on a dark, rocky beach at blue hour. The bag emits a soft, internal red glow, subtly illuminating its immediate surface and casting a gentle reflection in a small puddle below. The surrounding landscape of mountains and ocean is shrouded in the deep, dramatic blues of twilight, with a hint of mist adding to the atmospheric solitude.",
    response_format='b64_json',
    n=1,
)

for image_data in response.data:
  image = Image.open(BytesIO(base64.b64decode(image_data.b64_json)))
  image.show()