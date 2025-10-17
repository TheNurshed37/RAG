#first request with gemini api

from google import genai
import os

client = genai.Client(api_key="AIzaSyC6KcojG7D2Uq_lHryo9c3v6wmuDtT9Rm0")

# response = client.models.generate_content(
#     model="gemini-2.5-flash",
#     contents="why do fireflies have to die so soon?"
# )

# print(response.text)

response = client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents="why do fireflies have to die so soon?"
)

for stream in response:
    print(stream.text)