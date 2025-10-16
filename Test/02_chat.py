#setup chat

from google import genai
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

chat = client.chats.create(model="gemini-2.5-flash")

while True:
    msg = input("> ")
    if msg == "exit":
        break

    response = chat.send_message(msg)
    print(response.text)
