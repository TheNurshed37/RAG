#multimodel API with different types of file 

from google import genai
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

uploaded_file = client.files.upload(file="1.jpg") #.jpg, .mp3, .pdf etc

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=["Describe the uploaded file for me", uploaded_file]
)
print(response.text)