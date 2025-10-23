import requests

def gemini_describe_image(base64_image: str, api_key: str, model="gemini-2.5-flash") -> str:
    """
    Send an image to Gemini API and get a single-paragraph description.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [{
            "parts": [
                {"text": "Describe this image in sentences in a single paragraph."},
                {"inline_data": {
                    "mime_type": "image/png",
                    "data": base64_image
                }}
            ]
        }]
    }

    res = requests.post(url, headers=headers, json=payload, timeout=60)
    res.raise_for_status()
    data = res.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]
