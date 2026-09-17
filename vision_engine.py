import os
import cv2
from google import genai
from google.genai import types
from PIL import Image

def get_genai_client():
    """Initializes the Gemini API client using the environment variable."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise ValueError("GEMINI_API_KEY is missing or invalid in the .env file.")
    return genai.Client(api_key=api_key)

def analyze_image(client, cv2_frame):
    """
    Sends the captured frame to Gemini to identify clothing/patterns and food items.
    Uses types.Part for safe, SDK-compliant multimodal input.
    """
    # Convert OpenCV BGR frame to RGB and then to PIL Image
    color_converted = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_converted)

    prompt = (
        "You are an assistive AI for people who need help identifying objects. "
        "Look at this image. If there is clothing, describe its type, color, and pattern clearly. "
        "If there is food, identify what it is. "
        "Keep the description concise, natural, and helpful for a daily activity assistant. "
        "If neither is prominent, briefly describe what the main subject of the image is."
    )

    # Use types.Part.from_image() for proper SDK-compliant multimodal input
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents=[
            types.Part.from_image(pil_image),
            types.Part.from_text(prompt),
        ]
    )

    return response.text
