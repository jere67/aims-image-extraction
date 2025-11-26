from .base_classifier import classify_image_with_gpt
from .config import STAGE_1_PROMPT

MODEL_NAME = "gpt-4.1-nano"

def classify_image(image_path: str, system_prompt: str = STAGE_1_PROMPT):
    """
    Passes the image to the GPT-4.1-Nano model for classification.
    """
    return classify_image_with_gpt(image_path, MODEL_NAME, system_prompt)

