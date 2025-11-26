from .base_classifier import classify_image_with_gpt
from .config import STAGE_2_PROMPT

MODEL_NAME = "gpt-5-nano"

def classify_image(image_path: str, system_prompt: str = STAGE_2_PROMPT):
    """
    Passes the image to the GPT-5-Nano model for classification.
    """
    return classify_image_with_gpt(image_path, MODEL_NAME, system_prompt)

