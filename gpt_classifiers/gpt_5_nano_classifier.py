from .base_classifier import classify_images_with_gpt
from .config import STAGE_2_PROMPT

MODEL_NAME = "gpt-5-nano"

def classify_images(image_paths: list[str], system_prompt: str = STAGE_2_PROMPT):
    """
    Passes a batch of images to the GPT-5-Nano model for classification.
    """
    return classify_images_with_gpt(image_paths, MODEL_NAME, system_prompt)

