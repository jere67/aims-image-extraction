import os
import base64
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

try:
    load_dotenv()
except Exception as e:
    print(f"Error loading .env file: {e}")
    exit()

try:
    client = AzureOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        api_version=os.environ['API_VERSION'],
        azure_endpoint=os.environ['OPENAI_API_BASE'],
        organization=os.environ.get('OPENAI_ORGANIZATION')
    )
except KeyError as e:
    print(f"FATAL: {e}. Please ensure your .env file is correct.")
    exit()

def _encode_image_to_base64(image_path: str):
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"ERROR: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"ERROR: Could not encode image {image_path}: {e}")
        return None

def classify_images_with_gpt(image_paths: list[str], model_name: str, system_prompt: str):
    """
    Classifies a batch of images using a specified GPT model and system prompt.

    Args:
        image_paths (list[str]): A list of local paths to the image files.
        model_name (str): The name of the GPT model to use (e.g., "gpt-4.1-nano").
        system_prompt (str): The detailed instructions for the model.

    Returns:
        dict: A dictionary containing the classification results (keyed by filename) or an error message.
    """
    user_content = [
        {
            "type": "text",
            "text": "Please classify the following images according to the system instructions. Return a JSON object where keys are the filenames and values are the classification results."
        }
    ]

    for image_path in image_paths:
        base64_image = _encode_image_to_base64(image_path)
        if not base64_image:
            print(f"WARNING: Failed to encode image at {image_path}. Skipping this image in batch.")
            continue
        
        filename = os.path.basename(image_path)
        user_content.append({
            "type": "text",
            "text": f"Image Filename: {filename}"
        })
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    if len(user_content) == 1: # Only the initial text instruction exists
        return {"error": "No valid images to classify in this batch."}

    try:
        api_params = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "response_format": {"type": "json_object"},
        }

        response = client.chat.completions.create(**api_params)
        response_content = response.choices[0].message.content

        result = json.loads(response_content)
        
        # Add usage stats if available
        if hasattr(response, 'usage') and response.usage:
            result['usage'] = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
        return result

    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON from model response", "raw_response": response_content}
    except Exception as e:
        error_message = f"An API error occurred: {e}"
        return {"error": error_message}