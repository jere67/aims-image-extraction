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

def classify_image_with_gpt(image_path: str, model_name: str, system_prompt: str):
    """
    Classifies a single image using a specified GPT model and system prompt.

    Args:
        image_path (str): The local path to the image file.
        model_name (str): The name of the GPT model to use (e.g., "gpt-4.1-nano").
        system_prompt (str): The detailed instructions for the model.

    Returns:
        dict: A dictionary containing the classification result or an error message.
    """
    base64_image = _encode_image_to_base64(image_path)
    if not base64_image:
        return {"error": f"Failed to encode image at {image_path}"}

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
                    "content": [
                        {
                            "type": "text",
                            "text": "Please classify the following image according to the system instructions."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
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