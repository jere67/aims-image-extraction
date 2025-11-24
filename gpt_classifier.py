import os
import base64
import json
import pandas as pd
import time
from openai import AzureOpenAI
from dotenv import load_dotenv
import plotly.graph_objects as go

# --- Configuration ---
MODELS_TO_TEST = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5", "gpt-5-mini", "gpt-5-nano", "o4-mini"]
TRAINING_IMAGE_DIRECTORY = "training_data/"
LABELS_FILE = "plots.csv"
UNIFIED_LOG_FILE = "classification_results.json"
API_CALL_DELAY_SECONDS = 2

# Pricing data per 1 Million tokens
MODEL_PRICING = {
    "gpt-4o": {"input": 2.75, "output": 11.00},
    "gpt-4o-mini": {"input": 0.165, "output": 0.66},
    "gpt-4.1": {"input": 2.20, "output": 8.80},
    "gpt-4.1-mini": {"input": 0.44, "output": 1.76},
    "gpt-4.1-nano": {"input": 0.11, "output": 0.44},
    "o4-mini": {"input": 1.21, "output": 4.84},
    "gpt-5": {"input": 1.38, "output": 11.00},
    "gpt-5-mini": {"input": 0.28, "output": 2.20},
    "gpt-5-nano": {"input": 0.06, "output": 0.44},
}

# --- Load Environment Variables & Initialize Client ---
try:
    if not load_dotenv(): 
        print("Warning: .env file not found.")
except Exception as e:
    print(f"Error loading .env file: {e}")
    exit()

try:
    client = AzureOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        api_version=os.environ['API_VERSION'],
        azure_endpoint=os.environ['OPENAI_API_BASE'],
        organization=os.environ['OPENAI_ORGANIZATION']
    )
except KeyError as e:
    print(f"Error: Missing environment variable: {e}.")
    exit()


# --- Helper Functions ---
def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file: 
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError: 
        return None

def classify_image(base64_image, model_name):
    """Classifies an image and returns the full API response object."""
    if not base64_image: return None
    try:
        api_params = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a meticulous expert in technical document analysis, specializing in distinguishing between data visualizations and engineering schematics."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
                                Your task is to perform a highly accurate binary classification of the provided image to determine if it is a data plot.

                                Please adhere to the following steps:

                                1.  **Analyze Visual Evidence**: First, carefully examine the image to identify its core structure and purpose.
                                    *   For a **`Plot`**, look for the fundamental elements of graphical data representation. A plot uses a coordinate system (like X-Y axes, or even 3D axes) to show the relationship between two or more variables. Key features include:
                                        *   Axes (e.g., X, Y, Z) with tick marks or labels.
                                        *   Data represented visually as points, lines, curves, bars, surfaces, or other shapes.
                                        *   Informational text like a title, legend, or labels that describe the data.
                                        *   **Important:** If it uses a coordinate system to graphically illustrate a relationship or data, it is a `Plot`. Do not be misled by engineering symbols if the underlying structure is a graph.

                                    *   For **`Not a Plot`**, look for images whose primary purpose is not to graphically represent data within a coordinate system. Examples include:
                                        *   Photographs of real-world objects or scenes.
                                        *   Illustrations or artistic drawings.
                                        *   Diagrams that show structure, layout, or flow but not data on axes (e.g., a flowchart, a mind map, or a schematic showing the arrangement of parts like a reactor core layout).

                                2.  **Formulate Reasoning**: Based on your analysis, write a brief 'reasoning' statement. Explain whether the image contains the fundamental elements of a plot (axes, data representation) and why that leads to your conclusion. If it's not a plot, explain what kind of image it is instead.

                                3.  **Provide Classification**: Finally, classify the image into one of the two categories (and only the two) below.

                                The categories are:
                                *   `Plot`
                                *   `Not a Plot`

                                Your output MUST be a valid JSON object with exactly two keys: "reasoning" and "classification". The value for "classification" must be one of the two exact strings listed above.
                            """
                        },
                        {
                            "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            # max_completion_tokens=300, <-- could put limit on this, but restrictive for gpt-5-nano
            # temperature=0.0, <-- gpt-5-nano does not support
            "response_format": {"type": "json_object"}
        }

        if model_name in ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]:
            api_params["temperature"] = 0.0

        response = client.chat.completions.create(**api_params)
        return response
    except Exception as e:
        if "The model" in str(e) and "does not exist" in str(e): 
            print(f"  - Error: Model '{model_name}' not found.")
        else: 
            print(f"  - An error occurred during the API call: {e}")
        return None

def generate_result_charts(results):
    """Generates and saves five separate Plotly bar charts for each key metric."""
    print("\nGenerating result charts...")
    models = list(results.keys())
    
    # Data Extraction
    accuracies = [data.get('accuracy', 0) for data in results.values()]
    costs = [data.get('total_cost', 0) for data in results.values()]
    times = [data.get('classification_time_minutes', 0) for data in results.values()]
    input_tokens = [data.get('total_input_tokens', 0) for data in results.values()]
    output_tokens = [data.get('total_output_tokens', 0) for data in results.values()]

    # Charting Function
    def create_chart(values, title, y_axis_title, filename, text_format):
        fig = go.Figure([go.Bar(x=models, y=values, text=[text_format.format(v) for v in values], textposition='auto')])
        fig.update_layout(title=title, xaxis_title="Model Name", yaxis_title=y_axis_title)
        try:
            fig.write_image(filename)
            print(f"- Chart saved successfully to '{filename}'")
        except Exception as e:
            print(f"- Error saving chart '{filename}'. Ensure 'kaleido' is installed: {e}")

    # Generate all 5 charts
    create_chart(accuracies, "Model Comparison: Accuracy", "Accuracy", "model_accuracy.png", "{:.2%}")
    create_chart(costs, "Model Comparison: Total Cost", "Total Cost ($)", "model_total_cost.png", "${:.4f}")
    create_chart(times, "Model Comparison: Classification Time", "Time (minutes)", "model_classification_time.png", "{:.2f} min")
    create_chart(input_tokens, "Model Comparison: Total Input Tokens", "Input Tokens", "model_input_tokens.png", "{:,}")
    create_chart(output_tokens, "Model Comparison: Total Output Tokens", "Output Tokens", "model_output_tokens.png", "{:,}")

def main():
    try:
        labels_df = pd.read_csv(LABELS_FILE)
        print(f"Successfully loaded labels for {len(labels_df)} images.")
    except FileNotFoundError:
        print(f"Error: The labels file '{LABELS_FILE}' was not found.")
        return

    all_results = {}
    if os.path.exists(UNIFIED_LOG_FILE):
        with open(UNIFIED_LOG_FILE, 'r') as f: 
            all_results = json.load(f)
        print(f"Loaded existing results from '{UNIFIED_LOG_FILE}'.")

    for model_name in MODELS_TO_TEST:
        print("\n" + "="*50 + f"\nEVALUATING MODEL: {model_name}\n" + "="*50)

        if model_name not in all_results:
            all_results[model_name] = {"classifications": [], "total_input_tokens": 0, "total_output_tokens": 0, "total_cost": 0.0}
        
        processed_files = {entry['filename'] for entry in all_results[model_name]['classifications']}
        
        start_time = time.time()

        for index, row in labels_df.iterrows():
            filename = row['image']
            if filename in processed_files: 
                continue

            print(f"  - Processing ({index + 1}/{len(labels_df)}): {filename}")
            image_path = os.path.join(TRAINING_IMAGE_DIRECTORY, filename)
            base64_image = encode_image_to_base64(image_path)
            if not base64_image: 
                continue

            response = classify_image(base64_image, model_name)
            expected_classification = "Plot" if row['isPlot'] == 1 else "Not a Plot"
            log_entry = {
                "filename": filename,
                "expected_classification": expected_classification
            }

            if response:
                prompt_tokens, completion_tokens = response.usage.prompt_tokens, response.usage.completion_tokens
                all_results[model_name]["total_input_tokens"] += prompt_tokens
                all_results[model_name]["total_output_tokens"] += completion_tokens
                
                pricing = MODEL_PRICING.get(model_name, {"input": 0, "output": 0})
                cost = ((prompt_tokens / 1e6) * pricing["input"]) + ((completion_tokens / 1e6) * pricing["output"])
                all_results[model_name]["total_cost"] += cost
                log_entry["tokens"] = prompt_tokens

                try:
                    result = json.loads(response.choices[0].message.content)
                    log_entry.update({"predicted_classification": result.get("classification"), "reasoning": result.get("reasoning", "No reasoning.")})
                    print(f"    - Classified as: {log_entry['predicted_classification']} | Tokens: {prompt_tokens}")
                except (json.JSONDecodeError, AttributeError):
                    log_entry.update({"error": "Failed to parse API response", "raw_response": response.choices[0].message.content})
            else:
                log_entry.update({"error": "API call failed", "tokens": 0})
            
            all_results[model_name]['classifications'].append(log_entry)
            
            # Continuously save progress
            with open(UNIFIED_LOG_FILE, 'w') as f: 
                json.dump(all_results, f, indent=4)
            time.sleep(API_CALL_DELAY_SECONDS)

        # --- Post-run calculations and final save for the model ---
        end_time = time.time()
        duration_minutes = (end_time - start_time) / 60
        
        correct = sum(1 for e in all_results[model_name]['classifications'] if e.get('predicted_classification') == e.get('expected_classification'))
        total = len(all_results[model_name]['classifications'])
        accuracy = (correct / total) if total > 0 else 0
        
        all_results[model_name]['accuracy'] = accuracy
        all_results[model_name]['classification_time_minutes'] = duration_minutes

        print(f"\nFinished evaluation for {model_name}.")
        print(f"  - Accuracy: {accuracy:.2%}")
        print(f"  - Time Taken: {duration_minutes:.2f} minutes")

        # Final save for the model with summary stats included
        with open(UNIFIED_LOG_FILE, 'w') as f: 
            json.dump(all_results, f, indent=4)

    print("\n\n" + "="*50 + "\nFINAL SUMMARY\n" + "="*50)
    for model_name, data in all_results.items():
        print(f"- {model_name}: Accuracy: {data.get('accuracy', 0):.2%}, Cost: ${data.get('total_cost', 0):.4f}, Time: {data.get('classification_time_minutes', 0):.2f} min")
    
    if all_results:
        generate_result_charts(all_results)

if __name__ == "__main__":
    main()