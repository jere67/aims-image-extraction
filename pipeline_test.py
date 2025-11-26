import os
import pandas as pd
import time
import json
import plotly.graph_objects as go
from tqdm import tqdm
from gpt_classifiers import gpt_4_1_nano_classifier, gpt_5_nano_classifier
from gpt_classifiers.config import BINARY_CLASSIFICATION_PROMPT

# Configuration
TEST_CSV = 'dataset/test.csv'
IMAGES_DIR = 'dataset/test'
OUTPUT_FILE = 'pipeline_test_results.json'

# Pricing data per 1 Million tokens
MODEL_PRICING = {
    "gpt-4.1-nano": {"input": 0.11, "output": 0.44},
    "gpt-5-nano": {"input": 0.06, "output": 0.44},
}

API_CALL_DELAY_SECONDS = 2

def calculate_cost(model_name, prompt_tokens, completion_tokens):
    pricing = MODEL_PRICING.get(model_name, {"input": 0, "output": 0})
    cost = ((prompt_tokens / 1e6) * pricing["input"]) + ((completion_tokens / 1e6) * pricing["output"])
    return cost

def parse_classification(res):
    """
    Parses the result from the binary classification prompt.
    Returns 'Plot', 'Not a Plot', or None if error.
    """
    if not res or res.get('error'):
        return None
    
    classification = res.get('classification')
    if classification in ['Plot', 'Not a Plot']:
        return classification
    return None

def main():
    print("Starting Pipeline Test (Binary Classification)...")
    
    # Load test data
    try:
        df = pd.read_csv(TEST_CSV)
    except Exception as e:
        print(f"FATAL: Could not read {TEST_CSV}: {e}")
        return

    results = {
        "total_images": len(df),
        "correct_classifications": 0,
        "total_cost": 0.0,
        "total_time_seconds": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "details": []
    }
    
    start_time_all = time.time()

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Testing Pipeline"):
        filename = row['filename']
        label = row['label'] # 1 = Plot, 0 = Not Plot
        
        # Ground Truth Label
        expected_classification = "Plot" if label == 1 else "Not a Plot"
        
        image_path = os.path.join(IMAGES_DIR, filename)
        if not os.path.exists(image_path):
            print(f"WARNING: Image {filename} not found.")
            continue

        # --- Run Both Models ---
        # We run both to apply the tie-breaking logic:
        # If M1 != M2 -> "Not a Plot"
        # If M1 == M2 == "Plot" -> "Plot"
        # If M1 == M2 == "Not Plot" -> "Not a Plot"
        
        # Model 1: GPT-4.1-Nano
        t0 = time.time()
        res1 = gpt_4_1_nano_classifier.classify_image(image_path, system_prompt=BINARY_CLASSIFICATION_PROMPT)
        t1 = time.time()

        time.sleep(API_CALL_DELAY_SECONDS)
        
        # Model 2: GPT-5-Nano
        t2 = time.time()
        res2 = gpt_5_nano_classifier.classify_image(image_path, system_prompt=BINARY_CLASSIFICATION_PROMPT)
        t3 = time.time()

        time.sleep(API_CALL_DELAY_SECONDS)
        
        # Metrics Calculation
        stage1_time = t1 - t0
        stage2_time = t3 - t2
        
        stage1_cost = 0.0
        stage1_input = 0
        stage1_output = 0
        if res1 and 'usage' in res1:
            stage1_input = res1['usage']['prompt_tokens']
            stage1_output = res1['usage']['completion_tokens']
            stage1_cost = calculate_cost("gpt-4.1-nano", stage1_input, stage1_output)

        stage2_cost = 0.0
        stage2_input = 0
        stage2_output = 0
        if res2 and 'usage' in res2:
            stage2_input = res2['usage']['prompt_tokens']
            stage2_output = res2['usage']['completion_tokens']
            stage2_cost = calculate_cost("gpt-5-nano", stage2_input, stage2_output)
            
        # Parse Results
        cls1 = parse_classification(res1)
        cls2 = parse_classification(res2)
        
        # Tie-Breaking Logic
        final_classification = "Not a Plot" # Default
        
        if cls1 == "Plot" and cls2 == "Plot":
            final_classification = "Plot"
        elif cls1 == "Not a Plot" and cls2 == "Not a Plot":
            final_classification = "Not a Plot"
        else:
            # Disagreement or Error -> "Not a Plot"
            # User: "gpt4.1-nano says it's a plot but gpt5-nano says it's not... break the tie by saying that it is not a plot."
            final_classification = "Not a Plot"
            
        # Check Correctness
        is_correct = (final_classification == expected_classification)
        
        # Update Aggregates
        total_image_cost = stage1_cost + stage2_cost
        total_image_time = stage1_time + stage2_time
        total_image_input = stage1_input + stage2_input
        total_image_output = stage1_output + stage2_output
        
        if is_correct:
            results["correct_classifications"] += 1
        results["total_cost"] += total_image_cost
        results["total_time_seconds"] += total_image_time
        results["total_input_tokens"] += total_image_input
        results["total_output_tokens"] += total_image_output
        
        results["details"].append({
            "filename": filename,
            "label": label,
            "expected": expected_classification,
            "model1_pred": cls1,
            "model2_pred": cls2,
            "final_pred": final_classification,
            "is_correct": is_correct,
            "cost": total_image_cost,
            "time": total_image_time
        })

    # Calculate Final Stats
    accuracy = results["correct_classifications"] / results["total_images"] if results["total_images"] > 0 else 0
    results["accuracy"] = accuracy
    
    print("\n" + "="*50)
    print("PIPELINE TEST RESULTS (BINARY)")
    print("="*50)
    print(f"Total Images: {results['total_images']}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Total Cost: ${results['total_cost']:.4f}")
    print(f"Total Time: {results['total_time_seconds']/60:.2f} minutes")
    print(f"Total Input Tokens: {results['total_input_tokens']}")
    print(f"Total Output Tokens: {results['total_output_tokens']}")
    
    # Save Results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
