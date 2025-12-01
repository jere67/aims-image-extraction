import os
import sys
import pandas as pd
import shutil
import time
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# --- Local Module Imports ---
import parser
from gpt_classifiers import gpt_4_1_nano_classifier, gpt_5_nano_classifier

# --- Configuration ---
# Stage 1 (Parser)
PDF_DIR = "pdf_files/"
PARSER_OUTPUT_DIR = "saved/"
PARSER_METADATA_FILE = "metadata.csv"

# Stage 2 (GPT-4.1-Nano)
STAGE2_OUTPUT_DIR = "gpt_4_1_nano_approved/"
STAGE2_METADATA_FILE = "metadata_stage2.csv"

# Stage 3 (GPT-5-Nano)
FINAL_OUTPUT_DIR = "final_output/"
FINAL_METADATA_FILE = "final_metadata.csv"

API_CALL_DELAY_SECONDS = 2
BATCH_SIZE = 5

# Pricing (per 1M tokens)
GPT_4_1_NANO_INPUT_PRICE = 0.10
GPT_4_1_NANO_OUTPUT_PRICE = 0.40
GPT_5_NANO_INPUT_PRICE = 0.05
GPT_5_NANO_OUTPUT_PRICE = 0.40

def run_stage1_parser():
    """Checks for PDFs and runs the parser main function if they exist."""
    print("\n" + "="*60)
    print("Stage 1: PDF Parsing and Initial Filtering")
    print("="*60)
    
    start_time = time.time()
    
    if not os.path.exists(PDF_DIR) or not any(f.lower().endswith('.pdf') for f in os.listdir(PDF_DIR)):
        print("INFO: No new PDFs found in 'pdf_files/'. Skipping parser run.")
        if not os.path.exists(PARSER_OUTPUT_DIR):
             print(f"FATAL: No PDFs to process and the parser output directory ('{PARSER_OUTPUT_DIR}') does not exist. Cannot continue.")
             sys.exit(1)
        return

    print("INFO: New PDFs found. Running the local parser and classifier...")
    try:
        images_input, images_approved = parser.main()
        print("SUCCESS: parser.py executed successfully.")
    except Exception as e:
        print(f"FATAL: An error occurred while running parser.py: {e}")
        sys.exit(1)
    
    duration = time.time() - start_time
    
    images_approved = 0
    if os.path.exists(PARSER_METADATA_FILE):
        df = pd.read_csv(PARSER_METADATA_FILE)
        images_approved = len(df)
    
    print(f"\n✓ Stage 1 complete in {duration:.1f}s")
    print(f"  Images approved: {images_approved}")
    
    return {
        'duration': duration,
        'images_approved': images_approved,
        'images_input': images_input,
        'skipped': False
    }

def run_stage2_gpt_4_1_nano():
    """Runs GPT-4.1-Nano classification on images from the parser stage."""
    print("\n" + "="*60)
    print("Stage 2: GPT-4.1-Nano Classification")
    print("="*60)
    
    start_time = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    api_calls = 0

    if not os.path.exists(PARSER_OUTPUT_DIR) or not os.path.exists(PARSER_METADATA_FILE):
        print(f"FATAL: Cannot start Stage 2. Missing parser output directory ('{PARSER_OUTPUT_DIR}') or metadata file ('{PARSER_METADATA_FILE}').")
        print("Please run Stage 1 first.")
        sys.exit(1)
    
    try:
        df = pd.read_csv(
            PARSER_METADATA_FILE,
            dtype={'file_name': str, 'caption': str},
            on_bad_lines='warn'
        )
    except Exception as e:
        print(f"FATAL: Could not read {PARSER_METADATA_FILE}. Error: {e}")
        sys.exit(1)
        
    df_filtered = df[
        df['file_name'].notna() &
        df['caption'].notna() & (df['caption'] != 'N/A')
    ].copy()
    
    if df_filtered.empty:
        print("INFO: No images with valid filenames and captions found in parser metadata. Nothing to process for Stage 2.")
        return {
            'duration': 0,
            'images_input': 0,
            'images_approved': 0,
            'api_calls': 0,
            'total_tokens': 0,
            'cost': 0.0
        }

    images_input = len(df_filtered)
    print(f"Found {images_input} images from Stage 1 to process through GPT-4.1-Nano.")
    os.makedirs(STAGE2_OUTPUT_DIR, exist_ok=True)
    
    stage2_approved_metadata = []
    
    chunks = [df_filtered[i:i + BATCH_SIZE] for i in range(0, df_filtered.shape[0], BATCH_SIZE)]

    for chunk in tqdm(chunks, desc="GPT-4.1-Nano (Batched)"):
        batch_start_time = time.time()
        
        batch_image_paths = []
        batch_rows = []
        
        for index, row in chunk.iterrows():
            image_filename = row['file_name']
            image_path = os.path.join(PARSER_OUTPUT_DIR, image_filename)

            if not os.path.exists(image_path):
                tqdm.write(f"WARNING: Image {image_filename} not found in '{PARSER_OUTPUT_DIR}'. Skipping.")
                continue
            
            batch_image_paths.append(image_path)
            batch_rows.append(row)
        
        if not batch_image_paths:
            continue

        # Call API with batch
        res = gpt_4_1_nano_classifier.classify_images(batch_image_paths)
        api_calls += 1
        
        # Track token usage
        if res and res.get('usage'):
            total_prompt_tokens += res['usage'].get('prompt_tokens', 0)
            total_completion_tokens += res['usage'].get('completion_tokens', 0)
        
        # Process results
        if res and not res.get('error'):
             for i, image_path in enumerate(batch_image_paths):
                filename = os.path.basename(image_path)
                # The result should be keyed by filename
                image_result = res.get(filename)
                
                if image_result:
                    if not image_result.get('is_data_plot') and image_result.get('is_nuclear_schematic'):
                        tqdm.write(f"PASS (Nano): {filename}")
                        shutil.copy(image_path, os.path.join(STAGE2_OUTPUT_DIR, filename))
                        # Find the original row data
                        original_row = next((r for r in batch_rows if r['file_name'] == filename), None)
                        if original_row is not None:
                            stage2_approved_metadata.append(original_row.to_dict())
                    else:
                        reason = image_result.get('reasoning', 'Unknown reason')
                        tqdm.write(f"FILTERED (Nano): {filename}. Reason: {reason}")
                else:
                    tqdm.write(f"ERROR: No result returned for {filename} in batch response.")
        else:
            error_msg = res.get('error', 'Unknown error') if res else 'No response'
            tqdm.write(f"BATCH ERROR: {error_msg}")

        elapsed_time = time.time() - batch_start_time
        sleep_needed = max(0, API_CALL_DELAY_SECONDS - elapsed_time)
        time.sleep(sleep_needed)

    duration = time.time() - start_time
    total_tokens = total_prompt_tokens + total_completion_tokens
    cost = (total_prompt_tokens / 1_000_000 * GPT_4_1_NANO_INPUT_PRICE + 
            total_completion_tokens / 1_000_000 * GPT_4_1_NANO_OUTPUT_PRICE)
    
    images_approved = len(stage2_approved_metadata)
    
    if not stage2_approved_metadata:
        print("\nINFO: No images passed the GPT-4.1-Nano classification stage.")
    else:
        pd.DataFrame(stage2_approved_metadata).to_csv(STAGE2_METADATA_FILE, index=False)
        print(f"\nSUCCESS: Stage 2 complete. Saved {images_approved} approved images to '{STAGE2_OUTPUT_DIR}'")
    
    print(f"\n✓ Stage 2 complete in {duration:.1f}s")
    print(f"  Images processed: {images_input}")
    print(f"  Images approved: {images_approved} ({images_approved/images_input*100:.1f}%)")
    print(f"  API calls: {api_calls}")
    print(f"  Tokens used: {total_tokens:,} (prompt: {total_prompt_tokens:,}, completion: {total_completion_tokens:,})")
    print(f"  Cost: ${cost:.4f}")
    
    return {
        'duration': duration,
        'images_input': images_input,
        'images_approved': images_approved,
        'api_calls': api_calls,
        'total_tokens': total_tokens,
        'prompt_tokens': total_prompt_tokens,
        'completion_tokens': total_completion_tokens,
        'cost': cost
    }

def run_stage3_gpt_5_nano():
    """Runs GPT-5-Nano validation on images from the GPT-4.1-Nano stage."""
    print("\n" + "="*60)
    print("Stage 3: GPT-5-Nano Validation")
    print("="*60)
    
    start_time = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    api_calls = 0
    
    if not os.path.exists(STAGE2_OUTPUT_DIR) or not os.path.exists(STAGE2_METADATA_FILE):
        print(f"FATAL: Cannot start Stage 3. Missing Stage 2 output directory ('{STAGE2_OUTPUT_DIR}') or metadata ('{STAGE2_METADATA_FILE}').")
        print("Please run Stage 2 first.")
        sys.exit(1)

    try:
        df = pd.read_csv(
            STAGE2_METADATA_FILE,
            dtype={'file_name': str, 'caption': str},
            on_bad_lines='warn'
        )
    except Exception as e:
        print(f"FATAL: Could not read {STAGE2_METADATA_FILE}. Error: {e}")
        sys.exit(1)
        
    df_filtered = df[df['file_name'].notna()].copy()

    if df_filtered.empty:
        print("INFO: No images with valid filenames found in Stage 2 metadata. Nothing to process for Stage 3.")
        return {
            'duration': 0,
            'images_input': 0,
            'images_approved': 0,
            'api_calls': 0,
            'total_tokens': 0,
            'cost': 0.0
        }

    images_input = len(df_filtered)
    print(f"Found {images_input} images from Stage 2 to validate with GPT-5-Nano.")
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
    
    final_metadata = []
    
    chunks = [df_filtered[i:i + BATCH_SIZE] for i in range(0, df_filtered.shape[0], BATCH_SIZE)]

    for chunk in tqdm(chunks, desc="GPT-5-Nano (Batched)"):
        batch_start_time = time.time()
        
        batch_image_paths = []
        batch_rows = []

        for index, row in chunk.iterrows():
            image_filename = row['file_name']
            image_path = os.path.join(STAGE2_OUTPUT_DIR, image_filename)

            if not os.path.exists(image_path):
                tqdm.write(f"WARNING: Image {image_filename} not found in '{STAGE2_OUTPUT_DIR}'. Skipping.")
                continue
            
            batch_image_paths.append(image_path)
            batch_rows.append(row)

        if not batch_image_paths:
            continue
            
        res = gpt_5_nano_classifier.classify_images(batch_image_paths)
        api_calls += 1
        
        # Track token usage
        if res and res.get('usage'):
            total_prompt_tokens += res['usage'].get('prompt_tokens', 0)
            total_completion_tokens += res['usage'].get('completion_tokens', 0)

        if res and not res.get('error'):
            for i, image_path in enumerate(batch_image_paths):
                filename = os.path.basename(image_path)
                image_result = res.get(filename)
                
                if image_result:
                    if not image_result.get('is_data_plot') and image_result.get('is_nuclear_schematic'):
                        tqdm.write(f"SUCCESS: {filename} passed all filters.")
                        shutil.copy(image_path, os.path.join(FINAL_OUTPUT_DIR, filename))
                        
                        original_row = next((r for r in batch_rows if r['file_name'] == filename), None)
                        if original_row is not None:
                            final_metadata.append({
                                'file_name': original_row['file_name'],
                                'caption': original_row['caption'],
                                'reference': original_row['reference']
                            })
                    else:
                        reason = image_result.get('reasoning', 'Unknown reason')
                        tqdm.write(f"FILTERED (5-Nano): {filename}. Reason: {reason}")
                else:
                    tqdm.write(f"ERROR: No result returned for {filename} in batch response.")
        else:
             error_msg = res.get('error', 'Unknown error') if res else 'No response'
             tqdm.write(f"BATCH ERROR: {error_msg}")

        elapsed_time = time.time() - batch_start_time
        sleep_needed = max(0, API_CALL_DELAY_SECONDS - elapsed_time)
        time.sleep(sleep_needed)

    duration = time.time() - start_time
    total_tokens = total_prompt_tokens + total_completion_tokens
    cost = (total_prompt_tokens / 1_000_000 * GPT_5_NANO_INPUT_PRICE + 
            total_completion_tokens / 1_000_000 * GPT_5_NANO_OUTPUT_PRICE)
    
    images_approved = len(final_metadata)
    
    if not final_metadata:
        print("\nINFO: No images passed the final GPT-5-Nano validation stage.")
    else:
        pd.DataFrame(final_metadata).to_csv(FINAL_METADATA_FILE, index=False)
        print(f"\nSUCCESS: Stage 3 complete. Saved {images_approved} approved images to '{FINAL_METADATA_FILE}'.")
    
    print(f"\n✓ Stage 3 complete in {duration:.1f}s")
    print(f"  Images processed: {images_input}")
    if images_input > 0:
        print(f"  Images approved: {images_approved} ({images_approved/images_input*100:.1f}%)")
    else:
        print(f"  Images approved: {images_approved}")
    print(f"  API calls: {api_calls}")
    print(f"  Tokens used: {total_tokens:,} (prompt: {total_prompt_tokens:,}, completion: {total_completion_tokens:,})")
    print(f"  Cost: ${cost:.4f}")
    
    return {
        'duration': duration,
        'images_input': images_input,
        'images_approved': images_approved,
        'api_calls': api_calls,
        'total_tokens': total_tokens,
        'prompt_tokens': total_prompt_tokens,
        'completion_tokens': total_completion_tokens,
        'cost': cost
    }

def print_pipeline_summary(stage1_metrics, stage2_metrics, stage3_metrics, total_duration):
    """Print comprehensive pipeline summary."""
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    
    # Stage 1
    if stage1_metrics:
        print(f"\nStage 1 (Local Parser & Classifier):")
        if not stage1_metrics.get('skipped'):
            print(f"  Duration: {stage1_metrics['duration']:.1f}s")
        print(f"  Images processed: {stage1_metrics.get('images_input', 'N/A')}")
        if stage1_metrics.get('images_input', 0) > 0:
            pass_rate = stage1_metrics['images_approved'] / stage1_metrics['images_input'] * 100
            print(f"  Images approved: {stage1_metrics['images_approved']} ({pass_rate:.1f}%)")
        else:
            print(f"  Images approved: {stage1_metrics['images_approved']}")
    
    # Stage 2
    if stage2_metrics:
        print(f"\nStage 2 (GPT-4.1-Nano):")
        print(f"  Duration: {stage2_metrics['duration']:.1f}s")
        print(f"  Images processed: {stage2_metrics['images_input']}")
        if stage2_metrics['images_input'] > 0:
            pass_rate = stage2_metrics['images_approved'] / stage2_metrics['images_input'] * 100
            print(f"  Images approved: {stage2_metrics['images_approved']} ({pass_rate:.1f}%)")
        else:
            print(f"  Images approved: {stage2_metrics['images_approved']}")
        print(f"  API calls: {stage2_metrics['api_calls']}")
        print(f"  Tokens: {stage2_metrics['total_tokens']:,}")
        print(f"  Cost: ${stage2_metrics['cost']:.4f}")
    
    # Stage 3
    if stage3_metrics:
        print(f"\nStage 3 (GPT-5-Nano):")
        print(f"  Duration: {stage3_metrics['duration']:.1f}s")
        print(f"  Images processed: {stage3_metrics['images_input']}")
        if stage3_metrics['images_input'] > 0:
            pass_rate = stage3_metrics['images_approved'] / stage3_metrics['images_input'] * 100
            print(f"  Images approved: {stage3_metrics['images_approved']} ({pass_rate:.1f}%)")
        else:
            print(f"  Images approved: {stage3_metrics['images_approved']}")
        print(f"  API calls: {stage3_metrics['api_calls']}")
        print(f"  Tokens: {stage3_metrics['total_tokens']:,}")
        print(f"  Cost: ${stage3_metrics['cost']:.4f}")
    
    # Totals
    print(f"\n" + "-"*70)
    total_cost = 0
    if stage2_metrics:
        total_cost += stage2_metrics['cost']
    if stage3_metrics:
        total_cost += stage3_metrics['cost']
    
    final_images = stage3_metrics['images_approved'] if stage3_metrics else 0
    
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    
    print(f"Total pipeline execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Total API cost: ${total_cost:.4f}")
    if final_images > 0:
        print(f"Cost per final image: ${total_cost/final_images:.4f}")
    print(f"Final images saved: {final_images}")
    print("="*70)

def main():
    """Main function to run the entire pipeline."""
    parser_args = argparse.ArgumentParser(description="Nuclear Schematic Extraction Pipeline")
    parser_args.add_argument(
        '--start-stage',
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="The stage to start the pipeline from: 1 (Parser), 2 (GPT-4.1-Nano), 3 (GPT-5-Nano)."
    )
    args = parser_args.parse_args()

    print(f"Starting Nuclear Schematic Extraction Pipeline from Stage {args.start_stage}...")
    
    pipeline_start = time.time()
    stage1_metrics = None
    stage2_metrics = None
    stage3_metrics = None
    
    if args.start_stage <= 1:
        stage1_metrics = run_stage1_parser()
    
    if args.start_stage <= 2:
        stage2_metrics = run_stage2_gpt_4_1_nano()
    
    if args.start_stage <= 3:
        stage3_metrics = run_stage3_gpt_5_nano()
    
    total_duration = time.time() - pipeline_start
    
    print_pipeline_summary(stage1_metrics, stage2_metrics, stage3_metrics, total_duration)
    
    print(f"\nFinal output directory: '{FINAL_OUTPUT_DIR}'")
    print(f"Final metadata file: '{FINAL_METADATA_FILE}'")

if __name__ == "__main__":
    main()