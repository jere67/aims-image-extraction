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

def run_stage1_parser():
    """Checks for PDFs and runs the parser main function if they exist."""
    print("\n" + "="*60)
    print("Stage 1: PDF Parsing and Initial Filtering")
    print("="*60)
    
    if not os.path.exists(PDF_DIR) or not any(f.lower().endswith('.pdf') for f in os.listdir(PDF_DIR)):
        print("INFO: No new PDFs found in 'pdf_files/'. Skipping parser run.")
        if not os.path.exists(PARSER_OUTPUT_DIR):
             print(f"FATAL: No PDFs to process and the parser output directory ('{PARSER_OUTPUT_DIR}') does not exist. Cannot continue.")
             sys.exit(1)
        return

    print("INFO: New PDFs found. Running the local parser and classifier...")
    try:
        parser.main()
        print("SUCCESS: parser.py executed successfully.")
    except Exception as e:
        print(f"FATAL: An error occurred while running parser.py: {e}")
        sys.exit(1)

def run_stage2_gpt_4_1_nano():
    """Runs GPT-4.1-Nano classification on images from the parser stage."""
    print("\n" + "="*60)
    print("Stage 2: GPT-4.1-Nano Classification")
    print("="*60)

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
        return

    print(f"Found {len(df_filtered)} images from Stage 1 to process through GPT-4.1-Nano.")
    os.makedirs(STAGE2_OUTPUT_DIR, exist_ok=True)
    
    stage2_approved_metadata = []
    
    for index, row in tqdm(df_filtered.iterrows(), total=df_filtered.shape[0], desc="GPT-4.1-Nano"):
        image_filename = row['file_name']
        image_path = os.path.join(PARSER_OUTPUT_DIR, image_filename)

        if not os.path.exists(image_path):
            tqdm.write(f"WARNING: Image {image_filename} not found in '{PARSER_OUTPUT_DIR}'. Skipping.")
            continue

        res = gpt_4_1_nano_classifier.classify_image(image_path)
        
        if res and not res.get('error') and not res.get('is_data_plot') and res.get('is_nuclear_schematic'):
            tqdm.write(f"PASS (Nano): {image_filename}")
            shutil.copy(image_path, os.path.join(STAGE2_OUTPUT_DIR, image_filename))
            stage2_approved_metadata.append(row.to_dict())
        else:
            reason = res.get('reasoning', res.get('error', 'Unknown error'))
            tqdm.write(f"FILTERED (Nano): {image_filename}. Reason: {reason}")
        
        time.sleep(API_CALL_DELAY_SECONDS)

    if not stage2_approved_metadata:
        print("\nINFO: No images passed the GPT-4.1-Nano classification stage.")
        return

    pd.DataFrame(stage2_approved_metadata).to_csv(STAGE2_METADATA_FILE, index=False)
    print(f"\nSUCCESS: Stage 2 complete. Saved {len(stage2_approved_metadata)} approved images to '{STAGE2_OUTPUT_DIR}'")

def run_stage3_gpt_5_nano():
    """Runs GPT-5-Nano validation on images from the GPT-4.1-Nano stage."""
    print("\n" + "="*60)
    print("Stage 3: GPT-5-Nano Validation")
    print("="*60)
    
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
        return

    print(f"Found {len(df_filtered)} images from Stage 2 to validate with GPT-5-Nano.")
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
    
    final_metadata = []
    
    for index, row in tqdm(df_filtered.iterrows(), total=df_filtered.shape[0], desc="GPT-5-Nano"):
        image_filename = row['file_name']
        image_path = os.path.join(STAGE2_OUTPUT_DIR, image_filename)

        if not os.path.exists(image_path):
            tqdm.write(f"WARNING: Image {image_filename} not found in '{STAGE2_OUTPUT_DIR}'. Skipping.")
            continue
            
        res = gpt_5_nano_classifier.classify_image(image_path)

        if res and not res.get('error') and not res.get('is_data_plot') and res.get('is_nuclear_schematic'):
            tqdm.write(f"SUCCESS: {image_filename} passed all filters.")
            shutil.copy(image_path, os.path.join(FINAL_OUTPUT_DIR, image_filename))
            final_metadata.append({
                'file_name': row['file_name'],
                'caption': row['caption'],
                'reference': row['reference']
            })
        else:
            reason = res.get('reasoning', res.get('error', 'Unknown error'))
            tqdm.write(f"FILTERED (5-Nano): {image_filename}. Reason: {reason}")

        time.sleep(API_CALL_DELAY_SECONDS)

    if not final_metadata:
        print("\nINFO: No images passed the final GPT-5-Nano validation stage.")
        return

    pd.DataFrame(final_metadata).to_csv(FINAL_METADATA_FILE, index=False)
    print(f"\nSUCCESS: Stage 3 complete. Saved {len(final_metadata)} approved images to '{FINAL_METADATA_FILE}'.")

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
    
    if args.start_stage <= 1:
        run_stage1_parser()
    
    if args.start_stage <= 2:
        run_stage2_gpt_4_1_nano()
    
    if args.start_stage <= 3:
        run_stage3_gpt_5_nano()
    
    print("\n" + "="*60)
    print("Pipeline execution complete.")
    print(f"Final approved images are in: '{FINAL_OUTPUT_DIR}'")
    print(f"Final metadata is in: '{FINAL_METADATA_FILE}'")
    print("="*60)

if __name__ == "__main__":
    main()