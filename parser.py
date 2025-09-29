import os
import re
import fitz
from urllib.parse import urljoin
import io
import csv
import json
import hashlib
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch

USERNAME_PREFIX = "jeremoon"

class ImageClassifier:
    """
    A multi-stage classifier using an expanded semantic vocabulary and a strict,
    multi-condition filter that combines text and visual similarity for maximum accuracy.
    """
    def __init__(self, model_name='clip-ViT-B-32', reference_dir='reference_images/'):
        self.model = SentenceTransformer(model_name)
        
        # --- Thresholds ---
        self.text_keep_threshold = 0.25  # hyperparam #1
        self.image_similarity_threshold = 0.75  # hyperparam #2

        # --- Classification labels ---
        self.text_class_labels = [
            # === Descriptions of images TO KEEP ===
            "a schematic diagram of a nuclear reactor core with a hexagonal grid",
            "a technical cross-section diagram of a reactor vessel showing internal components",
            "a colorful heat map or simulation plot of a reactor core's power distribution",
            "a black and white engineering blueprint with dimensions and annotations",
            "a photograph of the glowing blue Cherenkov radiation in a reactor core",
            "a 3D CAD model rendering of a nuclear component or assembly with a plain background",
            "a diagram of a particle detector or experimental setup with labeled parts",
            "a schematic of a power plant's cooling system or steam cycle with arrows indicating flow",
            "a MCNP, Serpent, or Geant4 geometry plot showing particle tracks or cell layouts",
            "a cutaway view of a complex machine showing its interior",
            "an illustration of a fuel pellet, fuel rod, or fuel assembly",
            "a diagram showing concrete, lead, or polyethylene shielding layers",
            "a technical illustration of a circuit board or electronics module for a detector",
            "a simplified 2D drawing of a reactor facility layout",
            "a diagram of coolant flow paths in a reactor",
            "a top-down schematic of a circular or grid-based core configuration",

            # === Descriptions of images TO REJECT ===
            "a data plot with a logarithmic scale, labeled x and y axes, and a legend",
            "a line graph showing experimental data points with error bars and a best-fit curve",
            "a histogram or bar chart comparing statistical data",
            "a scatter plot showing a correlation between two variables",
            "a phase diagram or a plot of a mathematical function",
            "the cover page of an academic journal, textbook, or scientific paper",
            "a title page with a list of authors and institutional logos",
            "an aerial photograph of a power plant or industrial facility",
            "a photograph of a person, a group of people, or a piece of equipment in a laboratory",
            "a simple flowchart with text boxes and arrows",
            "a company, university, or government agency logo or emblem",
            "a blank white image, a simple color gradient, or a placeholder box",
            "a decorative graphic element, background texture, or border",
            "a screenshot of a software user interface or a computer screen",
            "a simple geometric shape like a cube or sphere with no context",
            "a map or satellite image", "a table of data",
            "a picture of a presentation slide with bullet points",
            "a photograph of a physical object on a table", "a hand-drawn sketch or doodle"
        ]
        self.num_desired_text_categories = 18
        
        self.reference_embeddings = self._load_reference_images(reference_dir)

    def _load_reference_images(self, reference_dir):
        if not os.path.isdir(reference_dir):
            print(f"ERROR: The reference image directory '{reference_dir}' was not found.")
            exit(1)
        embeddings = []
        for filename in os.listdir(reference_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    path = os.path.join(reference_dir, filename)
                    embeddings.append(self.model.encode(Image.open(path).convert('RGB'), convert_to_tensor=True))
                except Exception as e:
                    print(f"Warning: Could not load reference image {filename}: {e}")
        return torch.stack(embeddings) if embeddings else None

    def classify_image(self, image_bytes):
        result = {'decision': False, 'label': 'N/A', 'text_score': 0.0, 'similarity_score': 0.0, 'reason': ''}
        
        try:
            image_embedding = self.model.encode(Image.open(io.BytesIO(image_bytes)).convert('RGB'), convert_to_tensor=True)
            
            # --- Text Classification using Cosine Similarity ---
            text_similarities = torch.nn.functional.cosine_similarity(image_embedding, self.model.encode(self.text_class_labels, convert_to_tensor=True))
            max_text_similarity = text_similarities.max().item()
            top_text_index = text_similarities.argmax().item()
            
            result['label'] = self.text_class_labels[top_text_index]
            result['text_score'] = round(max_text_similarity, 3)
            
            is_potential_keep = top_text_index < self.num_desired_text_categories

            # --- Text filter ---
            if not is_potential_keep:
                result['reason'] = 'Rejected by text filter (classified as junk)' 
                return result
            if max_text_similarity < self.text_keep_threshold:
                result['reason'] = f'Rejected by text filter (low confidence score: {result["text_score"]})'
                return result
            
            # --- Visual Similarity ---
            if self.reference_embeddings is not None:
                visual_similarities = torch.nn.functional.cosine_similarity(image_embedding, self.reference_embeddings)
                max_visual_similarity = visual_similarities.max().item()
                result['similarity_score'] = round(max_visual_similarity, 3)

                if max_visual_similarity < self.image_similarity_threshold:
                    result['reason'] = f'Rejected by visual filter (low similarity score: {result["similarity_score"]})'
                    return result
            
            # If all checks pass
            result['decision'] = True
            result['reason'] = 'Passed all filters'
            
        except Exception as e:
            result['reason'] = f'Image processing error: {e}'
        
        return result


def get_pdf_doi(doc):
    if 'doi' in doc.metadata and doc.metadata['doi']: return doc.metadata['doi'].split('doi:')[-1].strip()
    doi_pattern = re.compile(r'doi:\s*(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', re.IGNORECASE)
    for page in doc:
        match = doi_pattern.search(page.get_text())
        if match: return match.group(1).strip()
    return None

def extract_pdf_data(pdf_path, classifier, processed_hashes):
    print(f"Processing PDF: {pdf_path}...")
    approved_images = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return approved_images

    doi = get_pdf_doi(doc)
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        for img in page.get_images(full=True):
            try:
                base_image = doc.extract_image(img[0])
                image_bytes = base_image["image"]
                image_hash = hashlib.sha256(image_bytes).hexdigest()
                if image_hash in processed_hashes: 
                    continue

                classification_result = classifier.classify_image(image_bytes)
                if not classification_result['decision']:
                    print(f"  - Rejected: {classification_result['reason']} | Label: '{classification_result['label']}'")
                    continue

                processed_hashes.add(image_hash)
                
                img_bbox = page.get_image_bbox(img)
                search_rect = fitz.Rect(img_bbox.x0 - 50, img_bbox.y1, img_bbox.x1 + 50, img_bbox.y1 + 250)
                
                potential_captions = []
                for block in page.get_text("blocks", clip=search_rect):
                    block_text = " ".join(block[4].strip().split())
                    if re.match(r'^(Fig(?:ure)?|Table)\.?\s*\d+', block_text, re.IGNORECASE):
                        potential_captions.append((block[1], block_text))
                
                final_caption = "No caption found"
                if potential_captions:
                    potential_captions.sort(key=lambda x: x[0])
                    final_caption = potential_captions[0][1]

                approved_images.append({
                    "image_bytes": image_bytes, "caption": final_caption, "doi": doi,
                    "extension": base_image['ext'], "classification_result": classification_result
                })
            except Exception as e:
                print(f"  - Warning: An unexpected error occurred while processing an image on page {page_num + 1}. Skipping. Error: {e}")
                continue

    doc.close()
    return approved_images


def main():
    classifier = ImageClassifier()
    
    master_output_dir = "saved"
    os.makedirs(master_output_dir, exist_ok=True)
    
    metadata_filename = "metadata.csv"
    classification_log = []
    
    if not os.path.exists(metadata_filename):
        with open(metadata_filename, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['file_name', 'text', 'Reference'])

    # Hashset to handle duplicates
    processed_hashes = set()
    image_counter = 1

    print("\n--- Image Extraction Script ---")
    
    all_data_to_process = []
    pdf_dir = "pdf_files"
    if not os.path.isdir(pdf_dir): 
        print(f"\nError: Directory '{pdf_dir}' not found.") 
        return
    pdf_paths = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    for pdf_path in pdf_paths: 
        all_data_to_process.append(("pdf", pdf_path))

    for data_type, path in all_data_to_process:
        approved_images = extract_pdf_data(path, classifier, processed_hashes)

        if approved_images is None:
            print(f"  - Warning: No images were processed for {path}. This might be due to an error.")
            continue

        for image_data in approved_images:
            file_name = f"{USERNAME_PREFIX}_{image_counter}.{image_data['extension']}"
            output_path = os.path.join(master_output_dir, file_name)
            
            with open(output_path, "wb") as f: f.write(image_data['image_bytes'])
            
            doi_url = f"https://doi.org/{image_data['doi']}" if image_data['doi'] else "N/A"
            with open(metadata_filename, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([file_name, image_data['caption'], doi_url])

            result = image_data['classification_result']
            log_entry = {
                'file_name': file_name, 'source': path, 'decision_reason': result['reason'],
                'text_classification': result['label'], 'text_score': result['text_score'],
                'visual_similarity_score': result['similarity_score']
            }
            classification_log.append(log_entry)
            print(f"  -> SAVED {file_name}")
            
            image_counter += 1

    log_filename = "classification_log.json"
    with open(log_filename, 'w', encoding='utf-8') as f:
        json.dump(classification_log, f, indent=4)
        
    print(f"\nScript finished. Saved {len(classification_log)} images.")
    print(f"Metadata updated in '{metadata_filename}'.")
    print(f"Detailed classification report saved to '{log_filename}'.")

if __name__ == "__main__":
    main()