import os
import re
import fitz
import io
import csv
import json
import hashlib
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# REPLACE WITH YOUR UNIQNAME
USERNAME_PREFIX = "jeremoon"
PDF_DIR = "pdf_files/"
REFERENCE_DIR = "reference_images/"

class ImageClassifier:
    """
    A multi-stage classifier using an expanded semantic vocabulary and a strict,
    multi-condition filter that combines text and visual similarity for maximum accuracy.
    """
    def __init__(self, model_name='clip-ViT-L-14', reference_dir=REFERENCE_DIR):
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():  # For Apple Silicon (M1/M2/M3)
            self.device = 'mps'
        else:
            self.device = 'cpu'
        
        print(f"PASS: Device selected: {self.device.upper()}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # --- Model Hyperparameters (GOAL: prioritize recall over precision) ---
        self.keep_threshold = 0.20
        self.strong_reject_threshold = 0.28
        self.visual_similarity_threshold = 0.72
        
        self.keep_labels = [
            "a detailed schematic diagram of a nuclear reactor core with hexagonal fuel assemblies arranged in a grid pattern",
            "a technical cross-section engineering diagram of a reactor pressure vessel showing internal components and structures",
            "a colorful heat map showing power distribution or neutron flux across a reactor core with a color scale bar",
            "a detailed engineering blueprint with dimensions, annotations, and technical specifications",
            "a 3D CAD model rendering of a nuclear fuel assembly, control rod, or reactor component with technical details",
            "a cutaway technical illustration of a reactor vessel showing the interior arrangement of fuel and coolant systems",
            "a schematic diagram of a nuclear power plant cooling system with pipes, pumps, and heat exchangers labeled",
            "a top-down view of a hexagonal reactor core configuration showing fuel rod positions",
            "a technical diagram showing radiation shielding layers made of concrete, lead, or water",
            "an illustration of nuclear fuel pellets, fuel rods, or fuel bundle assemblies with structural details",
            "a diagram of coolant flow paths through a reactor core with arrows indicating direction",
            "a simulation visualization from MCNP, Serpent, or similar nuclear physics code showing geometry",
            "a photograph of Cherenkov radiation glowing blue in a research reactor pool",
            "a technical diagram of a particle detector or experimental physics apparatus with labeled components",
            "a detailed illustration of reactor control mechanisms including control rods and drive systems",
            "a cross-sectional view of a nuclear fuel element showing cladding, pellets, and internal structure",
            "a simplified two-dimensional top-down schematic of a reactor core layout with labeled regions",
            "a clean 2D overhead diagram of a nuclear reactor showing fuel assembly positions and symmetry",
            "a technical illustration of a reactor core arrangement viewed from above with grid coordinates",
            "a labeled diagram showing the radial layout of fuel assemblies, reflectors, and control elements in a reactor",
            "a colored schematic of a reactor vessel cross-section with component labels and arrows"
        ]
        
        self.reject_labels = [
            "a line graph with x and y axes showing curves or data points with a legend",
            "a scatter plot with x and y axes showing experimental data points with error bars or markers",
            "a histogram or bar chart comparing numerical data across categories with vertical or horizontal bars",
            "a mathematical plot showing functions, curves, or statistical distributions on a coordinate system with gridlines",
            "a graph with logarithmic scale axes showing exponential or power-law relationships",
            "a plot with multiple colored curves or lines comparing different cases or scenarios",
            "a chart showing trends over time with data points connected by lines",
            "a graph with a legend showing Case 1, Case 2, Case 3, or similar labels",
            "a plot showing irradiation time, burnup percentage, or activity measurements on the axes",
            "a graph comparing different targets, enrichments, or experimental conditions",
            "a scientific data plot with markers, error bars, and curve fitting",
            "a phase diagram or contour plot showing relationships between two variables",
            "the front cover of a journal, book, or academic paper with title and author names",
            "a title page or acknowledgments page with text listing authors and institutions",
            "an aerial photograph or satellite image of a building or industrial facility",
            "a photograph of people in a laboratory or workplace setting",
            "a photograph of physical equipment, machinery, or devices sitting on a table or floor",
            "a simple flowchart with rectangular boxes and arrows connecting them",
            "a company logo, university emblem, or institutional seal",
            "a blank or mostly white page with minimal content",
            "a decorative border, background texture, or graphic design element",
            "a screenshot of computer software, a user interface, or application window",
            "a simple 3D geometric shape like a sphere, cube, or cylinder with no technical context",
            "a world map, geographical map, or navigation chart",
            "a table containing rows and columns of numerical data or text",
            "a presentation slide with bullet points and minimal graphics",
            "a hand-drawn sketch or informal doodle",
            "a portrait photograph or headshot of a person",
            "a generic stock photo or clipart image"
        ]
        
        self.reference_embeddings = self._load_reference_images(reference_dir)
        
        # --- OPTIMIZATION: PRE-COMPUTE TEXT EMBEDDINGS---
        self.keep_embeddings = self.model.encode(self.keep_labels, convert_to_tensor=True)
        self.reject_embeddings = self.model.encode(self.reject_labels, convert_to_tensor=True)

    def _load_reference_images(self, reference_dir):
        """Load and embed reference images for visual similarity comparison."""
        if not os.path.isdir(reference_dir):
            print(f"WARNING: Reference directory '{reference_dir}' not found. Visual filtering disabled.")
            return None
        
        embeddings = []
        loaded_count = 0
        for filename in os.listdir(reference_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    path = os.path.join(reference_dir, filename)
                    img = Image.open(path).convert('RGB')
                    embeddings.append(self.model.encode(img, convert_to_tensor=True))
                    loaded_count += 1
                except Exception as e:
                    print(f"WARNING: Could not load reference image {filename}: {e}")
        
        if loaded_count > 0:
            print(f"PASS: Loaded {loaded_count} reference images for visual comparison.")
            return torch.stack(embeddings)
        else:
            print("WARNING: No reference images loaded. Visual filtering disabled.")
            return None

    def encode_batch(self, images):
        """Encodes a list of PIL images in one batch."""
        if not images:
            return None
        return self.model.encode(images, convert_to_tensor=True)

    def classify_image(self, image_input, embedding=None):
        """
        Classify an image using improved dual-scoring logic. 
        Args:
            image_input: Either raw bytes or a PIL Image object (batch).
            embedding: Pre-computed embedding tensor (optional).
        Returns dict with decision, scores, and reasoning.
        """
        result = {
            'decision': False,
            'keep_label': 'N/A',
            'reject_label': 'N/A', 
            'keep_score': 0.0,
            'reject_score': 0.0,
            'similarity_score': 0.0,
            'reason': ''
        }
        
        try:
            if isinstance(image_input, bytes):
                image = Image.open(io.BytesIO(image_input)).convert('RGB')
            else:
                image = image_input

            if embedding is not None:
                image_embedding = embedding
            else:
                image_embedding = self.model.encode(image, convert_to_tensor=True)

            # --- PRELIMINARY IMAGE QUALITY CHECKS ---
            width, height = image.size
            aspect_ratio = max(width, height) / min(width, height)
            
            # Rule 1: Filter out very small images (GOAL: reject thumbnails/icons)
            if width < 100 or height < 100:
                result['reason'] = f'REJECTED: Image too small ({width}x{height})'
                return result
            
            # Rule 2: Filter out extreme aspect ratios (GOAL: reject banners/decorative elements)
            if aspect_ratio > 4.0:
                result['reason'] = f'REJECTED: Extreme aspect ratio ({aspect_ratio:.1f})'
                return result
            
            # Rule 3: Check color variance (GOAL: detect and detect blank/solid color images)
            img_array = np.array(image)
            color_variance = np.var(img_array)
            if color_variance < 100:  # low variance = likely blank
                result['reason'] = f'REJECTED: Low color variance (likely blank/solid, variance: {color_variance:.1f})'
                return result
            
            # --- DUAL TEXT CLASSIFICATION ---
            keep_similarities = torch.nn.functional.cosine_similarity(
                image_embedding.unsqueeze(0), self.keep_embeddings
            )
            reject_similarities = torch.nn.functional.cosine_similarity(
                image_embedding.unsqueeze(0), self.reject_embeddings
            )
            
            max_keep_score = keep_similarities.max().item()
            max_reject_score = reject_similarities.max().item()
            best_keep_idx = keep_similarities.argmax().item()
            best_reject_idx = reject_similarities.argmax().item()
            
            result['keep_label'] = self.keep_labels[best_keep_idx]
            result['reject_label'] = self.reject_labels[best_reject_idx]
            result['keep_score'] = round(max_keep_score, 3)
            result['reject_score'] = round(max_reject_score, 3)
            
            # --- PERMISSIVE FILTERING LOGIC (Goal: Prioritize Recall) ---
            # Philosophy: Better to save some junk than miss good diagrams
            
            # Rule 4: Only reject if reject score is SIGNIFICANTLY higher than keep score AND reject score exceeding a strong threshold
            if max_reject_score > self.strong_reject_threshold and max_reject_score > max_keep_score + 0.05:
                result['reason'] = f'REJECTED: Strong match to reject category (reject: {result["reject_score"]}, keep: {result["keep_score"]})'
                return result
            
            # Rule 5: Very weak keep signal - almost no confidence
            if max_keep_score < self.keep_threshold:
                result['reason'] = f'REJECTED: Very low confidence match to keep categories (best score: {result["keep_score"]})'
                return result
            
            # Rule 6: Edge case - if BOTH scores are nearly identical BUT in reject territory
            # Indended to catch data plots that match both keep and reject equally
            if abs(max_keep_score - max_reject_score) < 0.02 and max_reject_score > 0.27:
                result['reason'] = f'REJECTED: Equal match to both categories with high reject score (keep: {result["keep_score"]}, reject: {result["reject_score"]})'
                return result
            
            # Rule 7: If keep score >= threshold, allow through to visual check
            # --- VISUAL SIMILARITY CHECK ---
            if self.reference_embeddings is not None:
                visual_similarities = torch.nn.functional.cosine_similarity(
                    image_embedding.unsqueeze(0), self.reference_embeddings
                )
                max_visual_similarity = visual_similarities.max().item()
                result['similarity_score'] = round(max_visual_similarity, 3)
                
                if max_visual_similarity < self.visual_similarity_threshold:
                    result['reason'] = f'REJECTED: Low visual similarity to reference images (score: {result["similarity_score"]})'
                    return result
            
            # --- ALL CHECKS PASSED ---
            result['decision'] = True
            result['reason'] = f'PASS - Best match: "{self.keep_labels[best_keep_idx]}"'
            
        except Exception as e:
            result['reason'] = f'ERROR: Image processing error: {str(e)}'
        
        return result


def get_pdf_doi(doc):
    """Extract DOI from PDF metadata or content."""
    # Check metadata first
    if 'doi' in doc.metadata and doc.metadata['doi']:
        return doc.metadata['doi'].split('doi:')[-1].strip()
    
    # Search first 3 pages for DOI
    doi_pattern = re.compile(r'(?:doi:|DOI:|https?://doi\.org/)(\s*10\.\d{4,9}/[-._;()/:A-Z0-9]+)', re.IGNORECASE)
    for page_num in range(min(3, len(doc))):
        page = doc.load_page(page_num)
        match = doi_pattern.search(page.get_text())
        if match:
            return match.group(1).strip()
    
    return None


def extract_caption_near_image(page, img_bbox, page_blocks, max_distance=300):
    """Enhanced caption extraction using spatial proximity and pattern matching. Searches below the image for text starting with Fig/Figure/Table."""
    # Define search region below image
    search_rect = fitz.Rect(
        img_bbox.x0 - 100,          # Extend left to catch left-aligned captions
        img_bbox.y1,                # Start below image
        img_bbox.x1 + 100,          # Extend right
        img_bbox.y1 + max_distance  # Search downward
    )
    
    # Filter pre-fetched blocks based on coordinates manually
    blocks = [
        b for b in page_blocks 
        if (search_rect.x0 < b[0] < search_rect.x1) and # X check (simplified intersection)
           (search_rect.y0 < b[1] < search_rect.y1)     # Y check
    ]
    
    potential_captions = []
    caption_pattern = re.compile(
        r'^(Fig(?:ure)?|Table|Scheme)\.?\s*\d+',
        re.IGNORECASE
    )
    
    for block in blocks:
        block_text = " ".join(block[4].strip().split())  # Normalize whitespace
        
        # Check if block starts with caption pattern
        if caption_pattern.match(block_text):
            distance = block[1] - img_bbox.y1  # y-coordinate of block top
            potential_captions.append((distance, block_text))
    
    if potential_captions:
        # Sort by proximity and return closest caption
        potential_captions.sort(key=lambda x: x[0])
        return potential_captions[0][1]
    
    # FALLBACK: search for any text that looks like a caption within region
    for block in blocks:
        block_text = " ".join(block[4].strip().split())

        # Look for descriptive text (longer than 20 chars, contains common caption words)
        if len(block_text) > 20 and any(word in block_text.lower() for word in ['diagram', 'schematic', 'cross-section', 'view', 'configuration', 'system']):
            return block_text[:200]  # Truncate long text
    
    return "N/A"


def extract_pdf_data(pdf_path, classifier, processed_hashes):
    """Extract approved images and captions from PDF."""
    print(f"\nProcessing: {pdf_path}")
    approved_images = []
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"ERROR: Cannot open PDF - {e}")
        return approved_images, 0

    doi = get_pdf_doi(doc)
    if doi:
        print(f"PASS - DOI found: {doi}")
    
    total_images = 0
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # --- OPTIMIZATION: Get text blocks once per page ---
        page_blocks = page.get_text("blocks")
        
        images = page.get_images(full=True)
        total_images += len(images)
        
        # Collect valid images for this page
        page_image_candidates = []
        
        for img_index, img in enumerate(images):  # img is a tuple: (xref, smask, width, height, bpc, colorspace, ...)
            xref = img[0]
            width = img[2]
            height = img[3]
            
            # --- OPTIMIZATION: Filter before extraction ---
            if width < 100 or height < 100:
                continue
            
            try:
                # Extract image data
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Check for duplicates in hashset
                image_hash = hashlib.sha256(image_bytes).hexdigest()
                if image_hash in processed_hashes:
                    continue
                
                # Convert to PIL for batch encoding
                pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                page_image_candidates.append({
                    'index': img_index,
                    'xref': xref,
                    'bytes': image_bytes,
                    'hash': image_hash,
                    'pil': pil_image,
                    'ext': base_image['ext'],
                    'img_obj': img
                })
                
            except Exception as e:
                 print(f"  WARNING: Error extracting image {img_index + 1} on page {page_num + 1}: {e}")
                 continue

        if not page_image_candidates:
            continue

        # --- OPTIMIZATION: Batch Inference ---
        pil_images = [c['pil'] for c in page_image_candidates]
        try:
            batch_embeddings = classifier.encode_batch(pil_images)
        except Exception as e:
            print(f"  WARNING: Batch encoding failed on page {page_num + 1}: {e}")
            continue

        # Process results
        for i, candidate in enumerate(page_image_candidates):
            embedding = batch_embeddings[i]
            
            # Classify using pre-computed embedding
            classification_result = classifier.classify_image(candidate['pil'], embedding=embedding)
            
            if not classification_result['decision']:
                print(f"DEBUG - Page {page_num + 1}, Image {candidate['index'] + 1}: {classification_result['reason']}")
                continue
            
            # Mark as processed
            processed_hashes.add(candidate['hash'])
            
            # Extract caption using cached blocks
            img_bbox = page.get_image_bbox(candidate['img_obj'])
            caption = extract_caption_near_image(page, img_bbox, page_blocks)
            
            approved_images.append({
                "image_bytes": candidate['bytes'],
                "caption": caption,
                "doi": doi,
                "extension": candidate['ext'],
                "page_num": page_num + 1,
                "classification_result": classification_result
            })
            
            print(f"  ✓ Page {page_num + 1}, Image {candidate['index'] + 1}: APPROVED")
            print(f"    Keep score: {classification_result['keep_score']}, Visual: {classification_result['similarity_score']}")
            print(f"    Caption: {caption[:80]}...")

    doc.close()
    print(f"  Found {len(approved_images)} approved images from {total_images} total images")
    return approved_images, total_images


def main():
    print("=" * 60)
    print("Nuclear Reactor Diagram Extraction Script")
    print("=" * 60)
    
    # Initialize classifier
    print("\nInitializing classifier...")
    classifier = ImageClassifier()
    
    # Setup output directories
    master_output_dir = "saved"
    os.makedirs(master_output_dir, exist_ok=True)
    
    # Initialize metadata CSV
    metadata_filename = "metadata.csv"
    if not os.path.exists(metadata_filename):
        with open(metadata_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['file_name', 'caption', 'reference', 'page_num', 'keep_score', 'visual_score'])
    
    # Track processed images in hashset
    processed_hashes = set()
    image_counter = 1
    classification_log = []

    # Add counters for total images processed and filtered
    total_images_processed = 0
    total_images_approved = 0
    
    # Find PDFs to process
    if not os.path.isdir(PDF_DIR):
        print(f"\nERROR: Directory '{PDF_DIR}' not found!")
        print("Please create 'pdf_files/' directory and add PDF files.")
        return
    
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"\nERROR: No PDF files found in '{PDF_DIR}'")
        return
    
    print(f"\nPASS: Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    for pdf_filename in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_filename)
        approved_images, images_in_pdf = extract_pdf_data(pdf_path, classifier, processed_hashes)
        total_images_processed += images_in_pdf
        total_images_approved += len(approved_images)
        
        # Save approved images
        for image_data in approved_images:
            file_name = f"{USERNAME_PREFIX}_{image_counter}.{image_data['extension']}"
            output_path = os.path.join(master_output_dir, file_name)
            
            # Write image file
            with open(output_path, "wb") as f:
                f.write(image_data['image_bytes'])
            
            # Append to metadata CSV
            doi_url = f"https://doi.org/{image_data['doi']}" if image_data['doi'] else "N/A"
            with open(metadata_filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                result = image_data['classification_result']
                writer.writerow([
                    file_name,
                    image_data['caption'],
                    doi_url,
                    image_data['page_num'],
                    result['keep_score'],
                    result['similarity_score']
                ])
            
            # Log classification details
            log_entry = {
                'file_name': file_name,
                'source': pdf_path,
                'page_num': image_data['page_num'],
                'caption': image_data['caption'],
                'keep_classification': result['keep_label'],
                'reject_classification': result['reject_label'],
                'keep_score': result['keep_score'],
                'reject_score': result['reject_score'],
                'visual_similarity_score': result['similarity_score'],
                'decision_reason': result['reason']
            }

            # Update classification_log.json for logging purposes
            log_filename = "classification_log.json"
            try:
                if os.path.exists(log_filename) and os.path.getsize(log_filename) > 0:
                    with open(log_filename, 'r', encoding='utf-8') as f:
                        logs = json.load(f)
                else:
                    logs = []
                
                logs.append(log_entry)
                with open(log_filename, 'w', encoding='utf-8') as f:
                    json.dump(logs, f, indent=2)
            except (IOError, json.JSONDecodeError) as e:
                print(f"ERROR: Could not update log file {log_filename}: {e}")
            
            image_counter += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total images processed: {total_images_processed}")
    print(f"Total images approved: {total_images_approved}")
    if total_images_processed > 0:
        percentage_filtered = ((total_images_processed - total_images_approved) / total_images_processed) * 100
        print(f"Percentage of images filtered out: {percentage_filtered:.2f}%")
    else:
        print("Percentage of images filtered out: 0.00%")
    print("=" * 60)
    
    return total_images_processed, total_images_approved

if __name__ == "__main__":
    main()