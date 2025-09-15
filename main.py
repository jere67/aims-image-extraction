import os
import re
import string
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# --- Helper Functions ---

def sanitize_filename(filename):
    """
    Sanitizes a string to be used as a valid filename.
    Removes invalid characters and replaces spaces with underscores.
    """
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    sanitized_filename = ''.join(c for c in filename if c in valid_chars)
    sanitized_filename = sanitized_filename.replace(' ', '_').replace('/', '_').replace('\\', '_')
    return sanitized_filename

# --- PDF Processing Functions ---

def get_pdf_doi(doc):
    """
    Tries to find the DOI of a PDF document from metadata or text.
    """
    if 'doi' in doc.metadata and doc.metadata['doi']:
        return doc.metadata['doi'].split('doi:')[-1].strip()

    doi_pattern = re.compile(r'doi:\s*(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', re.IGNORECASE)
    for page_num in range(min(2, len(doc))):
        page = doc.load_page(page_num)
        text = page.get_text()
        match = doi_pattern.search(text)
        if match:
            return match.group(1).strip()
    return None

def extract_images_from_pdf(pdf_path, master_output_dir="saved"):
    """
    Core logic to extract images from a single PDF file.
    """
    print(f"Processing PDF: {pdf_path}...")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  - Error opening PDF: {e}")
        return

    doi = get_pdf_doi(doc)
    if doi:
        output_dir = os.path.join(master_output_dir, sanitize_filename(doi))
    else:
        pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join(master_output_dir, sanitize_filename(pdf_filename))
        print(f"  - Warning: DOI not found. Using filename '{pdf_filename}' for folder.")

    os.makedirs(output_dir, exist_ok=True)
    extracted_count = 0

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        if not image_list:
            continue
        
        text_blocks = page.get_text("blocks")
        for img_index, img in enumerate(image_list):
            img_bbox = page.get_image_bbox(img)
            caption_bbox = fitz.Rect(img_bbox.x0 - 20, img_bbox.y1, img_bbox.x1 + 20, img_bbox.y1 + 100)
            potential_captions = [b[4] for b in text_blocks if fitz.Rect(b[:4]).intersects(caption_bbox)]
            caption = " ".join(" ".join(c.strip().split()) for c in potential_captions)
            
            fig_match = re.search(r'(Figure|Fig\.?|Table)\s*([a-zA-Z0-9.]+)', caption, re.IGNORECASE)
            base_filename = f"{fig_match.group(1).replace('.', '')}_{fig_match.group(2)}" if fig_match else f"image_{page_num+1}_{img_index+1}"

            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes, image_ext = base_image["image"], base_image["ext"]
            
            image_filename = f"{sanitize_filename(base_filename)}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)
            
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            extracted_count += 1
    
    print(f"  - Finished. Extracted {extracted_count} images into '{output_dir}'.\n")
    doc.close()

# --- Web Scraping Functions ---

def get_url_doi(soup):
    """
    Tries to find the DOI of an article from a parsed HTML soup.
    """
    meta_doi = soup.find("meta", attrs={"name": "citation_doi"})
    if meta_doi and meta_doi.get("content"):
        return meta_doi.get("content")
    
    doi_pattern = re.compile(r'doi\.org/(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', re.IGNORECASE)
    body_text = soup.get_text()
    match = doi_pattern.search(body_text)
    if match:
        return match.group(1)
    return None

def extract_images_from_url(url, master_output_dir="saved"):
    """
    Core logic to scrape images from a single website URL.
    """
    print(f"Processing URL: {url}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  - Error fetching URL: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    doi = get_url_doi(soup)
    if doi:
        output_dir = os.path.join(master_output_dir, sanitize_filename(doi))
    else:
        url_path = url.split('/')[-1] or url.split('/')[-2]
        output_dir = os.path.join(master_output_dir, sanitize_filename(url_path))
        print(f"  - Warning: DOI not found. Using URL path '{url_path}' for folder.")

    os.makedirs(output_dir, exist_ok=True)
    extracted_count = 0
    
    figures = soup.find_all('figure')
    for fig_index, fig in enumerate(figures):
        img = fig.find('img')
        caption_tag = fig.find('figcaption')
        caption = caption_tag.get_text(strip=True) if caption_tag else ""
        
        if not img or not img.get('src'):
            continue
            
        fig_match = re.search(r'(Figure|Fig\.?|Table)\s*([a-zA-Z0-9.]+)', caption, re.IGNORECASE)
        base_filename = f"{fig_match.group(1).replace('.', '')}_{fig_match.group(2)}" if fig_match else f"figure_{fig_index + 1}"

        img_url = urljoin(url, img.get('src'))
        
        try:
            img_response = requests.get(img_url, headers=headers, timeout=10)
            img_response.raise_for_status()
            
            content_type = img_response.headers.get('content-type', '')
            ext = content_type.split('/')[-1] if '/' in content_type else 'jpg'
            
            image_filename = f"{sanitize_filename(base_filename)}.{ext}"
            image_path = os.path.join(output_dir, image_filename)

            with open(image_path, "wb") as f:
                f.write(img_response.content)
            extracted_count += 1
            print(f"  - Saved {image_path}")

        except requests.exceptions.RequestException as e:
            print(f"  - Failed to download image {img_url}: {e}")

    print(f"  - Finished. Extracted {extracted_count} images into '{output_dir}'.\n")


# --- Main Execution Logic ---

def main():
    """
    Main function to run the script.
    """
    print("Image Extraction Script")
    print("-----------------------")
    print("1. Process local PDF files (from 'pdf_files/' directory)")
    print("2. Process website URLs (from 'website_links.txt' file)")

    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        pdf_dir = "pdf_files"
        if not os.path.isdir(pdf_dir):
            print(f"\nError: The directory '{pdf_dir}' was not found.")
            print("Please create this directory and place your PDF files inside it.")
            return
        
        print(f"\nScanning for PDF files in '{pdf_dir}'...")
        pdf_paths = [
            os.path.join(pdf_dir, f) 
            for f in os.listdir(pdf_dir) 
            if f.lower().endswith(".pdf")
        ]

        if not pdf_paths:
            print(f"  - No PDF files found in '{pdf_dir}'.")
            return
            
        print(f"Found {len(pdf_paths)} PDF file(s). Starting processing...")
        for pdf_path in pdf_paths:
            extract_images_from_pdf(pdf_path)

    elif choice == '2':
        url_input_file = "website_links.txt"
        if not os.path.exists(url_input_file):
            print(f"\nError: The input file '{url_input_file}' was not found.")
            print("Please create it and add website URLs, one per line.")
            return

        print(f"\nStarting website scraping from '{url_input_file}'...")
        with open(url_input_file, "r") as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        
        for url in urls:
            extract_images_from_url(url)

    else:
        print("\nInvalid choice. Please run the script again and enter 1 or 2.")

    print("Script finished.")


if __name__ == "__main__":
    main()