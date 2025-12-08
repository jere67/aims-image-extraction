import os
import time
import random
import math
from seleniumbase import SB
from selenium.common.exceptions import TimeoutException


TOTAL_ARTICLES = 10460
PAGE_SIZE = 30
BASE_SEARCH_URL = "https://www.tandfonline.com/action/doSearch?AllField=nuclear&SeriesKey=tnst20&pageSize=30&sortBy=Earliest_desc"


def download_all_pdfs(base_url):
    """Main handler function for scraping PDFs from TANF."""
    total_pages = math.ceil(TOTAL_ARTICLES / PAGE_SIZE)
    print(f"Calculated {total_pages} pages to process based on {TOTAL_ARTICLES} total articles.")

    with SB(uc=True, headed=True) as sb:
        # Bypass Cloudflare verification
        print("="*60)
        print("DEBUG: Loading initial page and handling Cloudflare challenge...")
        sb.open(f"{base_url}&startPage=0")
        sb.maximize_window()
        print("="*60)
        
        # --- Main Pagination Loop ---
        # This loop will iterate from page 0 to the final page number.
        for page_num in range(total_pages):
            current_start_page = page_num
            page_url = f"{base_url}&startPage={current_start_page}"

            try:
                if page_num > 0:
                    print(f"\nDEBUG: Navigating to Page {page_num + 1}...")
                    sb.open(page_url)
                    
                    # Reset JavaScript event listeners
                    print("DEBUG: Refreshing the page to ensure a clean state...")
                    sb.refresh_page()

                print(f"--- DEBUG: Processing Page {page_num + 1} of {total_pages} (startPage={current_start_page}) ---")
                
                # 1. Wait for the page to be fully interactive after load/refresh, bot bypass
                sb.wait_for_element_clickable("label.markallLabel", timeout=60)
                print("DEBUG: Page content is ready and interactive.")

                # 2. Handle cookie banner on first page
                if page_num == 0:
                    if sb.is_element_visible('button:contains("Accept all")'):
                        print("DEBUG: Cookie banner found. Clicking 'Accept all'.")
                        sb.click('button:contains("Accept all")')
                        time.sleep(2)

                # 3. Select the 30 articles (with the main circle)
                time.sleep(random.uniform(2.0, 3.0))  # Bot detection bypass: brief pause to ensure stability
                sb.js_click('label.markallLabel')
                print("DEBUG: Clicked 'select all'.")

                # 4. Click the 'Download PDFs' button
                time.sleep(random.uniform(1.5, 2.5))
                sb.js_click('a.download-pdfs')
                print("DEBUG: Clicked 'Download PDFs'.")

                # 5. Handle confirmation pop-up
                print("DEBUG: Waiting for confirmation pop-up...")
                sb.wait_for_element_visible("#btn-download-pdfs", timeout=60)
                time.sleep(random.uniform(2.0, 3.5))
                sb.js_click('#btn-download-pdfs')
                print("DEBUG: SUCCESS! Download request sent.")

                # 6. Wait for download to complete
                print("DEBUG: Waiting 15 seconds for the ZIP file to download to the 'downloaded_files' folder...")
                time.sleep(15)

            except Exception as e:
                print(f"DEBUG: FATAL ERROR on page {page_num + 1}. The script will stop.")
                print(f"DEBUG: Error details: {e}")
                break
            
    print("\n--- DEBUG: All tasks finished ---")


if __name__ == "__main__":
    download_all_pdfs(BASE_SEARCH_URL)