import urllib.request
import zipfil
import os
import shutil
import sys

# --- CONFIGURATION ---
# TODO: Update this link with real one
DATA_URL = "https://zenodo.org/record/YOUR_RECORD_ID/files/processed_data.zip?download=1"
DESTINATION = "./data/processed"

def download_and_extract():
    # 1. Check if URL is still the placeholder
    if "YOUR_RECORD_ID" in DATA_URL:
        print("Error: You must update the DATA_URL in download_data.py with your actual Zenodo link.")
        sys.exit(1)

    # 2. Setup Directory
    if not os.path.exists(DESTINATION):
        print(f"Creating directory: {DESTINATION}")
        os.makedirs(DESTINATION, exist_ok=True)
    
    zip_path = os.path.join(DESTINATION, "data.zip")
    
    # 3. Download
    print(f"Downloading data from Zenodo...")
    try:
        # Add a simple progress reporter
        def report(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = downloaded * 100 / total_size
                print(f"\rProgress: {percent:.1f}%", end="")
            else:
                print(f"\rDownloaded: {downloaded} bytes", end="")

        urllib.request.urlretrieve(DATA_URL, zip_path, reporthook=report)
        print("\nDownload complete.")
        
    except Exception as e:
        print(f"\nFailed to download data: {e}")
        sys.exit(1)
    
    # 4. Extract
    print("Extracting files...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DESTINATION)
        print("Extraction complete.")
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid zip file.")
        sys.exit(1)
        
    # 5. Cleanup
    os.remove(zip_path)
    print(f"Setup successful! Data is ready in {DESTINATION}")

if __name__ == "__main__":
    download_and_extract()
