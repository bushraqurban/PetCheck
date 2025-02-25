import zipfile
from pathlib import Path

# Define paths
zip_file_path = Path('data/raw/dataset.zip')
extracted_dir = Path('data/raw')

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)

print(f"Data extracted successfully into {extracted_dir}")
