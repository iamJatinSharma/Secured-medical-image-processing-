"""
Module: unzip_dataset.py
Description: Unzips all zip files in the data/ folder automatically.
"""
import os
import zipfile

def unzip_all_in_data(data_dir='data'):
    for file in os.listdir(data_dir):
        if file.endswith('.zip'):
            zip_path = os.path.join(data_dir, file)
            extract_dir = os.path.join(data_dir, file.replace('.zip', ''))
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"Unzipped {file} to {extract_dir}")

if __name__ == "__main__":
    unzip_all_in_data()
