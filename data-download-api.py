import os
import kagglehub

# Define the target directory
data_folder = "data"

# Ensure the data folder exists
os.makedirs(data_folder, exist_ok=True)

# Function to download and save datasets
def download_and_save(dataset_name, folder):
    path = kagglehub.dataset_download(dataset_name, path=folder)
    print(f"Dataset '{dataset_name}' saved to: {path}")

# List of datasets to download
datasets = [
    "uwrfkaggler/ravdess-emotional-speech-audio",
    "ejlok1/cremad",
    "barelydedicated/savee-database",
    "ejlok1/toronto-emotional-speech-set-tess",
]

# Download datasets
for dataset in datasets:
    download_and_save(dataset, data_folder)
