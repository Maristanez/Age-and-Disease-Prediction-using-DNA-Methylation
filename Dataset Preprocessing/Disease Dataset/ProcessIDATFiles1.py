import os
import shutil

# Path to your main directory containing subfolders like GSM4649388/
source_dir = "GSE153712_RAW"

# Destination folder where all .idat files will be copied
target_dir = "Age-and-Disease-Prediction-using-DNA-Methylation/Dataset Preprocessing/Disease Dataset/IDAT Files"

# Create target folder if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Walk through all subdirectories
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.lower().endswith(".idat"):
            source_path = os.path.join(root, file)
            target_path = os.path.join(target_dir, file)
            shutil.copy2(source_path, target_path)  # or use move() to delete original

print("All IDAT files have been moved to:", target_dir)