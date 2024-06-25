import os
import glob
import shutil
from tqdm import tqdm
from pathlib import Path

data_path = "/media/hdd1/neo/pooled_deepheme_data"
save_dir = "/media/hdd1/neo/pooled_deepheme_data_M1_confused"

selected_classes = ["M1", "MO2", "M2", "L2", "ER1"]

# Create the directory structure
for split in ["train", "val", "test"]:
    for cellname in selected_classes:
        os.makedirs(os.path.join(save_dir, split, cellname), exist_ok=True)

# Copy the images to the save directory
for split in ["train", "val", "test"]:
    for cellname in tqdm(selected_classes, desc=split):
        files = glob.glob(os.path.join(data_path, split, cellname, "*.jpg"))
        for file in files:
            new_file_path = os.path.join(
                save_dir, split, cellname, os.path.basename(file)
            )
            shutil.copy(file, new_file_path)
