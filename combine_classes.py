import os
import glob
import shutil
from tqdm import tqdm
import pandas as pd
from pathlib import Path

data_path = "/media/hdd1/neo/pooled_deepheme_data_M1_confused"
save_dir = "/media/hdd1/neo/pooled_deepheme_data_M1_confused_blast_binary"
metadata_file = os.path.join(save_dir, "metadata.csv")

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# Define the classes of interest
target_class = "M1"
not_target_class = "not-M1"

# Create directories for train, val, and test sets
sets = ["train", "val", "test"]
for set_type in sets:
    os.makedirs(os.path.join(save_dir, set_type, target_class), exist_ok=True)
    os.makedirs(os.path.join(save_dir, set_type, not_target_class), exist_ok=True)

# Initialize a list to store metadata
metadata = []


# Function to copy images to the respective folders and record metadata
def copy_images(set_type):
    set_path = os.path.join(data_path, set_type)
    class_folders = glob.glob(os.path.join(set_path, "*"))

    for class_folder in tqdm(class_folders, desc=f"Processing {set_type} set classes"):
        class_name = os.path.basename(class_folder)
        image_files = glob.glob(os.path.join(class_folder, "*"))

        if class_name == target_class:
            target_folder = os.path.join(save_dir, set_type, target_class)
        else:
            target_folder = os.path.join(save_dir, set_type, not_target_class)

        for image_file in tqdm(
            image_files, desc=f"Copying {class_name} images in {set_type} set"
        ):
            shutil.copy(image_file, target_folder)
            metadata.append(
                {
                    "original_class": class_name,
                    "original_path": image_file,
                    "new_path": os.path.join(
                        target_folder, os.path.basename(image_file)
                    ),
                }
            )


# Copy images for each set
for set_type in tqdm(sets, desc="Processing sets"):
    copy_images(set_type)

# Create a DataFrame from the metadata and save it as a CSV file
metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv(metadata_file, index=False)

print("Dataset preparation complete. Metadata saved to", metadata_file)
