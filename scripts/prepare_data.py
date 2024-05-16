import os
import shutil
import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import Image

data_dirs = [
    "/media/ssd2/dh_labelled_data/DeepHeme1/UCSF_repo",
    "/media/ssd2/dh_labelled_data/DeepHeme1/MSK_repo_normal",
    "/media/ssd2/dh_labelled_data/DeepHeme1/MSK_repo_mixed",
    "/media/ssd2/dh_labelled_data/DeepHeme2/PBS/labelled_cartridge_2",
    "/media/ssd2/dh_labelled_data/DeepHeme2/PBS/labelled_cartridge_1",
    "/media/ssd2/dh_labelled_data/DeepHeme2/PBS/labelled_cartridge_0",
    "/media/ssd2/dh_labelled_data/DeepHeme2/BMA/cartridge_1",
    "/media/hdd3/neo/LabelledBMASkippocytes",
    "/media/hdd1/neo/blasts_normal_confirmed",
]

save_dir = "/media/hdd3/neo/pooled_deepheme_data"

cellnames = [
    "B1",
    "B2",
    "E1",
    "E4",
    "ER1",
    "ER2",
    "ER3",
    "ER4",
    "ER5",
    "ER6",
    "L2",
    "L4",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "MO2",
    "PL2",
    "PL3",
    "U1",
    "U4",
]

# The purpose of this script is to pool the classification data from these folders into a single folder
# This will allow us to train a model on the pooled data

# First, we need to create the directory structure
# The directory structure will be as follows:
# pooled_deepheme_data
# ├── train
# │   ├── B1
# │   ├── B2
# │   ├── E1
# │   ├── ...
# ├── val
# │   ├── B1
# │   ├── B2
# │   ├── E1
# │   ├── ...
# ├── test
# │   ├── B1
# │   ├── B2
# │   ├── E1
# │   ├── ...

# if there is a class that contains the words "cell detection error", it goes into U1 (upper lower case should not matter)
# if there is a class that contains the words "Skippocyte" or "Skiptocyte", it goes into U1 (upper lower case should not matter)
# all the classes that contain the words "others", but not "cell detection error" or "Skippocyte" or "Skiptocyte" are skipped

# the images copied to the save directory should be renamed 0.jpg, 1.jpg, 2.jpg, etc.
# and we use a metadata file to keep track of the original image path

# we will use a 70-15-15 split for train-val-test

# create the directory structure
for split in ["train", "val", "test"]:
    for cellname in cellnames:
        os.makedirs(os.path.join(save_dir, split, cellname), exist_ok=True)

# copy the images
metadata = {
    "idx": [],
    "original_path": [],
    "split": [],
}

current_idx = 0

for data_dir in data_dirs:
    print(f"Processing {data_dir}")

    # get a list of all subdirectories in the data_dir
    subdirs = [
        f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))
    ]

    for subdir in tqdm(subdirs, desc="Processing Subdirectories"):

        # if the name of the subdir contains the word "Skippo" or "Skiptocyte", it goes into U1
        if "skippo" in subdir.lower() or "skiptocyte" in subdir.lower():
            cellname = "U1"
        # if the name of the subdir contains the word "cell detection error", it goes into U1
        elif "cell detection error" in subdir.lower():
            cellname = "U1"
        elif "others" in subdir.lower() or "other" in subdir.lower():
            continue
        elif "U2" in subdir:
            continue
        else:
            cellname = subdir

        # get a list of all jpg files in the subdir
        img_files = [
            f
            for f in os.listdir(os.path.join(data_dir, subdir))
            if f.lower().endswith(".jpg") or f.lower().endswith(".png")
        ]

        # if there are no jpg files, skip
        if len(img_files) == 0:
            continue

        for i, jpg_file in tqdm(enumerate(img_files), desc="Copying Images"):
            # randomly assign the image to train, val or test using random numbers
            rand = np.random.rand()
            if rand < 0.7:
                split = "train"
            elif rand < 0.85:
                split = "val"
            else:
                split = "test"

            # open the image using PIL, if the image is RGBA conver to RGB and then save a jpg file

            image_pil = Image.open(os.path.join(data_dir, subdir, jpg_file))
            if image_pil.mode == "RGBA":
                image_pil = image_pil.convert("RGB")
            image_pil.save(
                os.path.join(save_dir, split, cellname, f"{current_idx}.jpg")
            )

            metadata["idx"].append(current_idx)
            metadata["original_path"].append(os.path.join(data_dir, subdir, jpg_file))
            metadata["split"].append(split)

            current_idx += 1

metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv(os.path.join(save_dir, "metadata.csv"), index=False)

# print the number of images in each split and as well as the number of images in each class
print("Number of images in each split and class:")
for split in ["train", "val", "test"]:
    print(f"Split: {split}")
    for cellname in cellnames:
        num_images = len(os.listdir(os.path.join(save_dir, split, cellname)))
        print(f"{cellname}: {num_images}")
    print()

print("Data preparation complete.")
