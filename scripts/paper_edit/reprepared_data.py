import os
import pandas
import random
import shutil
from tqdm import tqdm

original_data_dir = "/media/hdd3/neo/pooled_deepheme_data"
original_metadata_dir = "/media/hdd3/neo/pooled_deepheme_data/metadata.csv"
new_data_dir = "/media/hdd3/neo/pooled_deepheme_data_MSK_to_UCSF"

os.makedirs(new_data_dir, exist_ok=True)
os.makedirs(os.path.join(new_data_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(new_data_dir, "val"), exist_ok=True)
os.makedirs(os.path.join(new_data_dir, "test"), exist_ok=True)

metadata = pandas.read_csv(original_metadata_dir)

train_prob = 0.9
val_prob = 0.1

new_metadata_dict = {
    "original_path": [],
    "old_image_path": [],
    "split": [],
    "cell_class": [],
    "idx": [],
    "institution": [],
    "new_image_path": [],
}

# iterate over rows of the metadata
for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
    # get the image path
    record_path = row["original_path"]
    split = row["split"]
    idx = row["idx"]

    # based name of the image
    base_name = os.path.basename(record_path)

    # get the immediate parent directory name
    cell_class = os.path.basename(os.path.dirname(record_path))
    image_name = f"{idx}.jpg"
    image_path = os.path.join(original_data_dir, split, cell_class, image_name)

    os.makedirs(os.path.join(new_data_dir, split, cell_class), exist_ok=True)

    if "UCSF_repo" in record_path:
        institution = "UCSF"
    elif "MSK_repo" in record_path:
        institution = "MSK"
    else:
        continue  # skip data that is not from UCSF or MSK repo

    if institution == "MSK":
        new_split = random.choices(["train", "val"], [train_prob, val_prob])[0]

    elif institution == "UCSF":
        new_split = "test"

    # copy the image to the new directory under the new split
    new_image_path = os.path.join(new_data_dir, new_split, cell_class)

    shutil.copy(image_path, new_image_path)

    new_metadata_dict["original_path"].append(record_path)
    new_metadata_dict["old_image_path"].append(image_path)
    new_metadata_dict["split"].append(new_split)
    new_metadata_dict["cell_class"].append(cell_class)
    new_metadata_dict["idx"].append(idx)
    new_metadata_dict["institution"].append(institution)
    new_metadata_dict["new_image_path"].append(new_image_path)

new_metadata = pandas.DataFrame(new_metadata_dict)
new_metadata.to_csv(os.path.join(new_data_dir, "metadata.csv"), index=False)
print("Done!")
