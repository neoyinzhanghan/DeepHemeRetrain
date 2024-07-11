import os
import shutil
import pandas as pd
from tqdm import tqdm


def create_imagenet_structure(metadata_csv, save_dir):
    # Load the metadata CSV
    df = pd.read_csv(metadata_csv)

    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    # Create directories for train, val, test splits
    for split in df["split"].unique():
        split_dir = os.path.join(save_dir, split)
        os.makedirs(split_dir, exist_ok=True)

    # Iterate through the metadata and copy files
    for _, row in tqdm(df.iterrows(), desc="Copying files"):
        src_path = row["fpath"]
        label = row["label"]
        split = row["split"]

        print(type(src_path))
        print(type(label))
        print(type(split))

        import sys

        sys.exit()

        # Create label directory under the respective split
        label_dir = os.path.join(save_dir, split, label)
        os.makedirs(label_dir, exist_ok=True)

        # Copy the file to the new directory
        shutil.copy(src_path, label_dir)


if __name__ == "__main__":
    metadata_csv = "/media/hdd3/neo/AML_metadata.csv"  # Update this path
    save_dir = "/media/hdd3/neo/pooled_aml_deepheme_data"  # Update this path
    if not os.path.exists(metadata_csv):
        print(f"Metadata CSV not found at {metadata_csv}")
    create_imagenet_structure(metadata_csv, save_dir)
