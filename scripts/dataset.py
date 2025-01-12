import os
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm


def create_plasma_cell_dataset_metadata(base_data_csv, plasma_cell_data_dir):
    """The base_metadata_csv should have columns idx,original_path,split
    The original_path should be the format '/media/ssd2/dh_labelled_data/DeepHeme1/UCSF_repo/ER2/45823.png'
    The immediate directory name should be the label
    """
    base_metadata = pd.read_csv(base_data_csv)
    # remove the split column
    base_metadata = base_metadata.drop(columns=["split"])

    # remove the idx column
    base_metadata = base_metadata.drop(columns=["idx"])

    # create the label column
    base_metadata["label"] = base_metadata["original_path"].apply(
        lambda x: x.split("/")[-2]
    )

    # create a data_group column valued "original"
    base_metadata["data_group"] = "original"

    plasma_cell_jpgs = []
    plasma_cell_labels = []
    plasma_cell_data_group = []

    # recursively find all the jpg files in the plasma_cell_data_dir
    print("Finding plasma cell images...")
    for root, dirs, files in tqdm(
        os.walk(plasma_cell_data_dir),
        desc="Finding plasma cell images",
        total=len(os.walk(plasma_cell_data_dir)),
    ):
        for file in files:
            if file.endswith(".jpg"):
                plasma_cell_jpgs.append(os.path.join(root, file))
                plasma_cell_labels.append("L4")
                plasma_cell_data_group.append(
                    "plasma_cells_from_high_plasma_cell_slides"
                )

    print("Creating plasma cell metadata...")
    # add the plasma_cell_jpgs to the dataframe's original_path, label should be "L4"
    plasma_cell_metadata = pd.DataFrame(
        {
            "original_path": plasma_cell_jpgs,
            "label": plasma_cell_labels,
            "data_group": plasma_cell_data_group,
        }
    )

    # concatenate the base_metadata and plasma_cell_metadata
    combined_metadata = pd.concat([base_metadata, plasma_cell_metadata])

    print("Assigning train/val/test splits...")
    # now randomly assign train, val, test split based on a 0.7, 0.15, 0.15 proportion
    # Create a new column for the split
    combined_metadata["split"] = np.random.choice(
        ["train", "val", "test"], size=len(combined_metadata), p=[0.7, 0.15, 0.15]
    )

    # save the combined_metadata to a csv
    combined_metadata.to_csv("combined_metadata.csv", index=False)

    return combined_metadata


class CustomDataset(Dataset):
    def __init__(
        self,
        base_data_dir,
        results_dirs_list,
        cell_types_list,
        base_data_sample_probability,
        sample_probabilities,
        transform=None,
    ):
        self.base_data_dir = base_data_dir
        self.results_dirs_list = results_dirs_list
        self.cell_types_list = cell_types_list
        self.base_data_sample_probability = base_data_sample_probability
        self.sample_probabilities = sample_probabilities
        self.transform = transform

        # Load the base dataset
        self.base_dataset = ImageFolder(base_data_dir)
        self.base_class_indices = {
            cls: [] for cls in range(len(self.base_dataset.classes))
        }
        for idx, (_, label) in enumerate(self.base_dataset.samples):
            self.base_class_indices[label].append(idx)

        self.num_base_data = len(self.base_dataset)
        self.num_data_points = 2 * self.num_base_data

    def sample_from_base_dir(self):
        # Class balancing: Choose a random class
        class_choice = random.choice(list(self.base_class_indices.keys()))
        # Randomly sample from that class
        sample_idx = random.choice(self.base_class_indices[class_choice])
        image, label = self.base_dataset.samples[sample_idx]
        image = Image.open(image)
        return image, label

    def sample_from_result_dirs(self):
        # Choose which result directory to sample from
        result_dir_choice = random.choices(
            self.results_dirs_list, self.sample_probabilities
        )[0]
        corresponding_cell_type = self.cell_types_list[
            self.results_dirs_list.index(result_dir_choice)
        ]

        # Sample a subdirectory
        sub_dirs = [
            d
            for d in os.listdir(result_dir_choice)
            if os.path.isdir(os.path.join(result_dir_choice, d))
        ]
        sub_dir_choice = random.choice(sub_dirs)

        # Sample a jpg file from the corresponding cell type folder
        cell_dir = os.path.join(
            result_dir_choice, sub_dir_choice, "cells", corresponding_cell_type
        )
        if os.path.exists(cell_dir):
            jpg_files = [f for f in os.listdir(cell_dir) if f.endswith(".jpg")]
            if jpg_files:
                image_file = random.choice(jpg_files)
                image_path = os.path.join(cell_dir, image_file)
                image = Image.open(image_path)
                label = self.base_dataset.class_to_idx[
                    corresponding_cell_type
                ]  # Assuming the cell types map to base classes
                return image, label
        return None, None

    def __len__(self):
        return self.num_data_points

    def __getitem__(self, index):
        # Decide whether to sample from base data or result directories
        if random.random() < self.base_data_sample_probability:
            image, label = self.sample_from_base_dir()
        else:
            image, label = None, None
            while image is None:
                image, label = self.sample_from_result_dirs()

        if self.transform:
            image = self.transform(image)

        return image, label


class CustomPlasmaCellDataset(Dataset):
    def __init__(
        self,
        base_data_dir,
        results_dirs_list,
        cell_types_list,
        base_data_sample_probability,
        sample_probabilities,
        transform=None,
    ):
        self.base_data_dir = base_data_dir
        self.results_dirs_list = results_dirs_list
        self.cell_types_list = cell_types_list
        self.base_data_sample_probability = base_data_sample_probability
        self.sample_probabilities = sample_probabilities
        self.transform = transform

        # Load the base dataset
        self.base_dataset = ImageFolder(base_data_dir)
        self.base_class_indices = {
            cls: [] for cls in range(len(self.base_dataset.classes))
        }
        for idx, (_, label) in enumerate(self.base_dataset.samples):
            self.base_class_indices[label].append(idx)

        self.num_base_data = len(self.base_dataset)
        self.num_data_points = 2 * self.num_base_data

    def sample_from_base_dir(self):
        # Class balancing: Choose a random class
        class_choice = random.choice(list(self.base_class_indices.keys()))
        # Randomly sample from that class
        sample_idx = random.choice(self.base_class_indices[class_choice])
        image, label = self.base_dataset.samples[sample_idx]
        image = Image.open(image)
        return image, label

    def sample_from_result_dirs(self):
        # Choose which result directory to sample from
        result_dir_choice = random.choices(
            self.results_dirs_list, self.sample_probabilities
        )[0]
        corresponding_cell_type = self.cell_types_list[
            self.results_dirs_list.index(result_dir_choice)
        ]

        # Sample a subdirectory
        sub_dirs = [
            d
            for d in os.listdir(result_dir_choice)
            if os.path.isdir(os.path.join(result_dir_choice, d))
        ]
        sub_dir_choice = random.choice(sub_dirs)

        # Sample a jpg file from the corresponding cell type folder
        cell_dir = os.path.join(
            result_dir_choice, sub_dir_choice, "cells", corresponding_cell_type
        )
        if os.path.exists(cell_dir):
            jpg_files = [f for f in os.listdir(cell_dir) if f.endswith(".jpg")]
            if jpg_files:
                image_file = random.choice(jpg_files)
                image_path = os.path.join(cell_dir, image_file)
                image = Image.open(image_path)
                label = self.base_dataset.class_to_idx[
                    corresponding_cell_type
                ]  # Assuming the cell types map to base classes
                return image, label
        return None, None

    def __len__(self):
        return self.num_data_points

    def __getitem__(self, index):
        # Decide whether to sample from base data or result directories
        if random.random() < self.base_data_sample_probability:
            image, label = self.sample_from_base_dir()
        else:
            image, label = None, None
            while image is None:
                image, label = self.sample_from_result_dirs()

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    base_data_csv = "/media/hdd3/neo/pooled_deepheme_data/metadata.csv"
    plasma_cell_data_dir = "/media/ssd2/dh_labelled_data/DeepHeme1/UCSF_repo/ER2"
    save_path = "/media/hdd3/neo/new_plasma_cell_deepheme_training_metadata"

    combined_metadata = create_plasma_cell_dataset_metadata(
        base_data_csv, plasma_cell_data_dir
    )

    # save the combined_metadata to a csv
    combined_metadata.to_csv(
        os.path.join(save_path, "new_plasma_cell_deepheme_training_metadata.csv"),
        index=False,
    )
