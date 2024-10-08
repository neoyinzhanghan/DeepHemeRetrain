import os
import random
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image


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
