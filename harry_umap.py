# This script takes all the images as input and produces a 2D UMAP plot
# of the images. The images are first converted to a 2D array and then UMAP is applied.

import cv2
import numpy as np
import os
from umap import umap_
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import time
import albumentations
import pandas as pd
import argparse
import sys
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder  # Creating instance of one-hot-encoder

# Adding the project directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# # Internal imports
# from models.ResNext50 import Myresnext50
# from train.train_classification_cells import trainer_classification


class Ruby_UMAP:
    def __init__(self, image_dir, labels, **kwargs):
        """
        Initialize the Ruby_UMAP class with parameters for UMAP visualization.

        Args:
            image_dir (str): Directory containing the images.
            labels (list): List of labels corresponding to the images.
            **kwargs: Additional keyword arguments for UMAP parameters and model path.
        """
        self.image_dir = image_dir
        self.labels = labels
        self.model_path = kwargs.get("model_path", None)
        self.model_name = kwargs.get("model_name", "resnext50")
        self.n_neighbors = kwargs.get("n_neighbors", 15)
        self.min_dist = kwargs.get("min_dist", 0.1)
        self.n_components = kwargs.get("n_components", 2)
        self.save_path = kwargs.get("save_path", "umap.png")

        # Define the transformation pipeline for images
        transform_pipeline = albumentations.Compose(
            [
                albumentations.Normalize(
                    mean=(0.5642, 0.5026, 0.6960), std=(0.2724, 0.2838, 0.2167)
                ),
            ]
        )
        self.transform_pipeline = transform_pipeline

    def run(self):
        """
        Run the feature extraction and UMAP embedding process, then save the resulting plot.
        """
        # Load all the npy files in the image directory
        image_features = []
        for image_path in self.image_dir:
            image_feature = np.load(image_path)
            assert image_feature.shape == (
                1000,
            ), f"features shape is {image_feature.shape} instead of (2048,)"
            image_features.append(image_feature)

        image_features = np.array(image_features)
        print(image_features.shape)

        print("Feature extraction done")

        # Apply UMAP to the extracted features
        reducer = umap_.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=self.n_components,
        )
        embedding = reducer.fit_transform(image_features)

        # Save the embedding to a file
        np.save("embedding.npy", embedding)
        print("UMAP done")

        # Create a scatter plot of the UMAP embedding
        sns.set(style="white", context="notebook", rc={"figure.figsize": (14, 10)})
        plt.figure()

        # Generate colors for each unique label
        unique_labels = list(set(self.labels))
        num_labels = len(unique_labels)
        palette = sns.color_palette("tab20", num_labels)
        color_dict = dict(zip(unique_labels, palette))
        colors = [color_dict[label] for label in self.labels]

        # Plot the UMAP embedding with colors
        scatter = plt.scatter(
            embedding[:, 0], embedding[:, 1], c=colors, cmap=cm.tab20, s=10
        )
        plt.colorbar(scatter)
        plt.gca().set_aspect("equal", "datalim")

        # Create a legend for the plot
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=color_dict[label],
                linestyle="",
                markersize=10,
            )
            for label in unique_labels
        ]
        plt.legend(handles, unique_labels, title="Labels", loc="best")

        plt.title("UMAP projection", fontsize=24)

        # Save the plot to a file
        plt.savefig(self.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example script using argparse and **kwargs for UMAP"
    )

    # Load data arguments
    parser.add_argument(
        "--using-meta-data",
        default=False,
        action="store_true",
        help="If you are using meta data",
    )
    parser.add_argument(
        "--meta-data",
        type=str,
        default=os.path.join(
            "/home/harry/Documents/Data/ActivateLearning/proscia_cell/",
            "data_info_256_by_patch_addbenign.csv",
        ),
        help="Path to the meta data",
    )
    parser.add_argument(
        "--using_image_folder",
        default=True,
        action="store_true",
        help="If you are using image folder",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="/Users/neo/Documents/DATA/pooled_deepheme_data_features_frog_v2/test",
        help="Path to the image folder",
    )
    parser.add_argument(
        "--using_image_list",
        default=False,
        action="store_true",
        help="If you are using a list of image dirs",
    )
    parser.add_argument(
        "--image_list", type=str, default=[], help="Path to the image list"
    )

    # Load model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/harry/Documents/codes/PatchML/checkpoints_256_batch0-12-CE-addbenign/model_17_0.9962293741235031.pth",
        help="Path to the model",
    )
    parser.add_argument(
        "--model_name", type=str, default="resnext50", help="Name of the model"
    )

    # UMAP arguments
    parser.add_argument(
        "--n_neighbors", type=int, default=500, help="Number of neighbors"
    )
    parser.add_argument("--min_dist", type=float, default=0.1, help="Minimum distance")
    parser.add_argument(
        "--n_components", type=int, default=2, help="Number of components"
    )

    # Save arguments
    parser.add_argument(
        "--save_path", type=str, default="umap.png", help="Path to save the UMAP plot"
    )

    args = parser.parse_args()

    # Load data based on the provided arguments
    if args.using_meta_data:
        meta_data = pd.read_csv(args.meta_data)
        # Delete the rows with duplicate file paths
        meta_data = meta_data.drop_duplicates(subset="fpath")
        image_dirs = meta_data["fpath"].tolist()
        labels = meta_data["label"].tolist()
        assert len(image_dirs) == len(
            labels
        ), "Length of image dirs and labels should be the same"
    elif args.using_image_folder:
        print(
            "Getting data from image folder, should be in the format of 'label/image.npy'"
        )
        image_dirs = glob.glob(os.path.join(args.image_folder, "*", "*.npy"))
        labels = [
            os.path.basename(os.path.dirname(image_dir)) for image_dir in image_dirs
        ]

        print("Number of images:", len(image_dirs))
        print("Number of labels:", len(labels))

    elif args.using_image_list:
        image_dirs = args.image_list
        labels = [
            os.path.basename(os.path.dirname(image_dir)) for image_dir in image_dirs
        ]
    else:
        raise ValueError("Please provide the data source")

    print("Running UMAP with the following parameters:")
    # Initialize and run the Ruby_UMAP class with the provided arguments
    ruby_umap = Ruby_UMAP(image_dir=image_dirs, labels=labels, **vars(args))
    ruby_umap.run()

    print("UMAP plot saved to:", args.save_path)
