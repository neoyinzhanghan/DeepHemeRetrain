import os
import random
from PIL import Image
from tqdm import tqdm
from train import model_create, model_predict

data_dir = "/media/hdd3/neo/results_dir"

# get the list of all subdirectories in the data directory
all_subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# onlu keep the one that starts with BMA and PBS
result_dirs = [d for d in all_subdirs if "BMA-diff" in d or "PBS-diff" in d]

all_result_dir_paths = [os.path.join(data_dir, d) for d in result_dirs]

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

def get_all_cell_paths(result_dir_path):
    # first get all the subdirectories of result_dir_path/cells
    cell_dir = os.path.join(result_dir_path, "cells")
    all_cells = [d for d in os.listdir(cell_dir) if os.path.isdir(os.path.join(cell_dir, d))]

    # only keep the ones that are in cellnames
    cell_paths = [os.path.join(cell_dir, d) for d in all_cells if d in cellnames]

    all_cell_img_paths = []

    # get all the image paths in each cell path
    for cell_path in cell_paths:
        img_paths = [os.path.join(cell_path, f) for f in os.listdir(cell_path) if f.endswith(".jpg")]
        all_cell_img_paths.extend(img_paths)

    return all_cell_img_paths


num_errors = 0
num_dirs = len(all_result_dir_paths)
non_error_dirs = []

all_cell_paths = []
for result_dir_path in tqdm(all_result_dir_paths, desc="Filtering out error dirs:"):
    # check if the result_dir_path contains a file called "error.txt"
    if not os.path.exists(os.path.join(result_dir_path, "error.txt")):
        non_error_dirs.append(result_dir_path)

        cell_paths = get_all_cell_paths(result_dir_path)
        all_cell_paths.extend(cell_paths)
    else:
        num_errors += 1

print(f"Number of error directories: {num_errors} among {num_dirs} directories.")
print(f"Number of non-error directories: {len(non_error_dirs)}")
print(f"Number of cell image paths: {len(all_cell_paths)}")

# randomly select 100 images from all_cell_paths
randomly_selected_img_paths = random.sample(all_cell_paths, 100)

model_path = "/media/hdd1/neo/MODELS/2024-07-11 Frog Softmax Epochs=50/1/version_0/checkpoints/epoch=49-step=1400.ckpt"
model = model_create(path=model_path)

for image_path in tqdm(randomly_selected_img_paths, desc="Predicting on randomly selected images:"):
    img = Image.open(image_path).convert("RGB")

    model_predict(img, model)