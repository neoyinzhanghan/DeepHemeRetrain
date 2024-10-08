import ray

import os
import time
import random
from PIL import Image
from tqdm import tqdm
from train import model_create, model_predict
from ensemble_manager import EnsembleManager


def create_list_of_batches_from_list(list, batch_size):
    """
    This function creates a list of batches from a list.

    :param list: a list
    :param batch_size: the size of each batch
    :return: a list of batches

    >>> create_list_of_batches_from_list([1, 2, 3, 4, 5], 2)
    [[1, 2], [3, 4], [5]]
    >>> create_list_of_batches_from_list([1, 2, 3, 4, 5, 6], 3)
    [[1, 2, 3], [4, 5, 6]]
    >>> create_list_of_batches_from_list([], 3)
    []
    >>> create_list_of_batches_from_list([1, 2], 3)
    [[1, 2]]
    """

    list_of_batches = []

    for i in range(0, len(list), batch_size):
        batch = list[i : i + batch_size]
        list_of_batches.append(batch)

    return list_of_batches


if __name__ == "__main__":
    import doctest

    doctest.testmod()


data_dir = "/media/hdd3/neo/results_dir"

# get the list of all subdirectories in the data directory
all_subdirs = [
    d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
]

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
    all_cells = [
        d for d in os.listdir(cell_dir) if os.path.isdir(os.path.join(cell_dir, d))
    ]

    # only keep the ones that are in cellnames
    cell_paths = [os.path.join(cell_dir, d) for d in all_cells if d in cellnames]

    all_cell_img_paths = []

    # get all the image paths in each cell path
    for cell_path in cell_paths:
        img_paths = [
            os.path.join(cell_path, f)
            for f in os.listdir(cell_path)
            if f.endswith(".jpg")
        ]
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
randomly_selected_img_paths = random.sample(all_cell_paths, 10000)

model_path = "/media/hdd3/neo/MODELS/2024-06-11  DeepHemeRetrain non-frog feature deploy/1/version_0/checkpoints/epoch=499-step=27500.ckpt"
model = model_create(path=model_path)

start_time = time.time()
all_pil_images = [Image.open(img_path) for img_path in randomly_selected_img_paths]

all_pil_images_batches = create_list_of_batches_from_list(all_pil_images, 100)

end_time = time.time()
image_loading_time = end_time - start_time

start_time = time.time()

ray.shutdown()
ray.init()

# start 2 managers
managers = [EnsembleManager.remote(model_path) for _ in range(2)]

tasks = {}
all_results = []

# assign the batches to the managers
for i, batch in enumerate(all_pil_images_batches):
    manager = managers[i % 2]
    task = manager.async_predict_images_batch.remote(batch)
    tasks[task] = batch

with tqdm(
    total=len(all_pil_images), desc="Predicting using Ensemble Models in Parallel"
) as pbar:
    while tasks:
        done_ids, _ = ray.wait(list(tasks.keys()))

        for done_id in done_ids:
            try:
                batch_size, batch_results = ray.get(done_id)
                all_results.extend(batch_results)

                pbar.update(batch_size)
            except Exception as e:
                print(f"Task {done_id} failed with error: {e}")

            del tasks[done_id]

ray.shutdown()

end_time = time.time()
prediction_time = end_time - start_time

print(f"Total number of images: {len(all_pil_images)}")
print(f"Image loading time: {image_loading_time}")
print(f"Prediction time: {prediction_time}")
