import os
from tqdm import tqdm

data_dir = "/media/hdd3/neo/results_dir"

# get the list of all subdirectories in the data directory
all_subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# onlu keep the one that starts with BMA and PBS
result_dirs = [d for d in all_subdirs if "BMA-diff" in d or "PBS-diff" in d]

all_result_dir_paths = [os.path.join(data_dir, d) for d in result_dirs]


num_errors = 0
num_dirs = len(all_result_dir_paths)
non_error_dirs = []
for result_dir_path in tqdm(all_result_dir_paths, desc="Filtering out error dirs:"):
    # check if the result_dir_path contains a file called "error.txt"
    if not os.path.exists(os.path.join(result_dir_path, "error.txt")):
        non_error_dirs.append(result_dir_path)
    else:
        num_errors += 1

print(f"Number of error directories: {num_errors} among {num_dirs} directories.")
