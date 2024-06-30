import os
import numpy as np
from scripts.train import Myresnext50, model_create
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

model_ckpt_path = "/media/hdd1/neo/MODELS/2024-06-11  DeepHemeRetrain non-frog feature deploy/1/version_0/checkpoints/epoch=499-step=27500.ckpt"
data_dir = "/media/hdd1/neo/pooled_deepheme_data/val"
save_dir = "/media/hdd1/neo/pooled_deepheme_data_features/val"

model = model_create(Myresnext50, model_ckpt_path)

# use model.extract_features() to extract features (don't forget the batch dimension)
# save the extracted features to save_dir under the same name and folder structure as data_dir with .npy extension

for root, dirs, files in os.walk(data_dir):
    for file in tqdm(files):
        # check if the file is an image
        if file.endswith(".png") or file.endswith(".jpg"):
            img = Image.open(os.path.join(root, file)).convert("RGB")

            # transform image to 96x96 and tensor
            img = transforms.Resize((96, 96))(img)
            img = transforms.ToTensor()(img)

            # add batch dimension
            img = img.unsqueeze(0)

            # extract features
            features = model.extract_features(img)

            # REMOVE BATCH DIMENSION and assert that the image has shape (2048,)
            features = features.squeeze()
            assert features.shape == (
                2048,
            ), f"features shape is {features.shape} instead of (2048,)"

            # save features
            save_path = os.path.join(save_dir, root[len(data_dir) :], file)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            np.save(save_path, features)
