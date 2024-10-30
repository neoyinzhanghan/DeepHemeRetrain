import os
import torch
import pytorch_lightning as pl
import pandas as pd
from torchvision import transforms, datasets
from torchmetrics.classification import Precision, Recall, F1Score, AUROC
from torch.utils.data import DataLoader

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

from train_frog import (
    Myresnext50,
    ImageDataModule,
)  # Replace with the name of your training script

# Set up hyperparameters and data paths
data_dir = "/media/hdd3/neo/pooled_deepheme_data"
checkpoint_path = "/home/greg/Documents/neo/DeepHemeRetrain/lightning_logs/1/version_0/checkpoints/epoch=499-step=21000.ckpt"  # Set your checkpoint path here
batch_size = 512
num_classes = 23
num_workers = 36
downsample_factor = 1  # Set as per your requirements


# Define a function to perform testing and save results to CSV
def test_model(checkpoint_path):
    # Instantiate the data module
    data_module = ImageDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        downsample_factor=downsample_factor,
    )

    # Load the model from checkpoint
    model = Myresnext50.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
    model.eval()  # Set the model to evaluation mode
    model.to("cuda")  # Move the model to the GPU

    # Metrics
    precision = Precision(num_classes=num_classes, average=None, task="multiclass").to(
        "cuda"
    )
    recall = Recall(num_classes=num_classes, average=None, task="multiclass").to("cuda")
    f1_score = F1Score(num_classes=num_classes, average=None, task="multiclass").to(
        "cuda"
    )
    auc = AUROC(num_classes=num_classes, average=None, task="multiclass").to("cuda")

    # Dataloader for testing
    test_loader = data_module.test_dataloader()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images, labels = images.to("cuda"), labels.to("cuda")

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)  # Get probabilities for AUC
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds)
            all_labels.append(labels)
            all_probs.append(probs)

    # Concatenate all predictions, labels, and probabilities
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    # Compute metrics
    precision_values = precision(all_preds, all_labels).cpu().numpy()
    recall_values = recall(all_preds, all_labels).cpu().numpy()
    f1_values = f1_score(all_preds, all_labels).cpu().numpy()
    auc_values = auc(all_probs, all_labels).cpu().numpy()

    # Save metrics to CSV
    metrics_df = pd.DataFrame(
        {
            "Class": cellnames,
            "Precision": precision_values,
            "Recall": recall_values,
            "F1-Score": f1_values,
            "AUC": auc_values,
        }
    )

    metrics_df.to_csv("test_metrics.csv", index=False)
    print("Test metrics saved to 'test_metrics.csv'")


if __name__ == "__main__":
    test_model(checkpoint_path)
