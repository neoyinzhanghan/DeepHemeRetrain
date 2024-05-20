import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch import nn
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy, AUROC
import torch.nn.functional as F

############################################################################
####### DEFINE HYPERPARAMETERS AND DATA DIRECTORIES ########################
############################################################################

default_config = {"lr": 3.56e-07}
num_epochs = 500
data_dir = "/media/hdd1/neo/pooled_deepheme_data_SimCLR"
num_gpus = 3
num_workers = 20
num_classes = 23
batch_size = 256

############################################################################
####### DATASET AND DATA MODULE ############################################
############################################################################


def find_empty_directories(directory):
    """
    Recursively find and print all empty directories within a given directory.

    Args:
    directory (str): The path to the directory to start the search from.
    """
    empty_dirs = []
    for root, dirs, files in os.walk(directory, topdown=False):
        # Check if the directory is empty
        if not os.listdir(root):
            empty_dirs.append(root)

    return empty_dirs


class TensorDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_files = [
            os.path.join(data_dir, file)
            for file in os.listdir(data_dir)
            if file.endswith(".pt")
        ]
        self.labels = [
            (
                int(file.split("_")[-1].split(".")[0])
                if file.split("_")[-1].split(".")[0].isdigit()
                else -1
            )
            for file in self.data_files  # This should reference self.data_files, not os.listdir(data_dir)
        ]

        # Debug prints to verify files and labels
        print("Data files found:", self.data_files)
        print("Labels extracted:", self.labels)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        tensor = torch.load(self.data_files[idx])
        label = self.labels[idx]
        tensor = tensor.squeeze()  # Ensure tensor is the correct shape for the model
        return tensor, label


class TensorDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = TensorDataset(os.path.join(self.data_dir, "train"))
        self.val_dataset = TensorDataset(os.path.join(self.data_dir, "val"))
        self.test_dataset = TensorDataset(os.path.join(self.data_dir, "test"))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )


############################################################################
####### MODEL MODULE #######################################################
############################################################################


class FFNModel(pl.LightningModule):
    def __init__(self, num_classes=num_classes, config=default_config):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, num_classes)
        )
        # Simplified task determination
        task = "binary" if num_classes == 2 else "multiclass"
        self.train_accuracy = Accuracy(task=task, num_classes=num_classes)
        self.val_accuracy = Accuracy(task=task, num_classes=num_classes)
        self.train_auroc = AUROC(num_classes=num_classes, task=task)
        self.val_auroc = AUROC(num_classes=num_classes, task=task)
        self.config = config

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        self.train_accuracy(y_hat, y)
        self.train_auroc(y_hat, y)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True)
        self.log("train_auroc", self.train_auroc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_accuracy(y_hat, y)
        self.val_auroc(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=0
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


############################################################################
####### MAIN TRAINING LOOP #################################################
############################################################################


def train_model(batch_size):
    data_module = TensorDataModule(data_dir=data_dir, batch_size=batch_size)
    model = FFNModel(num_classes=num_classes, config=default_config)
    logger = TensorBoardLogger("lightning_logs")
    trainer = pl.Trainer(
        max_epochs=num_epochs, logger=logger, devices=num_gpus, accelerator="gpu"
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":

    empty_dirs = find_empty_directories(data_dir)

    assert not empty_dirs, f"Empty directories found: {empty_dirs}"

    train_model(batch_size=batch_size)
