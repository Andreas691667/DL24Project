import os

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt


TEST_DATA_PATH = os.path.join(
    kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset"), "Testing/"
)
TRAIN_DATA_PATH = os.path.join(
    kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset"), "Training/"
)


class BrainTumorDataset(Dataset):
    def __init__(self, file_path=TEST_DATA_PATH, transform=None):
        self.df = self.__load_to_df(file_path)
        self.file_path = file_path
        self.transform = transform

        # Create mappings between class names and indices
        self.unique_labels = np.sort(self.df.iloc[:, 1].unique())
        self.class_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.idx_to_class = {idx: label for label, idx in self.class_to_idx.items()}

    # load data to dataframe with file names and labels
    def __load_to_df(self, file_path):
        file_names = []
        labels = []
        folders = os.listdir(file_path)
        for folder in folders:
            files = os.listdir(file_path + folder)
            for file in files:
                file_names.append(file)
                labels.append(folder)

        df = pd.DataFrame({"file_name": file_names, "label": labels})
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]
        img = Image.open(self.file_path + label + "/" + file_name)

        if self.transform:
            img = self.transform(img)

        label = self.class_to_idx[label] if isinstance(label, str) else int(label)

        return img, label


# Function to show a grid of images from a single batch
def show_image_grid(images, labels, rows=4, cols=8):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()
    
    # Display each image in the batch
    for i in range(len(images)):
        img = images[i].numpy().transpose((1, 2, 0))  # Convert to HWC format for plotting
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis("off")  # Hide axes

    # Hide any extra subplots in case there are fewer images than grid cells
    for j in range(len(images), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()