import os

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import imutils
import torchvision


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


import torchvision.transforms.functional


class CropImgTransform:
    def __init__(self, add_pixels=0):
        self.add_pixels = add_pixels

    def __call__(self, img_):
        """
        Finds the extreme points on the image and crops the rectangular out of them
        """
        img = np.array(
            torchvision.transforms.functional.to_pil_image(img_)
        )  # Convert tensor to PIL image and then to numpy array
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        new_img = img[
            extTop[1] - self.add_pixels : extBot[1] + self.add_pixels,
            extLeft[0] - self.add_pixels : extRight[0] + self.add_pixels,
        ].copy()

        return torchvision.transforms.functional.to_tensor(
            new_img
        )  # Convert numpy array back to tensor


# Function to show a grid of images from a single batch
def show_image_grid(images, labels, rows=4, cols=8):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()

    # Display each image in the batch
    for i in range(len(images)):
        img = (
            images[i].numpy().transpose((1, 2, 0))
        )  # Convert to HWC format for plotting
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis("off")  # Hide axes

    # Hide any extra subplots in case there are fewer images than grid cells
    for j in range(len(images), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
