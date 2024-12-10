import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class BrainTumorDataset(Dataset):
    def __init__(self, file_path, transform=None):
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

        return pd.DataFrame({"file_name": file_names, "label": labels})

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