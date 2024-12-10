import os
import kagglehub

TEST_DATA_PATH = os.path.join(
    kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset"), "Testing/"
)
TRAIN_DATA_PATH = os.path.join(
    kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset"), "Training/"
)