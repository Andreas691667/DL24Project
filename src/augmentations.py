import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import copy


def augment_data(augmentations, file_path, num_augmentations=10, overwrite=True):
    """
    Applies albumentations augmentations to the images in the specified folder.
    Saves them in the same folder with new file names keeping original images (e.g., image1.jpg -> image1_aug0.jpg).
    If overwrite is set to True, all files with 'aug' in their names will be deleted.

    Args:
        augmentations (albumentations.Compose): The augmentation pipeline to apply.
        file_path (str): Path to the folder containing images organized by class.
        overwrite (bool): If True, removes all existing augmented files before augmenting.
    """
    if overwrite:
        deleted_files = 0
        for root, _, files in os.walk(file_path):
            for file in files:
                if "aug" in file:
                    os.remove(os.path.join(root, file))
                    deleted_files += 1
        print(f"Removed {deleted_files} existing augmented images.")

    # Apply augmentations
    for root, _, files in os.walk(file_path):
        augmented_files = 0
        for file_name in tqdm(files, desc=f"Processing images in {root}"):
            if "aug" in file_name: # Skip already augmented files
                continue

            file_path_full = os.path.join(root, file_name)
            img = cv2.imread(file_path_full)

            if img is None:
                print(f"Failed to read image: {file_path_full}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for Albumentations
            file_extension = os.path.splitext(file_name)[1]

            # Generate multiple augmentations for each image
            for i in range(num_augmentations):
                augmented = augmentations(image=img)
                augmented_img = augmented["image"]

                # Save augmented image
                aug_file_name = f"{os.path.splitext(file_name)[0]}_aug{i}{file_extension}"
                aug_file_path = os.path.join(root, aug_file_name)
                cv2.imwrite(aug_file_path, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))
                augmented_files += 1

        print(f"Added {augmented_files} augmented images in {root}.")


def show_augmentations(sample, augmentations):
    """Visualizes the effect of applying a list of augmentations to a sample image."""
    
    # show individual augmentations on the sample in a grid
    # Function to apply augmentation and convert to numpy array
    def apply_augmentation(augmentation, image):
        augmented = augmentation(image=np.array(image))["image"]
        return Image.fromarray(augmented)

    augmentations = copy.deepcopy(augmentations)

    # convert augmentations from A.Compose to list
    if not isinstance(augmentations, list):
        augmentations = augmentations.transforms

     # set all probability to 1 for visualization
    for aug in augmentations:
        aug.p = 1

    # Apply each augmentation to the sample image
    augmented_images = [apply_augmentation(aug, sample) for aug in augmentations]

    # Plot the original and augmented images in a grid
    n_cols = 3
    n_rows = (len(augmented_images) + 1) // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    # Show original image
    axes[0, 0].imshow(sample)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Show augmented images
    for i, aug_img in enumerate(augmented_images):
        row = (i + 1) // n_cols
        col = (i + 1) % n_cols
        axes[row, col].imshow(aug_img)
        axes[row, col].set_title(f"Augmentation {i + 1}")
        axes[row, col].axis("off")

    # Remove empty subplots
    for j in range(len(augmented_images) + 1, n_rows * n_cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.show()