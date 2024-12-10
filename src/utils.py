import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import os

# Function to show a grid of images from a single batch
def show_image_grid(images, labels, rows=4, cols=8):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()

    # Display each image in the batch
    for i in range(len(images)):
        img = images[i].numpy().transpose((1, 2, 0)) # Convert to HWC format for plotting
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis("off") # Hide axes

    # Hide any extra subplots in case there are fewer images than grid cells
    for j in range(len(images), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()