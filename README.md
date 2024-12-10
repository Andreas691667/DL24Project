# Explainable AI for Brain Tumor Detection

This repository contains the implementation and report for our Deep Learning final project: **Explainable AI for Brain Tumor Detection**. The project involves finetuning a Vision Transformer (ViT)-based model for detecting and classifying brain tumors from MRI images. We incorporate Explainable AI (XAI) techniques to provide interpretable insights into the model's predictions, ensuring transparency and trust in its outputs.

## Project Overview

Brain tumor detection is a critical application of AI in healthcare. While deep learning models achieve high accuracy, their complexity can make predictions challenging to interpret. This project combines state-of-the-art machine learning techniques with interpretability methods like Attention Rollout and LIME to address this gap.


## Repository Structure

- `logs/`: Directory containing training logs, validation metrics, and checkpoints for the model.
- `notebooks/`: Notebooks for experimentation, visualizations, and demonstrations.
- `src/`: Source code for dataset loading and preprocessing.
  - `augmentations.py`
  - `config.py`
  - `datasets.py`
  - `transforms.py`
  - `utils.py`
- `report/`: Final report and images used in the report.
- `README.md`: Project description, references, and setup instructions.

#### Changes to Source Code

- In `vision_transformer.py` line 114, the `forward` function of the `EncoderBlock` was modified as follows:
  ```python
  x, _ = self.self_attention(x, x, x, need_weights=True)
  ```
  The `need_weights` parameter has been explicitly set to `True` instead of `False`, such that we could use a hook to retrieve the attention matrices needed for attention rollout.
## Key Features

- **Dataset**: The [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) comprising four classes: glioma, meningioma, pituitary tumors, and no tumors.
- **Model**: Vision Transformer (ViT-B) fine-tuned on MRI images.
- **Explainable AI**:
  - **Attention Rollout**: Aggregates attention maps across layers to visualize the contribution of image patches to predictions.
  - **LIME**: Explains predictions by perturbing image regions and analyzing the impact on model output.

## Results

- **Performance**: 
  - Validation accuracy: **0.97**
  - Test accuracy: **0.97**
- **XAI Validation**: Explanations were compared against a medical student’s annotations for validation.

## Getting Started

### Prerequisites

- Python 3.8 or later
- PyTorch
- PyTorch Lightning
- Optuna
- Albumentations
- LIME

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Andreas691667/DL24Project.git

   cd DL24Project
## Report

The full project report is available in the `report/` directory. It includes detailed descriptions of the methodology, experiments, results, and insights.


## Contributors

* Andreas Kaag Thomsen
* Asger Song Høøck Poulsen
* Emil Hilligsøe Lauritsen
* Niels Viggo Stark Madsen

## References

1. **Rotary Position Embedding for Vision Transformer**  
   Rotary position encoding for transformers.  
   [Paper](https://arxiv.org/html/2403.13298v1)

2. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**  
   The original paper introducing Vision Transformers (ViT).  
   [Paper](https://arxiv.org/abs/2010.11929)

3. **Example on Vision Transformer Implementation**  
   A detailed guide on building Vision Transformers from scratch using PyTorch.  
   [Medium Article](https://medium.com/thedeephub/building-vision-transformer-from-scratch-using-pytorch-an-image-worth-16x16-words-24db5f159e27)

4. **Example on Vision Transformer in PyTorch Lightning**  
   PyTorch Lightning course material for Vision Transformers.  
   [Documentation](https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html#Transformers-for-image-classification)

5. **On Attention Visualization**  
   Video explanation of attention visualization techniques in transformers.  
   [YouTube Video](https://www.youtube.com/watch?v=7q3NGMkEtjI)

6. **Overview of Vision Transformer Explainability Methods**  
   Blog post explaining ViT explainability and visualization methods.  
   [Blog](https://jacobgil.github.io/deeplearning/vision-transformer-explainability)  
   [GitHub](https://github.com/jacobgil/vit-explain/blob/main/vit_rollout.py)

7. **Attention Rollout**  
   Explanation and implementation of attention rollout for ViT:  
    [Paper](https://arxiv.org/pdf/2005.00928)  
    [Medium Article](https://medium.com/@nivonl/exploring-visual-attention-in-transformer-models-ab538c06083a)

8. **Transformer Explainability Beyond Attention Visualization**  
   Insights into explainability techniques for transformers:  
    [Paper](https://arxiv.org/abs/2012.09838)  
    [Medium Article](https://medium.com/orohealth/explainable-ai-using-vision-transformers-on-skin-disease-images-9148d1583faf)

9. **Brain Tumor MRI Dataset**  
   Dataset containing MRI images categorized into glioma, meningioma, pituitary tumors, and no tumors.  
   [Kaggle Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)