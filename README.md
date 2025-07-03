# Face recognition - extended experiment

This project implements a Siamese Neural Network in PyTorch using a pretrained ResNet-18 as a feature extractor. The model learns to predict whether two input images belong to the same class or not. It represents the first of two tasks completed as part of my Master's thesis.

## Overview

- Uses `torchvision.models.resnet18` as the base CNN encoder.
- Trains on image pairs generated from class labels using a custom `PairDataset`.
- Evaluates precision, recall, and F1-score across various decision thresholds.
- Includes visualizations of image pairs with predicted similarity scores.

## Dataset Structure

The data should follow the structure expected by `torchvision.datasets.ImageFolder`:

```
dataset/
├── class_0/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_1/
│   ├── image1.jpg
│   └── ...
└── ...
```


## Outputs & Visualizations
The script produces:

- Training loss plot over epochs

- Precision, Recall, and F1-score vs. threshold line chart

- Summary table of evaluation metrics

- Bar chart comparing metric averages

- Grid of image pairs with predicted similarity scores, highlighting false negatives

## Code Structure
PairDataset: Custom dataset that generates matching/non-matching image pairs

SimpleNetwork: Siamese network architecture using ResNet-18

train(): Model training function

evaluate(): Evaluation function that computes accuracy and gathers predictions

run_pipeline(): Full training + evaluation loop

Visualization functions for:

Loss plots

Metric trends

Bar charts

Sample image pairs

## 📌 Notes
The model uses BCEWithLogitsLoss and outputs a similarity score between 0 and 1.

Thresholds for classification can be tuned; default evaluation runs for thresholds between 0.01 and 0.4.

Pair selection always includes one "anchor" image per class (the first in that class), and randomly selected positive or negative examples.

## Results

