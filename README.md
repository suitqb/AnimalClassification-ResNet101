# CNN_Animals10 — Animal Image Classification with Transfer Learning

Multi-class image classification across 10 animal categories using ResNet101 transfer learning on the [Animals10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) dataset (~26,000 images).

## Results

| Metric | Mean ± Std (5 runs) |
|---|---|
| Test Accuracy | **98.08% ± 0.07%** |
| Cohen's Kappa | **0.978 ± 0.001** |

> Evaluated on a held-out test set (20% of data), averaged over 5 independent runs for statistical robustness.

## Approach

**Transfer Learning** — ResNet101 pretrained on ImageNet (IMAGENET1K_V2). The convolutional backbone is frozen; only the final fully-connected layer is fine-tuned on the 10-class target task.

This lets the model leverage rich low-level visual features (edges, textures) learned on millions of images, while only adapting the classification head to the new domain — making training fast and sample-efficient.

## Stack

- **PyTorch** — model, training loop, data pipeline
- **torchvision** — ResNet101 pretrained weights, image transforms
- **scikit-learn** — stratified splits, accuracy, Cohen's Kappa, classification report, confusion matrix
- **GPU** — CUDA (training runs on GPU when available)

## Dataset

**Animals10** — 10 classes, ~26,000 images of varying sizes and quality.

| Classes | | | | |
|---|---|---|---|---|
| butterfly | cat | chicken | cow | dog |
| elephant | horse | sheep | spider | squirrel |

Download from Kaggle and extract into `data/`. The original Italian folder names should be renamed to their English equivalents:

```bash
mv data/raw-img/farfalla  data/butterfly
mv data/raw-img/gatto     data/cat
mv data/raw-img/gallina   data/chicken
mv data/raw-img/mucca     data/cow
mv data/raw-img/cane      data/dog
mv data/raw-img/elefante  data/elephant
mv data/raw-img/cavallo   data/horse
mv data/raw-img/pecora    data/sheep
mv data/raw-img/ragno     data/spider
mv data/raw-img/scoiattolo data/squirrel
```

Expected structure:

```
data/
├── butterfly/
├── cat/
├── chicken/
├── cow/
├── dog/
├── elephant/
├── horse/
├── sheep/
├── spider/
└── squirrel/
```

## Training Details

| Hyperparameter | Value |
|---|---|
| Backbone | ResNet101 (frozen) |
| Fine-tuned layers | FC head only |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| LR scheduler | ReduceLROnPlateau (factor=0.1, patience=2) |
| Loss | CrossEntropyLoss |
| Batch size | 64 |
| Epochs | 10 |
| Train / Val / Test split | 60% / 20% / 20% (stratified) |
| Data augmentation | RandomHorizontalFlip, RandomRotation(±10°) |

## Usage

Open and run `src/ResNet.ipynb` in Jupyter. The notebook covers the full pipeline: data loading, model setup, training, evaluation, and visualization (loss curves + confusion matrix).
