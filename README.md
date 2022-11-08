# PyTorch-ViT_2-way-classification-task

The pre-trained model comes from: https://github.com/jeonsworld/ViT-pytorch 

The dataset comes from: https://www.kaggle.com/datasets/gauravduttakiit/ants-bees 

Related documents are located at Google Drive [here](https://drive.google.com/drive/folders/1IAgC6-jFB4F_0LY0_etnp_qlL4qGGPYn?usp=sharing).  

A short report [here](https://drive.google.com/file/d/1D6pGS9TRc73ADaWxaJemCreopW1RGMtK/view?usp=sharing).

The training dashboard is located [here](https://tensorboard.dev/experiment/zMt91UzwQR6uMYaxyZyqOw/#scalars)

## TODO:
1.   [x] Use vectorized L2 distance in attention for **Discriminator**
2.   [x] Overlapping Image Patches
2.   [x] DiffAugment
3.   [x] Self-modulated LayerNorm (SLN)
4.   [x] Implicit Neural Representation for Patch Generation
5.   [x] ExponentialMovingAverage (EMA)
6.   [x] Balanced Consistency Regularization (bCR)
7.   [x] Improved Spectral Normalization (ISN)
8.   [x] Equalized Learning Rate
9.   [x] Weight Modulation

# ViT PyTorch


### Overview
The goal of this task is a bi-directional classification task (i.e. between bees and ants) of the dataset using vit

At the moment, you can easily:
 * Load pretrained ViT models
 * Evaluate on ImageNet or your own data
 * Finetune ViT on your own dataset

_(Upcoming features)_ Coming soon: 
 * Train ViT from scratch on ImageNet (1K)
 * Export to ONNX for efficient inference

### Table of contents
1. [About ViT](#about-vit)
2. [About ViT-PyTorch](#about-vit-pytorch)
3. [Installation](#installation)
4. [Usage](#usage)
    * [Load pretrained models](#loading-pretrained-models)
    * [Example: Classify](#example-classification)
    <!-- * [Example: Extract features](#example-feature-extraction) -->
    <!-- * [Example: Export to ONNX](#example-export) -->
6. [Contributing](#contributing)

### About ViT

Visual Transformers (ViT) are a straightforward application of the [transformer architecture](https://arxiv.org/abs/1706.03762) to image classification. Even in computer vision, it seems, attention is all you need. 

The ViT architecture works as follows: (1) it considers an image as a 1-dimensional sequence of patches, (2) it prepends a classification token to the sequence, (3) it passes these patches through a transformer encoder (like [BERT](https://arxiv.org/abs/1810.04805)), (4) it passes the first token of the output of the transformer through a small MLP to obtain the classification logits. 
ViT is trained on a large-scale dataset (ImageNet-21k) with a huge amount of compute. 

<div style="text-align: center; padding: 10px">
    <img src="https://raw.githubusercontent.com/google-research/vision_transformer/master/figure1.png" width="100%" style="max-width: 300px; margin: auto"/>
</div>


### About ViT-PyTorch

ViT-PyTorch is a PyTorch re-implementation of ViT. It is consistent with the [original Jax implementation](https://github.com/google-research/vision_transformer), so that it's easy to load Jax-pretrained weights.

At the same time, we aim to make our PyTorch implementation as simple, flexible, and extensible as possible.

### Installation

Install with pip:
```bash
pip install pytorch_pretrained_vit
```

Or from source:
```bash
git clone https://github.com/lukemelas/ViT-PyTorch
cd ViT-Pytorch
pip install -e .
```

### Usage

#### Loading pretrained models

Loading a pretrained model is easy:
```python
from pytorch_pretrained_vit import ViT
model = ViT('B_16_imagenet1k', pretrained=True)
```

Details about the models are below:

|    *Name*         |* Pretrained on *|*Finetuned on*|*Available? *|
|:-----------------:|:---------------:|:------------:|:-----------:|
| `B_16`            |  ImageNet-21k   | -            |      ✓      |
| `B_32`            |  ImageNet-21k   | -            |      ✓      |
| `L_16`            |  ImageNet-21k   | -            |      -      |
| `L_32`            |  ImageNet-21k   | -            |      ✓      |
| `B_16_imagenet1k` |  ImageNet-21k   | ImageNet-1k  |      ✓      |
| `B_32_imagenet1k` |  ImageNet-21k   | ImageNet-1k  |      ✓      |
| `L_16_imagenet1k` |  ImageNet-21k   | ImageNet-1k  |      ✓      |
| `L_32_imagenet1k` |  ImageNet-21k   | ImageNet-1k  |      ✓      |

#### Custom ViT

Loading custom configurations is just as easy: 
```python
from pytorch_pretrained_vit import ViT
# The following is equivalent to ViT('B_16')
config = dict(hidden_size=512, num_heads=8, num_layers=6)
model = ViT.from_config(config)
```


### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.

I look forward to seeing what the community does with these models!
