# FCTTD: VGG-16 / CIFAR-10 Reproducibility

This repository provides the model checkpoint, evaluation code, and end-to-end fine-tuning pipeline for the VGG-16/CIFAR-10 experiments reported in the manuscript.

## Model and data

| Item | Configuration |
|---|---|
| Architecture | torchvision VGG-16 (without batch normalization), 10 classes |
| Compressed layer | `classifier.0` (FC1) |
| Tensorization | Input `(16, 32, 49)`; output `(16, 16, 16)` |
| Ranks | Tucker `(8, 8, 8, 8, 8, 8)`; Kruskal `400` |
| FC1 parameters | 20,360 weights + 4,096 biases |
| Data split | CIFAR-10: 45,000 training / 5,000 validation / 10,000 test |
| Split seed | `torch.randperm(50000, generator=torch.Generator().manual_seed(43))`; the first 45,000 indices are used for training |
| Image preprocessing | Train: Resize(256), RandomCrop(224), horizontal flip; validation/test: Resize(256), CenterCrop(224) |
| Normalization | ImageNet mean `(0.485, 0.456, 0.406)`; std `(0.229, 0.224, 0.225)` |

## Checkpoint and evaluation

**Checkpoint:** `vgg16_cifar10_fcttd_rt8_rk400.pt`  
**Test Top-1:** **92.25%**

[Download model checkpoints (OSF)](https://osf.io/prqh2/overview?view_only=3c79b3b49d644c54a64ecbc19267e43a)

Place the checkpoint in `checkpoints/`, then run:

```bash
python -m pip install -r requirements.txt
python prepare_data.py
python evaluate.py --checkpoint checkpoints/vgg16_cifar10_fcttd_rt8_rk400.pt --batch-size 128
```
Earlier auxiliary FC1-only experiments used cached backbone features for validation. The manuscript evaluation uses the complete network directly and does not rely on these cached features.
Evaluation uses the complete model in FP32 without test-time augmentation or cached intermediate features. The standard torchvision CIFAR-10 dataset can also be used if the mirror prepared by `prepare_data.py` is unavailable.

**Checkpoint SHA256:** `28502da1c95833bdd89accdc210747ad5b230a65147f6350635785f19ac2efb2`

## Tests

```bash
python -m unittest discover -s tests -v
```
