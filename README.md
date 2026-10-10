# FCTTD VGG-16 / CIFAR-10 reproducibility release

This repository contains the runnable compression, fine-tuning, and evaluation pipeline for the VGG-16/CIFAR-10 FC1 experiments. It is intended to accompany the reviewer response and to make the Table VIII rank sweep independently auditable.

## What is included

* `fcttd/layers.py`: Tucker/HOSVD and CP-ALS (Kruskal) decomposition utilities and the `TuckerKruskalLinear` layer.
* `train_vgg16_cifar10.py`: CIFAR-10 preprocessing, optional FC1 replacement, fine-tuning, checkpointing, and evaluation.
* `decompose_fc1.py`: compress an existing dense VGG-16 checkpoint without training.
* `evaluate.py`: deterministic CIFAR-10 test evaluation for dense or FCTTD checkpoints.
* `run_table_viii.py` and `configs/table_viii.json`: the five `R_k` settings and seeds used for the released sweep.
* `checkpoints/`: the original dense checkpoint and the best released FCTTD checkpoint.
* `requirements.txt` and `environment.yml`: pinned software dependencies.

## Efficient FCTTD forward

The forward pass never reconstructs a dense FC1 matrix. For each mode it forms the effective factor `U_k @ C_k`, builds the output and input Khatri--Rao factors, and applies two small matrix products:

```text
Z = X @ KR(input factors)
Y = Z @ KR(output factors)^T + bias
```

The dense `W_hat = KR(output) @ KR(input)^T` exists only in the diagnostic `reconstruct_weight()` method; `forward()` does not call it. The included unit test monkey-patches that diagnostic to ensure the fast path remains contraction-only.

## Installation

```bash
conda env create -f environment.yml
conda activate fcttd-vgg16-cifar10
# or: python -m pip install -r requirements.txt
```

PyTorch CUDA wheels should be selected according to the target GPU. The listed versions are the versions used to create the released checkpoints.

## Checkpoints

`checkpoints/vgg16_cifar10_dense_best.pt` is the clean dense VGG-16 checkpoint (94.22% test Top-1 under the clean reference protocol). `checkpoints/vgg16_cifar10_fcttd_r400_seed43.pt` replaces only `classifier.0` by FCTTD (`R_t=(8,8,8,8,8,8)`, `R_k=400`) and reports 90.68% in the matched-control FC1-only fine-tuning protocol. SHA256 values are in `checkpoints/checkpoint_manifest.json`.

The compressed checkpoint is deliberately accompanied by its protocol metadata. It is not claimed to reproduce every historical number in the manuscript: the release uses one fixed tensorization, split, preprocessing, and seed protocol. This prevents the old HF-feature validation reuse issue and makes future reruns comparable.

## Evaluate the released checkpoints

```bash
python evaluate.py --checkpoint checkpoints/vgg16_cifar10_dense_best.pt
python evaluate.py --checkpoint checkpoints/vgg16_cifar10_fcttd_r400_seed43.pt --fcttd
```

The first run downloads CIFAR-10 into `./data`. Use `--device cpu` when CUDA is unavailable.

## Compress a new dense checkpoint

```bash
python decompose_fc1.py \
  --dense-checkpoint checkpoints/vgg16_cifar10_dense_best.pt \
  --output checkpoints/vgg16_cifar10_fcttd_new.pt \
  --in-shape 16 32 49 --out-shape 16 16 16 \
  --tucker-ranks 8 8 8 8 8 8 --cp-rank 400
python evaluate.py --checkpoint checkpoints/vgg16_cifar10_fcttd_new.pt --fcttd
```

`decompose_fc1.py` performs HOSVD/Tucker initialization followed by CP-ALS on the Tucker core. Increase `--tucker-iters` or `--cp-iters` for a more accurate but slower decomposition.

## Fine-tuning and Table VIII

The full training script supports dense and compressed FC1 runs. A dry run prints all 15 commands (five ranks × three seeds):

```bash
python run_table_viii.py
python run_table_viii.py --execute --data-root ./data --output-root ./runs/table_viii
```

The exact settings are in `configs/table_viii.json`: `R_k in {32,64,90,160,400}`, `R_t=(8,8,8,8,8,8)`, output tensorization `(16,16,16)`, input tensorization `(16,32,49)`, and seeds `{42,43,44}`. FC1 weight-only parameter counts are 2,696 / 4,232 / 5,480 / 8,840 / 20,360, giving 38,115.89× / 24,281.77× / 18,751.91× / 11,624.49× / 5,047.17× compression relative to 102,760,448 dense weights.

## Verification

```bash
python -m unittest discover -s tests -v
```

The test checks the published parameter counts and verifies that FCTTD forward propagation does not materialize a dense weight matrix.

## Citation and license

Please cite the FCTTD paper and retain the checkpoint metadata when reporting a rerun. The source is released under the MIT License.
