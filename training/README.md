# Training Pipeline - Quick Start Guide

This directory contains the research and training utilities for the custom face
embedding branch.

## Prerequisites

Install the training dependencies in a dedicated environment:

```bash
pip install -r training/requirements.txt
```

Training is CUDA-only in the current branch. Evaluation and export scripts can
run on CPU, but full training expects a working NVIDIA GPU.

## Directory Structure

```text
training/
  config.yaml              Main training config
  train.py                 ArcFace / AdaFace training script
  eval.py                  Closed-set centroid evaluation
  eval_lfw.py              LFW pairwise verification benchmark
  losses/                  ArcFace and AdaFace heads
  models/                  IR-18 / IR-50 / IR-100 backbones
  datasets/                Folder dataset loader
  outputs/                 Local checkpoints and metrics
```

## Dataset Layout

`training/config.yaml` is the source of truth for dataset paths:

```yaml
data:
  train_dir: "datasets/celeba_faces/train"
  val_dir: "datasets/celeba_faces/val"
  lfw_dir: "data/lfw_aligned"
  pairs_file: "data/lfw/pairs.txt"
```

For the custom IR-50 branch, use the prepared CelebA identity folders as the
main real-face fine-tuning dataset. LFW is evaluation-only and must not be used
for training or fine-tuning. DigiFace1M may be used as a synthetic warmstart
dataset or for generating million-scale retrieval data, but final custom model
fine-tuning should use the real CelebA identity split.

Each identity must be a directory, and each image under that directory belongs
to that identity:

```text
datasets/celeba_faces/train/
  1/
    000023.jpg
    ...
  10/
    002307.jpg
    ...
```

The real-face pytest file reads `data.train_dir` from `training/config.yaml`, so
change the config instead of hard-coding another dataset path.

## Train

```bash
python training/train.py --config training/config.yaml
```

Useful overrides:

```bash
python training/train.py --config training/config.yaml --epochs 28 --batch-size 128 --device cuda
python training/train.py --config training/config.yaml --loss adaface
python training/train.py --config training/config.yaml --resume training/outputs/checkpoint_epoch_020.pth
```

Default config:

- Backbone: IR-50
- Loss: ArcFace, margin 0.5, scale 64.0
- Optimizer: SGD, lr 0.1, momentum 0.9, weight decay 0.0005
- Scheduler: MultiStepLR, milestones `[8, 14, 20, 25]`, gamma 0.1
- Epochs: 28
- Augmentations: horizontal flip, color jitter, random erasing
- Gradient clipping: max norm 5.0

## Evaluate

Run LFW verification after training. This is biometric quality evaluation
only; it is separate from FAISS retrieval scalability tests:

```bash
python training/eval_lfw.py \
  --weights training/outputs/checkpoint_epoch_028.pth \
  --lfw-dir data/lfw_aligned \
  --pairs data/lfw/pairs.txt \
  --device cuda
```

The script reports accuracy, AUC, threshold, and TAR at selected FAR values.
It also writes `training/outputs/lfw_results.json`.

The reported FAR, FRR, EER, TAR@FAR, and threshold values must come from this
or another labeled verification protocol. They must not be inferred from FAISS
retrieval benchmarks.

## Export

Export the trained backbone to ONNX:

```bash
python scripts/export_onnx.py \
  --weights training/outputs/checkpoint_epoch_028.pth \
  --output models/custom_ir50.onnx \
  --validate
```

Then point the backend custom pipeline at the exported model in `.env`.
The current runtime uses the shared ONNX detector/embedder settings, so keep the
model paths consistent with the backend extractor you select.

## Retrieval Benchmark

```bash
python scripts/benchmark_retrieval.py --source-index backend/data/index/custom.faiss
```

This compares Flat, HNSW, and IVF-PQ with recall, latency, memory, and build
time metrics.

Million-scale retrieval tests are a separate concern from biometric quality:

- biometric quality: LFW-style FAR/FRR/EER on labeled pairs;
- retrieval scalability: FAISS indexes built on 1M/2M embeddings.

If the custom checkpoint, preprocessing, or embedding model changes, all custom
database embeddings and custom FAISS indexes must be rebuilt because the
embedding space is no longer compatible with previous vectors.

## Notes

- Keep datasets, checkpoints, and generated ONNX files out of git.
- Use the backend tests for API/index behavior and these training tests for
  local checkpoint sanity checks.
- Low separation between different identities means the model needs more data,
  more training, or a better checkpoint before it is useful in the app.
