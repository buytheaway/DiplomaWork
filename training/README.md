# Training Pipeline — Quick Start Guide

This directory contains everything needed to train a custom face embedding model.

## Prerequisites

```bash
pip install torch torchvision tqdm pyyaml scikit-learn
```

## Directory Structure

```
training/
├── config.yaml              # Main training config (hyperparameters)
├── train.py                 # Training script (ArcFace / AdaFace)
├── eval.py                  # Simple centroid-based evaluation
├── eval_lfw.py              # LFW pairwise verification benchmark
├── losses/
│   ├── arcface.py           # ArcFace loss (m=0.5, s=64)
│   └── adaface.py           # AdaFace loss (quality-adaptive margin)
├── models/
│   └── ir_resnet.py         # IR-18 / IR-50 / IR-100 backbones
├── datasets/
│   └── folder_dataset.py    # Folder-based dataset loader
└── outputs/                 # Checkpoints and metrics saved here
```

## Step 1: Prepare Data

Download **CASIA-WebFace** (pre-aligned 112×112) from InsightFace:
- Place training images in: `data/casia_webface/train/`
- Place validation images in: `data/casia_webface/val/`

Each subdirectory = one person identity:
```
data/casia_webface/train/
├── 0000001/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── 0000002/
│   └── ...
```

For LFW evaluation, also download aligned LFW images and the `pairs.txt`:
- `data/lfw_aligned/` — aligned 112×112 LFW images
- `data/lfw/pairs.txt` — standard LFW pairs file

## Step 2: Train with ArcFace (Primary)

```bash
# From the project root:
python training/train.py --config training/config.yaml

# Override params:
python training/train.py --epochs 28 --batch-size 128 --device cuda
```

Config defaults (in `config.yaml`):
- **Backbone**: IR-50
- **Loss**: ArcFace (m=0.5, s=64)
- **Optimizer**: SGD (lr=0.1, momentum=0.9, wd=5e-4)
- **Scheduler**: MultiStepLR (milestones=[8, 14, 20, 25], γ=0.1)
- **Epochs**: 28
- **Augmentations**: HFlip, ColorJitter, RandomErasing
- **Gradient clipping**: max_norm=5.0

## Step 3: Train with AdaFace (Optional — P2)

```bash
python training/train.py --config training/config.yaml --loss adaface
```

Or set `loss.type: "adaface"` in `config.yaml`.

## Step 4: Evaluate on LFW

```bash
python training/eval_lfw.py \
    --weights training/outputs/checkpoint_epoch_028.pth \
    --lfw-dir data/lfw_aligned \
    --pairs data/lfw/pairs.txt \
    --device cuda
```

Outputs:
- **Accuracy** (best threshold)
- **AUC**
- **TAR@FAR=1e-1, 1e-2, 1e-3, 1e-4**
- Results saved to `training/outputs/lfw_results.json`

## Step 5: Export to ONNX

```bash
python scripts/export_onnx.py \
    --weights training/outputs/checkpoint_epoch_028.pth \
    --output models/custom_ir50.onnx \
    --validate
```

## Step 6: Run Retrieval Benchmark

```bash
python scripts/benchmark_retrieval.py --source-index backend/data/index/test3.faiss
```

Compares Flat vs HNSW vs IVF-PQ with recall@k, latency, and memory metrics.

## Step 7: Update `.env` to Use Custom Model

After ONNX export, update `.env`:
```env
CUSTOM_BACKEND=onnx
# Point to your new ONNX model:
# (you'll need to add a config key for custom embedder path)
```

## Resume Training

```bash
python training/train.py --resume training/outputs/checkpoint_epoch_020.pth --epochs 28
```

## Expected Timeline

| Phase | Action | Time (approx) |
|-------|--------|---------------|
| Data prep | Download CASIA-WebFace + LFW | ~1 hour |
| Training (ArcFace) | 28 epochs on CASIA | ~2–6 hours (GPU) |
| Eval LFW | Run eval_lfw.py | ~5 min |
| ONNX export | Run export_onnx.py | ~1 min |
| Training (AdaFace) | 28 epochs, swap loss | ~2–6 hours (GPU) |
| Retrieval benchmark | Run benchmark_retrieval.py | ~2 min |
