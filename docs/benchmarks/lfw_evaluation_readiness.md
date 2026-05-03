# LFW Evaluation Readiness Checklist

This checklist records the current readiness state for running real LFW
threshold calibration. No numeric biometric results are reported here because
the evaluation was not run.

## Current File Check

- `data/lfw_aligned`: missing.
- `data/lfw/pairs.txt`: missing.
- `training/outputs/checkpoint_epoch_004.pth`: present locally.

## Evaluation Path

The LFW evaluation script uses the custom PyTorch training path:

- model builder: `training.models.ir_resnet.build_model`
- configured architecture: `ir50`
- configured embedding dimension: `512`
- checkpoint input: `--weights`
- pairwise similarity: dot product of L2-normalized embeddings

This is not the backend pretrained extractor path and does not use FAISS search.

## Stable Extractor Evaluation Path

PR 3.3 adds a standalone stable extractor evaluator:

`scripts/evaluate_verification_pairs.py`

This script can evaluate LFW-style pairs through the backend embedding extractor
interface without changing the backend API, database, runtime registry, or FAISS
search. Use `--backend onnx` for the current stable ONNX extractor path, or
`--backend insightface` for the InsightFace extractor path.

The current ONNX model files are present locally:

- `models/det_10g.onnx`
- `models/w600k_r50.onnx`

The LFW image directory and pairs file are still missing, so no stable extractor
FAR/FRR/EER values have been generated.

## Required Inputs

- Aligned LFW images in the directory expected by `--lfw-dir`.
- LFW `pairs.txt` in the path expected by `--pairs`.
- A compatible PyTorch checkpoint for the configured model architecture.
- A selected device, for example `--device cpu` for reproducible local CPU runs.

## Command To Run Later

Custom PyTorch branch:

```powershell
python training\eval_lfw.py `
  --weights training\outputs\checkpoint_epoch_004.pth `
  --lfw-dir data\lfw_aligned `
  --pairs data\lfw\pairs.txt `
  --device cpu `
  --output training\outputs\lfw_results.json
```

Stable ONNX extractor path:

```powershell
python scripts\evaluate_verification_pairs.py `
  --backend onnx `
  --images-dir data\lfw_aligned `
  --pairs data\lfw\pairs.txt `
  --output training\outputs\stable_onnx_lfw_results.json `
  --target-far 0.001,0.01,0.1 `
  --fail-on-missing
```

After the command completes successfully, numerical FAR, FRR, EER,
EER-threshold, TAR@FAR, and selected-threshold values may be copied into a
separate benchmark results document. Those values should not be written into
diploma documentation before a real evaluation run completes.

## Not Run In PR 3.1

The real LFW evaluation was not run in PR 3.1 because the required local LFW
data paths were missing.
