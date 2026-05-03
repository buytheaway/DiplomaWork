# Stable Extractor Verification Methodology

This document describes how to evaluate biometric verification thresholds for
the stable MVP extractor path without changing the backend API, database,
runtime registry, or FAISS search.

## Scope

The standalone evaluator is `scripts/evaluate_verification_pairs.py`. It reads
LFW-style verification pairs, extracts one embedding per image with a selected
backend extractor, computes dot-product similarity between L2-normalized
embeddings, and reports threshold metrics through `training.verification_metrics`.

When run with `--backend onnx` or `--backend insightface`, this evaluator
reflects the stable MVP extractor path more directly than `training/eval_lfw.py`.
The ONNX path uses the backend ONNX extractor configuration. The InsightFace
path uses the backend InsightFace extractor configuration.

## Separation From Other Benchmarks

This evaluator is separate from the synthetic FAISS retrieval benchmark. The
synthetic benchmark measures vector index behavior: latency, build time, memory
estimate, and top-k overlap against exact Flat search. It has no identity
labels and cannot produce biometric FAR, FRR, EER, or biometric hit@K.

This evaluator is also separate from `training/eval_lfw.py`. The training LFW
script evaluates the custom PyTorch checkpoint branch. The stable extractor
evaluator uses the backend embedding extractor interface and can evaluate ONNX,
InsightFace, Torch, or Dummy backends without using the backend API.

## Metrics

The evaluator can report:

- FAR: false accepts divided by all negative pairs.
- FRR: false rejects divided by all positive pairs.
- EER: point where FAR and FRR are closest or safely interpolated.
- EER threshold.
- TAR@FAR for selected target FAR values.
- Best accuracy threshold.
- FAR/FRR curve points.

Higher score means more similar, and `score >= threshold` means match.

## Backend Notes

- `onnx`: appropriate for stable MVP extractor claims if the configured ONNX
  detector and embedder files are present and the evaluation is actually run.
- `insightface`: appropriate for stable extractor comparison if the local
  InsightFace model dependency and model files are available.
- `torch`: evaluates the backend Torch extractor path, which is custom-model
  oriented and should not be described as the pretrained stable path.
- `dummy`: useful only for plumbing tests. It has no biometric meaning and must
  not be used for biometric accuracy claims.

## Claims Policy

FAR, FRR, EER, TAR@FAR, and threshold values must not be claimed in diploma
documentation unless `scripts/evaluate_verification_pairs.py` or another real
labeled verification evaluation was actually run on the target dataset.

Do not claim that a model is accurate, production-ready, optimal, or low-error
unless those claims are supported by recorded evaluation output.

## Example ONNX Command

```powershell
python scripts\evaluate_verification_pairs.py `
  --backend onnx `
  --images-dir data\lfw_aligned `
  --pairs data\lfw\pairs.txt `
  --output training\outputs\stable_onnx_lfw_results.json `
  --target-far 0.001,0.01,0.1 `
  --fail-on-missing
```

The output JSON may be copied into `docs/benchmarks` only after the command
successfully completes on the intended local dataset.
