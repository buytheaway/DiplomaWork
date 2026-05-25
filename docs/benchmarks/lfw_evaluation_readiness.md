# LFW Evaluation Status

This file used to track readiness for real LFW threshold calibration. The
evaluation is now complete for the final defense package.

## Inputs Used

- LFW root: `handoff_lfw_eval/lfw`
- Pairs file: `handoff_lfw_eval/lfw/pairs.txt`
- Custom checkpoint artifact: `handoff_lfw_eval/artifacts/best_lfw.pth`
- Pretrained baseline models:
  - `models/det_10g.onnx`
  - `models/w600k_r50.onnx`

The raw LFW images and model checkpoints remain local artifacts and must not be
committed to git.

## Evaluator

The final evaluator is:

```powershell
python scripts\evaluate_lfw_verification.py `
  --lfw-root handoff_lfw_eval\lfw `
  --pairs-file handoff_lfw_eval\lfw\pairs.txt `
  --pipeline custom `
  --output-dir reports\biometric_eval\lfw_custom
```

The same script was run with `--pipeline pretrained` for the
ONNX/InsightFace-style baseline.

## Results Document

The tracked results summary is:

`docs/benchmarks/lfw_biometric_verification_results.md`

That document reports EER, FAR/FRR at selected thresholds, TAR@FAR, positive and
negative pair counts, skipped-pair counts, and score statistics.

## Important Boundary

LFW results are biometric verification metrics. They are not the same as the
synthetic FAISS retrieval benchmark. LFW must remain evaluation-only in the
final diploma narrative; the final real-face fine-tuning dataset is CelebA
identity folders, with train and validation splits under `datasets/celeba_faces`.
