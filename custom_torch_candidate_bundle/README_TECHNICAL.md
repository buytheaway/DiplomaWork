# Custom Torch Candidate Bundle

This is the neutral technical bundle for the final custom Torch embedding pipeline.
It is configured as the approved runtime replacement in the local final-defense environment, but the bundle still does not imply any public claim beyond the measured evaluation results below.

## Model

- Pipeline name: `custom_torch_candidate`
- Architecture: `insightface_iresnet100`
- Embedding dimension: `512`
- Input: `112x112`
- Color order: `RGB`
- Normalization: `[-1, 1]`
- Checkpoint file: `model.pth`
- Source: ygtxr1997/CelebBasis `glint360k_cosface_r100_fp16_0.1/backbone.pth`

## LFW Evaluation

Command:

```powershell
python scripts\evaluate_torch_candidate_lfw.py `
  --checkpoint external_models\torch_candidates\ygtxr1997_glint360k_r100\glint360k_cosface_r100_fp16_0.1\backbone.pth `
  --architecture iresnet100 `
  --lfw-root handoff_lfw_eval\lfw `
  --pairs-file handoff_lfw_eval\lfw\pairs.txt `
  --output-dir reports\biometric_eval\external_glint_r100_full `
  --tta none `
  --normalization minus1_1 `
  --color-order rgb `
  --preprocess center_crop
```

No-TTA metrics:

- EER: `0.063`
- Best accuracy: `0.940333`
- TAR@FAR=0.01: `0.854333`
- TAR@FAR=0.001: `0.670667`
- Best threshold: `0.262402`

Hflip TTA metrics:

- EER: `0.057`
- Best accuracy: `0.947`
- TAR@FAR=0.01: `0.867`

## Best Preprocessing Sweep Result

The strongest sweep result used runtime-style preprocessing:

```powershell
python scripts\evaluate_torch_candidate_lfw.py `
  --checkpoint custom_torch_candidate_bundle\model.pth `
  --architecture iresnet100 `
  --lfw-root handoff_lfw_eval\lfw `
  --pairs-file handoff_lfw_eval\lfw\pairs.txt `
  --output-dir reports\biometric_eval\best_candidate_preprocessing_sweep\g_runtime_rgb_minus_hflip `
  --tta hflip `
  --normalization minus1_1 `
  --color-order rgb `
  --preprocess runtime
```

Runtime-style preprocessing metrics:

- Valid pairs: `5851`
- Skipped pairs: `149`
- Skipped reason: `no_face`
- EER: `0.009602`
- Best accuracy: `0.993505`
- TAR@FAR=0.01: `0.990460`
- TAR@FAR=0.001: `0.986371`
- Best threshold: `0.205047`

These metrics reach the target, but they are valid-pair metrics with skipped LFW pairs. This is a standalone evaluator preprocessing result, not a runtime replacement. For a clean no-skip 6000/6000 comparison, use the center-crop hflip metrics above.

## Final No-Skip Hybrid Evaluation

The final no-skip evaluation uses runtime-style preprocessing first and falls back to center crop only when runtime-style preprocessing fails for an image.

```powershell
python scripts\evaluate_torch_candidate_lfw.py `
  --checkpoint custom_torch_candidate_bundle\model.pth `
  --architecture auto `
  --lfw-root handoff_lfw_eval\lfw `
  --pairs-file handoff_lfw_eval\lfw\pairs.txt `
  --output-dir reports\biometric_eval\candidate_runtime_fallback_center_crop_hflip `
  --preprocess runtime_fallback_center_crop `
  --color-order rgb `
  --normalization minus1_1 `
  --tta hflip
```

Final no-skip metrics:

- Valid pairs: `6000/6000`
- Skipped pairs: `0`
- Fallback pairs: `149`
- Runtime-success image evaluations: `23789`
- Fallback center-crop image evaluations: `211`
- EER: `0.015000`
- Best accuracy: `0.990500`
- TAR@FAR=0.01: `0.984667`
- TAR@FAR=0.001: `0.981000`
- Best threshold: `0.205047`

This reaches the no-skip 6000/6000 target and exceeds 95% LFW accuracy.

## License Note

The Hugging Face file page reports license `cc`. The ArcFace Torch model zoo also states that its models are for non-commercial research use. Treat this bundle as evaluation-only until license suitability is confirmed by the project owner.

## Runtime Replacement

Approved and configured locally on `2026-05-24`.

Runtime settings:

- `TORCH_MODEL_PATH=custom_torch_candidate_bundle/model.pth`
- `TORCH_MODEL_ARCH=insightface_iresnet100`
- `TORCH_PREPROCESS=runtime_fallback_center_crop`
- `TORCH_TTA=hflip`
- `CUSTOM_MATCH_THRESHOLD=0.205047`
- `CUSTOM_INDEX_PATH=backend/data/index/custom_candidate.faiss`

Existing `torch_ir50` embeddings remain in the database as historical records. They are not compatible with this embedding space and cannot be converted without the original source images. New custom embeddings must be enrolled or imported through this runtime checkpoint before rebuilding the candidate custom index.
