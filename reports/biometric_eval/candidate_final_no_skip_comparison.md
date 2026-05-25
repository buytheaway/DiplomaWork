# Candidate Final No-Skip Comparison

Checkpoint: `custom_torch_candidate_bundle/model.pth`

Architecture: `insightface_iresnet100`

Backend, desktop, runtime checkpoint, search/enroll flow, and FAISS indexes were not changed.

| Evaluation mode | Valid pairs | Skipped | Fallback pairs | EER | Accuracy | TAR@FAR=0.01 | TAR@FAR=0.001 | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| center_crop no-skip | 6000 | 0 | 0 | 0.057000 | 0.947000 | 0.867000 | 0.740333 | 0.271411 |
| runtime-style valid-only | 5851 | 149 | - | 0.009602 | 0.993505 | 0.990460 | 0.986371 | 0.205047 |
| runtime_fallback_center_crop no-TTA | 6000 | 0 | 96 | 0.022667 | 0.985833 | 0.976000 | 0.971333 | 0.198553 |
| runtime_fallback_center_crop hflip | 6000 | 0 | 149 | 0.015000 | 0.990500 | 0.984667 | 0.981000 | 0.205047 |
| old custom | 5904 | 96 | - | 0.166949 | 0.834350 | 0.496449 | 0.213730 | 0.322468 |
| best distill custom | - | - | - | 0.151333 | 0.851000 | 0.492667 | - | - |
| reference baseline | 5957 | 43 | - | 0.027852 | 0.984556 | 0.971141 | 0.969128 | 0.252612 |

## Final Candidate Result

The final no-skip candidate result is `runtime_fallback_center_crop + RGB + [-1,1] + hflip`:

- valid pairs: `6000/6000`
- skipped pairs: `0`
- fallback pairs: `149`
- runtime-success image evaluations: `23789`
- fallback center-crop image evaluations: `211`
- final no-face pairs: `0`
- final extractor-error pairs: `0`
- EER: `0.015000`
- best accuracy: `0.990500`
- TAR@FAR=0.01: `0.984667`
- TAR@FAR=0.001: `0.981000`
- best threshold: `0.205047`

This reaches the requested no-skip 6000/6000 target and exceeds 95% LFW accuracy. It remains an isolated evaluation result; runtime replacement still requires explicit approval, architecture integration, custom embedding recomputation, and custom FAISS rebuild.
