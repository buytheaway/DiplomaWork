# LFW Biometric Verification Results

This document closes the real-biometric-metrics gap for the final diploma
version. It reports labeled-pair verification metrics.

## Protocol

- Dataset: LFW pairs protocol.
- Local evaluation root: `handoff_lfw_eval/lfw`.
- Pairs file: `handoff_lfw_eval/lfw/pairs.txt`.
- Score convention: embeddings are L2-normalized and compared by dot product;
  `score >= threshold` means predicted same identity.
- Positive pair: same person.
- Negative pair: different persons.
- LFW is evaluation-only for the final report. It must not be used as the final
  training dataset.

## Pipelines

| Pipeline | Role |
|---|---|
| Final custom `torch_insightface_iresnet100` pipeline | Final custom runtime model for the project. |
| Old custom `torch_ir50` pipeline | Historical custom runtime result kept for comparison. |
| Best distillation custom | Experimental training branch kept for comparison. |
| Pretrained ONNX/InsightFace baseline | External pretrained reference and comparison baseline. |

## Final Results

| Pipeline | Valid pairs | Positive pairs | Negative pairs | Skipped pairs | EER | EER threshold | Best accuracy | Best threshold | FAR at selected/best threshold | FRR at selected/best threshold | TAR@FAR=0.1 | TAR@FAR=0.01 | TAR@FAR=0.001 | Mean positive score | Mean negative score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Final custom `torch_insightface_iresnet100` | 6000 | 3000 | 3000 | 0 | 0.015000 | 0.154398 | 0.990500 | 0.205047 | 0.001667 | 0.017333 | 0.988333 | 0.984667 | 0.981000 | 0.653118 | 0.004843 |
| Old custom `torch_ir50` | 5904 | 2957 | 2947 | 96 | 0.166949 | 0.307675 | 0.834350 | 0.322468 | 0.142518 | 0.188705 | 0.764288 | 0.496449 | 0.213730 | 0.474010 | 0.182734 |
| Best distillation custom | 6000 | n/a | n/a | n/a | 0.151333 | n/a | 0.851000 | n/a | n/a | n/a | n/a | 0.492667 | n/a | n/a | n/a |
| Pretrained ONNX/InsightFace baseline | 5957 | 2980 | 2977 | 43 | 0.027852 | 0.115405 | 0.984556 | 0.252612 | 0.000000 | 0.030872 | 0.975503 | 0.971141 | 0.969128 | 0.654416 | 0.004299 |

## Definitions

- FAR = false accepts / total negative pairs.
- FRR = false rejects / total positive pairs.
- EER = the operating point where FAR and FRR are closest.
- TAR@FAR = true accept rate at a selected target false accept rate.
- Best accuracy threshold = threshold that maximizes pair-level verification
  accuracy on the evaluated protocol.

## Interpretation

The project now has real biometric verification metrics on labeled face pairs.
The final custom `torch_insightface_iresnet100` runtime reached 6000/6000 valid
LFW pairs, accuracy `0.990500`, EER `0.015000`, and TAR@FAR=0.01 `0.984667`.
The selected runtime threshold is `0.205047`, with FAR `0.001667` and FRR
`0.017333` at that threshold.

The historical `torch_ir50` and distillation checkpoints remain useful as
development comparisons, but they should not be presented as the final runtime
quality. Old `torch_ir50` embeddings are not compatible with the final
`torch_insightface_iresnet100` embedding space and cannot be converted without
the original source images.

If a checkpoint was selected using LFW feedback, the corresponding LFW numbers
must be treated as validation-stage evidence rather than a fully independent
generalization test.

## Separation From Retrieval Measurements

These LFW results evaluate biometric verification quality. They are separate
from FAISS retrieval and indexing measurements. Retrieval latency or index size
must not be described as biometric FAR, FRR, EER, or identification accuracy.
