# LFW Biometric Verification Metrics

This report uses the LFW pairs verification protocol.
A pair is predicted as the same identity when `score >= threshold`.

| Pipeline | Valid pairs | Positive pairs | Negative pairs | EER | EER threshold | Best accuracy | Best threshold | FAR | FRR | TAR@FAR=0.01 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pretrained ONNX/InsightFace baseline | 5957 | 2980 | 2977 | 0.027852 | 0.115405 | 0.984556 | 0.252612 | 0.000000 | 0.030872 | 0.971141 |

## Additional Statistics

- Mean positive score: `0.654416`
- Mean negative score: `0.004299`
- Skipped pairs: `43`

## Notes

- FAR = false accepts / total negative pairs.
- FRR = false rejects / total positive pairs.
- EER is estimated at the threshold where FAR and FRR are closest.
- These are real biometric verification metrics for the evaluated pairs.
