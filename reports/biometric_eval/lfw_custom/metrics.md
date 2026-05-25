# LFW Biometric Verification Metrics

This report uses the LFW pairs verification protocol.
A pair is predicted as the same identity when `score >= threshold`.

| Pipeline | Valid pairs | Positive pairs | Negative pairs | EER | EER threshold | Best accuracy | Best threshold | FAR | FRR | TAR@FAR=0.01 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Proposed custom Torch IR-50 pipeline | 5904 | 2957 | 2947 | 0.166949 | 0.307675 | 0.834350 | 0.322468 | 0.142518 | 0.188705 | 0.496449 |

## Additional Statistics

- Mean positive score: `0.47401`
- Mean negative score: `0.182734`
- Skipped pairs: `96`

## Notes

- FAR = false accepts / total negative pairs.
- FRR = false rejects / total positive pairs.
- EER is estimated at the threshold where FAR and FRR are closest.
- These are real biometric verification metrics for the evaluated pairs.
