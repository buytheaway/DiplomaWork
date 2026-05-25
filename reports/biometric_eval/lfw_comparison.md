# LFW Verification Comparison

| Pipeline | Valid pairs | EER | EER threshold | Best accuracy | Best threshold | TAR@FAR=0.1 | TAR@FAR=0.01 | TAR@FAR=0.001 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Proposed custom Torch IR-50 pipeline | 5904 | 0.166949 | 0.307675 | 0.834350 | 0.322468 | 0.764288 | 0.496449 | 0.213730 |
| Pretrained ONNX/InsightFace baseline | 5957 | 0.027852 | 0.115405 | 0.984556 | 0.252612 | 0.975503 | 0.971141 | 0.969128 |

## Interpretation

- The custom row is the proposed custom Torch IR-50 pipeline.
- The pretrained row is the ONNX/InsightFace baseline/reference.
- Lower EER is better; higher best accuracy and TAR@FAR are better.
- Do not claim the custom pipeline is better unless these metrics prove it.
- In this run, the pretrained baseline has lower EER than custom.
- In this run, the pretrained baseline has higher best-threshold accuracy.
