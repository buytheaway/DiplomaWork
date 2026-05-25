# External Torch Candidate Comparison

External Torch candidates were evaluated as neutral technical candidates for the custom pipeline. Runtime backend, desktop, search/enroll logic, and FAISS indexes were not replaced by this report.

| Model | Source | Architecture | TTA | EER | Accuracy | TAR@FAR=0.01 | TAR@FAR=0.001 | Threshold | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| Old custom | local project checkpoint | custom Torch IR-50 | none | 0.166949 | 0.834350 | 0.496449 | - | - | Existing custom reference |
| Best distill custom | local distillation run | custom Torch IR-50 | none | 0.151333 | 0.851000 | 0.492667 | - | - | Best local distillation result before external candidate search |
| Reference baseline | pretrained ONNX/InsightFace | external baseline | none | 0.027852 | 0.984556 | 0.971141 | - | - | Comparison baseline, not proposed custom model |
| camenduru_show_ms1mv3_r50 | camenduru/show `ms1mv3_arcface_r50_fp16.pth` | insightface_iresnet50 | none | 0.150000 | 0.855000 | 0.486667 | - | - | 600-pair smoke only |
| camenduru_show_ms1mv3_r100 | camenduru/show `ms1mv3_arcface_r100_fp16.pth` | insightface_iresnet100 | none | 0.118000 | 0.887667 | 0.672667 | 0.510667 | 0.325131 | Full LFW |
| camenduru_show_ms1mv3_r100 | camenduru/show `ms1mv3_arcface_r100_fp16.pth` | insightface_iresnet100 | hflip | 0.106667 | 0.897667 | 0.717667 | 0.571333 | 0.355348 | Full LFW |
| ygtxr1997_glint360k_r100 | ygtxr1997/CelebBasis `glint360k_cosface_r100_fp16_0.1/backbone.pth` | insightface_iresnet100 | none | 0.063000 | 0.940333 | 0.854333 | 0.670667 | 0.262402 | Best external candidate without TTA |
| ygtxr1997_glint360k_r100 | ygtxr1997/CelebBasis `glint360k_cosface_r100_fp16_0.1/backbone.pth` | insightface_iresnet100 | hflip | 0.057000 | 0.947000 | 0.867000 | 0.740333 | 0.271411 | Best external candidate with hflip TTA |

## Interpretation

The best external Torch candidate is `ygtxr1997_glint360k_r100`. It improves strongly over the old custom checkpoint and the best local distillation checkpoint, but it does not reach 95% LFW accuracy in this evaluation. It does meet the technical target by EER (`0.057 <= 0.07` with hflip) and TAR@FAR=0.01 (`0.867 >= 0.80` with hflip).

The reference baseline remains better on all listed metrics. Runtime replacement is therefore a technical option, not an automatic final claim. If this candidate is approved for runtime use, all custom embeddings and custom FAISS indexes must be recomputed because the embedding space changes.
