# External Torch Candidates

External candidates were downloaded only for technical evaluation. Downloaded weights are stored in ignored paths and must not be committed.

| Candidate | Source URL/name | Downloaded path | Size MB | Architecture guess | Embedding dim | Input | Normalization | License note | Notes |
|---|---|---|---:|---|---:|---|---|---|---|
| camenduru_show_ms1mv3_r50 | camenduru/show `models/arcface/ms1mv3_arcface_r50_fp16.pth` | `external_models/torch_candidates/camenduru_show_ms1mv3_r50/models/arcface/ms1mv3_arcface_r50_fp16.pth` | 165.98 | insightface_iresnet50 | 512 | 112x112 | minus1_1 | model zoo notes non-commercial research use | Smoke accuracy 0.855000 |
| camenduru_show_ms1mv3_r100 | camenduru/show `models/arcface/ms1mv3_arcface_r100_fp16.pth` | `external_models/torch_candidates/camenduru_show_ms1mv3_r100/models/arcface/ms1mv3_arcface_r100_fp16.pth` | 249.12 | insightface_iresnet100 | 512 | 112x112 | minus1_1 | model zoo notes non-commercial research use | Full hflip accuracy 0.897667 |
| ygtxr1997_glint360k_r100 | ygtxr1997/CelebBasis `glint360k_cosface_r100_fp16_0.1/backbone.pth` | `external_models/torch_candidates/ygtxr1997_glint360k_r100/glint360k_cosface_r100_fp16_0.1/backbone.pth` | 249.12 | insightface_iresnet100 | 512 | 112x112 | minus1_1 | HF page reports cc; evaluation-only until license confirmed | Best external Torch candidate; full hflip accuracy 0.947000 |
