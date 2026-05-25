# Final Evaluation Summary

Generated: 2026-05-24

## Final Custom Model Configuration

| Field | Value |
|---|---|
| Runtime model name | `torch_insightface_iresnet100` |
| Checkpoint | `custom_torch_candidate_bundle/model.pth` |
| Architecture | `insightface_iresnet100` |
| Preprocessing | `runtime_fallback_center_crop` |
| Color order | `RGB` |
| Normalization | `[-1, 1]` |
| Test-time augmentation | `hflip` |
| Selected runtime threshold | `0.205047` |

This model is a separate embedding space from the historical `torch_ir50`
custom model. Old `torch_ir50` vectors are not compatible with this model and
must not be mixed into the same FAISS index.

## LFW Real Biometric Verification Metrics

LFW is used as the labeled-pair biometric verification protocol. It is not used
for training in the final evaluation.

| Metric | Value |
|---|---:|
| Valid pairs | 6000 / 6000 |
| Skipped pairs | 0 |
| Accuracy | 0.990500 |
| EER | 0.015000 |
| EER threshold | 0.154398 |
| Selected threshold | 0.205047 |
| FAR at selected threshold | 0.001667 |
| FRR at selected threshold | 0.017333 |
| TAR@FAR=0.01 | 0.984667 |
| TAR@FAR=0.001 | 0.981000 |

## Model Comparison

| Model | Role | Valid pairs | Accuracy | EER | TAR@FAR=0.01 | Notes |
|---|---|---:|---:|---:|---:|---|
| Old custom `torch_ir50` | historical custom runtime | 5904 | 0.834350 | 0.166949 | 0.496449 | Kept only for historical compatibility. |
| Best distillation custom | experimental training branch | 6000 | 0.851000 | 0.151333 | 0.492667 | Did not reach final quality target. |
| Final custom `torch_insightface_iresnet100` | final custom runtime candidate | 6000 | 0.990500 | 0.015000 | 0.984667 | Selected runtime model. |
| Pretrained ONNX/InsightFace baseline | external comparison baseline | 5957 | 0.984556 | 0.027852 | 0.971141 | Reference baseline, not the proposed custom runtime. |

## Real-Image Embedding Retrieval Benchmark

This benchmark extracts embeddings from real image files using the final custom
runtime model and builds an isolated FAISS index. It does not touch the
production database/index.

| Metric | Value |
|---|---:|
| Real images scanned | 206802 |
| Images attempted | 10000 |
| Embeddings created | 9802 |
| Skipped no-face | 0 |
| Skipped invalid | 0 |
| Skipped multiple-faces | 198 |
| Extractor errors | 0 |
| Embedding dimension | 512 |
| Index type | HNSW |
| Index size | 21.76 MB |
| Extraction time | 364.928 s |
| Build time | 0.523 s |
| Search p50 | 0.141150 ms |
| Search p95 | 0.194645 ms |
| Search p99 | 0.338031 ms |

The source folders were `datasets/celeba_faces`, `handoff_lfw_eval/lfw`, and
`data/new_custom_enroll`. The output summaries are under
`reports/real_image_embedding_benchmark/`. Raw embeddings, FAISS index files,
mapping CSV files, and image datasets are generated artifacts and should not be
committed.

## Runtime Latency Summary

The latest smoke check used the final custom runtime model and the candidate
custom index.

| Scenario | Result | Total ms | Detect+embed ms | FAISS ms | DB ms | Search mode |
|---|---|---:|---:|---:|---:|---|
| Same-photo custom search | match | 65.78 | 63.85 | 0.26 | 1.63 | manual fast |
| Different-photo custom search | unknown | 61.60 | 59.86 | 0.35 | 1.36 | manual safe fallback |
| Live-style custom search | match | 41.75 | 40.15 | 0.19 | 1.37 | `live_fast` |

The live-style request uses fast webcam search settings and no safe fallback.
The measured bottleneck is detection plus embedding extraction, not FAISS.

Current smoke database/index state:

| Field | Value |
|---|---:|
| Active persons | 491 |
| Active embeddings total | 18670 |
| Active `custom/torch_insightface_iresnet100` embeddings | 104 |
| Active historical `custom/torch_ir50` embeddings | 9244 |
| Active `pretrained/onnx_w600k_r50` embeddings | 9305 |
| Active `pretrained/torch_ir50` embeddings | 17 |
| Loaded custom index type | HNSW |
| Loaded custom index vectors | 104 |
| Loaded custom index memory estimate | 266240 bytes |

Runtime identification must be reported separately from LFW pair verification.
The LFW accuracy `0.990500` is a biometric verification metric on labeled
pairs. It is not the same as desktop/live webcam top-1 identification accuracy.
If the desktop demo is observed around `79-80%`, that number reflects the
end-to-end runtime workflow and must not be replaced by the LFW metric.

An end-to-end runtime evaluator was added for this purpose:
`scripts/evaluate_runtime_identification.py`. It calls the real backend
`/v1/search` endpoint over labeled folders and reports correct labels,
false matches, no-matches, and latency.

Latest cleaned re-enroll-folder smoke result:

| Runtime identification smoke | Value |
|---|---:|
| Folder | `data/new_custom_enroll` |
| Pipeline | `custom` |
| Items | 81 |
| Correct | 81 |
| False matches | 0 |
| No matches | 0 |
| Accuracy | 1.000000 |
| Mean latency | 242.586 ms |
| p95 latency | 359.206 ms |

This is a smoke test because the same folder was used for re-enrollment. A
formal desktop/runtime identification accuracy claim still requires a separate
held-out test folder that was not used for enrollment.

## Migration and Integration Note

Historical `torch_ir50` embeddings cannot be converted into
`torch_insightface_iresnet100` embeddings. A face embedding is the numeric
output of a specific model and preprocessing pipeline. Changing the model
changes the embedding space.

The database does not preserve source image paths or raw image bytes for the old
records, so a lossless migration is impossible. The correct path is:

1. keep old `torch_ir50` embeddings as historical records;
2. enroll or import original face images again through the new model;
3. write new rows with `pipeline=custom` and
   `model_name=torch_insightface_iresnet100`;
4. build a separate custom FAISS index for the new model only.

## Safe Diploma Claims

Safe:

- The final custom Torch runtime was evaluated on the real LFW labeled-pair
  protocol with 6000/6000 valid pairs.
- The final custom runtime achieved accuracy 0.990500 and EER 0.015000 on that
  LFW evaluation.
- FAR, FRR, EER, TAR@FAR, and threshold behavior are reported separately from
  retrieval/index latency measurements.
- A real-image embedding retrieval benchmark was run on embeddings extracted
  from real face image files, with 9802 embeddings created from 10000 attempted
  images.
- The current local machine contains about 206k real image files available for
  real-image embedding extraction; a truthful 1M/2M real-image benchmark
  requires an additional real dataset such as VGGFace2.

Not safe:

- Do not claim 1M real face images were evaluated unless a real-image benchmark
  is actually run at that scale.
- Do not claim 2M real face images were evaluated unless a real-image benchmark
  is actually run at that scale.
- Do not present LFW `0.990500` accuracy as desktop/live webcam identification
  accuracy.
- Do not mix old `torch_ir50` and new `torch_insightface_iresnet100` embeddings
  in one index.
- Do not claim IVF-PQ is quality-equivalent to exact search in the current
  configuration.
- Do not claim liveness or spoofing resistance.
