# Final Tables For Diploma

Generated: 2026-05-24

## A. LFW Biometric Metrics

| Metric | Final custom value |
|---|---:|
| Protocol | LFW labeled pairs |
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

## B. Model Comparison

| Pipeline | Valid pairs | EER | EER threshold | Best accuracy | Best threshold | FAR | FRR | TAR@FAR=0.01 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Old custom `torch_ir50` | 5904 | 0.166949 | n/a | 0.834350 | n/a | n/a | n/a | 0.496449 |
| Best distillation custom | 6000 | 0.151333 | n/a | 0.851000 | n/a | n/a | n/a | 0.492667 |
| Final custom `torch_insightface_iresnet100` | 6000 | 0.015000 | 0.154398 | 0.990500 | 0.205047 | 0.001667 | 0.017333 | 0.984667 |
| Pretrained ONNX/InsightFace baseline | 5957 | 0.027852 | n/a | 0.984556 | n/a | n/a | n/a | 0.971141 |

## C. Real-Image Embedding Benchmark

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

## D. Experiment Separation

| Experiment | Data type | Measures | Does not measure |
|---|---|---|---|
| LFW biometric verification | Real labeled face pairs | FAR, FRR, EER, TAR@FAR, threshold behavior, verification accuracy | FAISS scalability or database UI performance |
| Real-image embedding retrieval benchmark | Embeddings extracted from real face image files | Embedding extraction success, FAISS index build/search latency on real embeddings | FAR/FRR/EER or 1M-scale claims |
| Synthetic 1M/2M FAISS benchmark | Synthetic L2-normalized 512D vectors | Retrieval scalability, latency, index size, build time, `top_k_overlap@K` | Biometric recognition accuracy or identity hit rate |
| Desktop/live identification smoke | Runtime camera/upload workflow | End-to-end identification behavior under demo conditions | LFW FAR/FRR/EER or formal biometric verification accuracy |

## E. Runtime Latency

| Scenario | Result | Total ms | Detect+embed ms | FAISS ms | DB ms | Notes |
|---|---|---:|---:|---:|---:|---|
| Same-photo custom search | match | 65.78 | 63.85 | 0.26 | 1.63 | Final custom model, candidate index |
| Different-photo custom search | unknown | 61.60 | 59.86 | 0.35 | 1.36 | Correct unknown result; manual safe fallback |
| Live-style custom search | match | 41.75 | 40.15 | 0.19 | 1.37 | `live_fast`, no fallback |

## E2. Runtime Identification Smoke

| Folder | Pipeline | Items | Correct | False matches | No matches | Accuracy | Mean latency ms | p95 latency ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `data/new_custom_enroll` cleaned | custom | 81 | 81 | 0 | 0 | 1.000000 | 242.586 | 359.206 |

This is an end-to-end backend search smoke test on the re-enrollment folder.
It is not a formal held-out desktop accuracy result.

## F. Synthetic 1M/2M Retrieval Scalability

| Dataset size | Method | p50 ms | p95 ms | p99 ms | Build s | Index MB | top_k_overlap@1 | top_k_overlap@5 | top_k_overlap@10 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000,000 | HNSW M=32 efSearch=128 | 1.738500 | 2.304975 | 2.523790 | 982.189322 | 2212.647921 | 0.860000 | 0.244000 | 0.168000 |
| 1,000,000 | IVF-PQ nlist=4096 m=32 nprobe=32 | 0.347850 | 0.392000 | 0.466494 | 38.337446 | 46.678394 | 1.000000 | 0.208000 | 0.108000 |
| 2,000,000 | IVF-PQ nlist=4096 m=32 nprobe=32 | 1.096600 | 1.142425 | 1.159439 | 54.552124 | 84.825367 | 1.000000 | 0.208000 | 0.104000 |

## G. Real 1M Readiness

| Item | Value |
|---|---:|
| Local real images available for extraction | ~206,885 |
| Local synthetic DigiFace sample images | 360 |
| Real 1M image benchmark completed | No |
| Real 2M image benchmark completed | No |
| Required next dataset for truthful 1M/2M real-image embeddings | VGGFace2 or equivalent |

The LFW `0.990500` value is pair verification accuracy, not desktop/live
identification accuracy. Desktop accuracy must be measured with a separate
held-out runtime protocol if it is required as a formal number.
