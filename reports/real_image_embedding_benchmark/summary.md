# Real-Image Embedding Retrieval Benchmark

## Methodology

This benchmark extracts embeddings from real face image files with the final custom Torch runtime model and builds a separate FAISS retrieval index. It does not touch the production database/index.

## Runtime

| Field | Value |
|---|---|
| Model | `torch_insightface_iresnet100` |
| Preprocess | `runtime_fallback_center_crop` |
| Color / normalization | `RGB`, `[-1, 1]` |
| TTA | `hflip` |
| Threshold | `0.205047` |

## Real-Image Retrieval Results

| Metric | Value |
|---|---:|
| Total real images scanned | 206802 |
| Images attempted | 10000 |
| Embeddings created | 9802 |
| Skipped no-face | 0 |
| Skipped invalid | 0 |
| Skipped multiple-faces | 198 |
| Extractor errors | 0 |
| Embedding dim | 512 |
| Index type | `hnsw` |
| Index size MB | 21.76 |
| Extraction time s | 364.928 |
| Build time s | 0.523 |
| Search p50 ms | 0.14115 |
| Search p95 ms | 0.194645 |
| Search p99 ms | 0.338031 |

Output paths:

- embeddings: `reports\real_image_embedding_benchmark\real_image_embeddings.float32.memmap`
- mapping: `reports\real_image_embedding_benchmark\real_image_mapping.csv`
- index: `reports\real_image_embedding_benchmark\real_image_hnsw.faiss`

Sample query result:

```json
{
  "query_row_id": 0,
  "query_identity": "10001",
  "query_image_path": "datasets\\celeba_faces\\test\\10001\\182659.jpg",
  "top_ids": [
    0,
    22,
    19,
    12,
    4,
    7,
    1,
    8,
    17,
    18
  ],
  "top_scores": [
    1.0,
    0.6717286109924316,
    0.6535125970840454,
    0.6369744539260864,
    0.6300750374794006,
    0.6100558042526245,
    0.6015669107437134,
    0.6002235412597656,
    0.5924512147903442,
    0.5881415605545044
  ],
  "top_results": [
    {
      "row_id": 0,
      "identity": "10001",
      "image_path": "datasets\\celeba_faces\\test\\10001\\182659.jpg",
      "score": 1.0
    },
    {
      "row_id": 22,
      "identity": "10001",
      "image_path": "datasets\\celeba_faces\\test\\10001\\200004.jpg",
      "score": 0.6717286109924316
    },
    {
      "row_id": 19,
      "identity": "10001",
      "image_path": "datasets\\celeba_faces\\test\\10001\\198498.jpg",
      "score": 0.6535125970840454
    },
    {
      "row_id": 12,
      "identity": "10001",
      "image_path": "datasets\\celeba_faces\\test\\10001\\194420.jpg",
      "score": 0.6369744539260864
    },
    {
      "row_id": 4,
      "identity": "10001",
      "image_path": "datasets\\celeba_faces\\test\\10001\\187108.jpg",
      "score": 0.6300750374794006
    },
    {
      "row_id": 7,
      "identity": "10001",
      "image_path": "datasets\\celeba_faces\\test\\10001\\190979.jpg",
      "score": 0.6100558042526245
    },
    {
      "row_id": 1,
      "identity": "10001",
      "image_path": "datasets\\celeba_faces\\test\\10001\\183505.jpg",
      "score": 0.6015669107437134
    },
    {
      "row_id": 8,
      "identity": "10001",
      "image_path": "datasets\\celeba_faces\\test\\10001\\191146.jpg",
      "score": 0.6002235412597656
    },
    {
      "row_id": 17,
      "identity": "10001",
      "image_path": "datasets\\celeba_faces\\test\\10001\\197107.jpg",
      "score": 0.5924512147903442
    },
    {
      "row_id": 18,
      "identity": "10001",
      "image_path": "datasets\\celeba_faces\\test\\10001\\198339.jpg",
      "score": 0.5881415605545044
    }
  ]
}
```

## LFW Biometric Verification

LFW is a real biometric verification protocol. These metrics evaluate identity verification quality, not FAISS scalability.

| Metric | Value |
|---|---:|
| Valid pairs | 6000 |
| Skipped pairs | 0 |
| Accuracy | 0.9905 |
| EER | 0.015 |
| EER threshold | 0.154398 |
| FAR at selected threshold | 0.001667 |
| FRR at selected threshold | 0.017333 |
| TAR@FAR=0.1 | 0.988333 |
| TAR@FAR=0.01 | 0.984667 |
| TAR@FAR=0.001 | 0.981 |

## Interpretation

- LFW verification closes the biometric-metrics gap with real image pairs.
- This real-image retrieval benchmark proves that retrieval measurements were also run on embeddings extracted from real images.
- Do not claim 1M real-image vectors unless a real dataset import actually creates 1M real-image embeddings.
