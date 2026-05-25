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
| Total real images scanned | 3141890 |
| Images attempted | 100000 |
| Embeddings created | 96070 |
| Skipped no-face | 0 |
| Skipped invalid | 0 |
| Skipped multiple-faces | 3930 |
| Extractor errors | 0 |
| Embedding dim | 512 |
| Index type | `hnsw` |
| Index size MB | 213.308 |
| Extraction time s | 4250.146 |
| Build time s | 4.369 |
| Search p50 ms | 0.1224 |
| Search p95 ms | 0.338335 |
| Search p99 ms | 0.380692 |

Output paths:

- embeddings: `C:\Users\mukha\OneDrive\Documents\GitHub\DiplomaWork\reports\real_image_embedding_benchmark_vggface2_100k\real_image_embeddings.float32.memmap`
- mapping: `C:\Users\mukha\OneDrive\Documents\GitHub\DiplomaWork\reports\real_image_embedding_benchmark_vggface2_100k\real_image_mapping.csv`
- index: `C:\Users\mukha\OneDrive\Documents\GitHub\DiplomaWork\reports\real_image_embedding_benchmark_vggface2_100k\real_image_hnsw.faiss`

Sample query result:

```json
{
  "query_row_id": 0,
  "query_identity": "n000002",
  "query_image_path": "D:\\datasets\\vggface2_hf\\extracted\\train\\n000002\\0001_01.jpg",
  "top_ids": [
    0,
    48,
    5,
    2,
    1,
    90,
    10,
    143,
    42,
    58
  ],
  "top_scores": [
    1.0,
    0.7554681301116943,
    0.7529345750808716,
    0.7337208986282349,
    0.7299247980117798,
    0.7235591411590576,
    0.7178374528884888,
    0.6828323602676392,
    0.6800453662872314,
    0.6799793243408203
  ],
  "top_results": [
    {
      "row_id": 0,
      "identity": "n000002",
      "image_path": "D:\\datasets\\vggface2_hf\\extracted\\train\\n000002\\0001_01.jpg",
      "score": 1.0
    },
    {
      "row_id": 48,
      "identity": "n000002",
      "image_path": "D:\\datasets\\vggface2_hf\\extracted\\train\\n000002\\0042_01.jpg",
      "score": 0.7554681301116943
    },
    {
      "row_id": 5,
      "identity": "n000002",
      "image_path": "D:\\datasets\\vggface2_hf\\extracted\\train\\n000002\\0006_01.jpg",
      "score": 0.7529345750808716
    },
    {
      "row_id": 2,
      "identity": "n000002",
      "image_path": "D:\\datasets\\vggface2_hf\\extracted\\train\\n000002\\0003_01.jpg",
      "score": 0.7337208986282349
    },
    {
      "row_id": 1,
      "identity": "n000002",
      "image_path": "D:\\datasets\\vggface2_hf\\extracted\\train\\n000002\\0002_01.jpg",
      "score": 0.7299247980117798
    },
    {
      "row_id": 90,
      "identity": "n000002",
      "image_path": "D:\\datasets\\vggface2_hf\\extracted\\train\\n000002\\0085_01.jpg",
      "score": 0.7235591411590576
    },
    {
      "row_id": 10,
      "identity": "n000002",
      "image_path": "D:\\datasets\\vggface2_hf\\extracted\\train\\n000002\\0011_01.jpg",
      "score": 0.7178374528884888
    },
    {
      "row_id": 143,
      "identity": "n000002",
      "image_path": "D:\\datasets\\vggface2_hf\\extracted\\train\\n000002\\0142_01.jpg",
      "score": 0.6828323602676392
    },
    {
      "row_id": 42,
      "identity": "n000002",
      "image_path": "D:\\datasets\\vggface2_hf\\extracted\\train\\n000002\\0035_01.jpg",
      "score": 0.6800453662872314
    },
    {
      "row_id": 58,
      "identity": "n000002",
      "image_path": "D:\\datasets\\vggface2_hf\\extracted\\train\\n000002\\0054_01.jpg",
      "score": 0.6799793243408203
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
