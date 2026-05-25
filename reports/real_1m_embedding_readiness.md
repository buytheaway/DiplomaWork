# Real 1M/2M Embedding Readiness

Generated: 2026-05-24

## Current Local Image Inventory

The current machine does not contain enough real face images to produce a
truthful 1,000,000 or 2,000,000 real-image embedding benchmark.

| Source | Type | Images found | Use |
|---|---|---:|---|
| `datasets/celeba_faces` | real face images | 193,569 | real-image embedding retrieval / training data |
| `handoff_lfw_eval/lfw` | real labeled verification images | 13,233 | LFW biometric verification only |
| `data/new_custom_enroll` | local re-enroll photos | 83 | runtime demo enrollment |
| `C:\Users\mukha\OneDrive\Documents\GitHub\datasets\digiface1m_small` | synthetic face images | 360 | optional synthetic-style scale support, not real biometric evidence |

Local real-image total available for extraction is approximately `206,885`
images. This supports real-image embedding retrieval benchmarks at 10k, 50k,
100k, or about 200k scale, but it does not support a truthful 1M/2M
real-image claim.

## What Can Be Claimed Now

- LFW provides real biometric verification metrics: FAR, FRR, EER,
  TAR@FAR, threshold behavior, and pair verification accuracy.
- The real-image embedding benchmark uses embeddings extracted from real image
  files and measures FAISS build/search behavior on real model outputs.
- The synthetic 1M/2M benchmark measures FAISS scalability only.

## What Cannot Be Claimed Now

- Do not claim that 1,000,000 or 2,000,000 real face images were evaluated.
- Do not claim that the synthetic 1M/2M benchmark proves biometric accuracy.
- Do not present LFW `0.990500` accuracy as desktop/live webcam identification
  accuracy. LFW is pair verification; desktop search is end-to-end
  identification.

## Requirement For A Truthful 1M/2M Real-Image Benchmark

To build a real 1M or 2M embedding benchmark, add a real face dataset with at
least 1,000,000 or 2,000,000 accessible face images. A suitable candidate is
VGGFace2 because the paper reports 3.31 million images of 9131 subjects. The
dataset must be stored outside git, for example:

```text
datasets/vggface2/
  train/
    identity_000001/
      image_000001.jpg
      ...
```

## Located Dataset Source

Candidate source found:

- Dataset: `ProgramComputer/VGGFace2` on Hugging Face.
- File: `data/vggface2_train.tar.gz`.
- Remote size: `37.9 GB`.
- Reported license on the dataset page: `cc-by-nc-4.0`.
- Target local path: `D:\datasets\vggface2_hf\data\vggface2_train.tar.gz`.

Download was started outside the git workspace with resume/retry enabled:

```powershell
curl.exe -L --fail --retry 10 --retry-delay 10 --continue-at - `
  --output D:\datasets\vggface2_hf\data\vggface2_train.tar.gz `
  https://huggingface.co/datasets/ProgramComputer/VGGFace2/resolve/main/data/vggface2_train.tar.gz
```

The public unauthenticated download is slow. If it stalls or takes too long,
set an authenticated Hugging Face token or use a Kaggle source with clear local
access credentials. Downloaded archives and extracted images must remain outside
git.

After the dataset is present, run 1M:

```powershell
python scripts\benchmark_real_image_embeddings.py `
  --sources datasets\vggface2 `
  --output-dir reports\real_image_embedding_benchmark_1m `
  --max-images 1000000 `
  --batch-size 512 `
  --n-queries 100 `
  --top-k 10 `
  --index-type hnsw
```

Run 2M:

```powershell
python scripts\benchmark_real_image_embeddings.py `
  --sources datasets\vggface2 `
  --output-dir reports\real_image_embedding_benchmark_2m `
  --max-images 2000000 `
  --batch-size 512 `
  --n-queries 100 `
  --top-k 10 `
  --index-type hnsw
```

Expected resource note: the current 10k real-image benchmark took about
`365 s` for extraction. A 1M run may take many hours on this machine; a 2M run
can take roughly twice as long and needs enough disk for raw embeddings and the
FAISS index. HNSW on 2M can also be RAM-heavy; if memory pressure becomes a
problem, use `--index-type ivfpq` and report the lower-memory index separately.

## Desktop Accuracy Boundary

The observed desktop/live accuracy around `79-80%` is an end-to-end runtime
identification observation, not the same metric as LFW verification accuracy.
If desktop accuracy is required as a formal number, it needs its own labeled
runtime protocol:

1. keep a separate test set not used for enrollment;
2. run `Search -> Custom` on every test image;
3. count correct label, no-match, and false-match outcomes;
4. report top-1 identification accuracy and false-match behavior separately
   from LFW FAR/FRR/EER.
