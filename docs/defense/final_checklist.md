# Final Defense Checklist

## Before The Demo

- Use consented demo identities and sanitized reports; do not expose private raw images or secrets.
- Confirm `.env` values are local and not example placeholders.
- Start backend and check `GET /v1/health`.
- Start desktop client from `desktop/.venv`.
- Prepare one enroll image and one search image.
- Open the security/privacy section in README or the diploma source.

## Numbers To Show

Use the tracked LFW labeled-pair results for biometric verification quality:

| Pipeline | Valid pairs | EER | Best accuracy | TAR@FAR=0.01 |
|---|---:|---:|---:|---:|
| Final custom `torch_insightface_iresnet100` | 6000 | 0.015000 | 0.990500 | 0.984667 |
| Pretrained ONNX/InsightFace baseline | 5957 | 0.027852 | 0.984556 | 0.971141 |

Use the real-image embedding benchmark to show that retrieval measurements were
also run on embeddings extracted from real face image files:

| Real images scanned | Attempted | Embeddings created | Index type | Index size MB | p95 ms | p99 ms |
|---:|---:|---:|---|---:|---:|---:|
| 206802 | 10000 | 9802 | HNSW | 21.76 | 0.194645 | 0.338031 |

## Do Not Claim

- The final custom model is guaranteed to outperform every external model in every operating condition.
- 1M real biometric images were evaluated unless a real-image benchmark at that scale is actually run.
- Full DDoS protection.
- Full RBAC.
- Hard purge implementation.
- Liveness or spoofing resistance.
- Results for a larger run that was not actually executed.

## Manual Document Conversion

Pandoc was not required for the markdown defense package. If Pandoc is
installed, convert the final diploma source with:

```powershell
pandoc docs\diploma\PreDefense_Diploma_2026_source.md `
  -f markdown+pipe_tables `
  -o docs\diploma\PreDefense_Diploma_2026.docx
```

If Pandoc is unavailable, install it with:

```powershell
winget install --id JohnMacFarlane.Pandoc -e
```

Alternative manual path:

1. Open `docs/diploma/PreDefense_Diploma_2026_source.md`.
2. Copy the content into Word.
3. Replace `[[PAGE_BREAK]]` markers with page breaks.
4. Update Table of Contents, List of Figures, and List of Tables in Word.
5. Export DOCX or PDF from Word.

## Final Sanity Checks

- `python -m pytest backend\tests -q`
- `python -m pytest training -q`
- `python -m ruff check backend training scripts`
- `python -m compileall -q backend\app desktop\app training scripts`
- `git status --short`
