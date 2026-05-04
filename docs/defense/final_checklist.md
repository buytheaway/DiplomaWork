# Final Defense Checklist

## Before The Demo

- Use demo or synthetic data only.
- Confirm `.env` values are local and not example placeholders.
- Start backend and check `GET /v1/health`.
- Start desktop client from `desktop/.venv`.
- Prepare one enroll image and one search image.
- Open `docs/benchmarks/retrieval_benchmark_pr2.md`.
- Open the security/privacy section in README or the diploma source.

## Numbers To Show

Use only the tracked PR2 synthetic retrieval benchmark values:

| Size | Method | p50 ms | p95 ms | p99 ms | Build s | Memory MB | top_k_overlap@1 | top_k_overlap@5 | top_k_overlap@10 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | flat | 0.004500 | 0.008610 | 0.021691 | 0.000081 | 0.195355 | 1.000000 | 1.000000 | 1.000000 |
| 100 | hnsw_ef64 | 0.022700 | 0.024515 | 0.034254 | 0.001760 | 0.221331 | 1.000000 | 1.000000 | 1.000000 |
| 100 | ivfpq_nprobe8 | 0.008500 | 0.009210 | 0.009929 | 0.019342 | 0.064320 | 1.000000 | 0.290000 | 0.401000 |
| 1000 | flat | 0.037450 | 0.052320 | 0.059548 | 0.000552 | 1.953168 | 1.000000 | 1.000000 | 1.000000 |
| 1000 | hnsw_ef64 | 0.061900 | 0.070200 | 0.072136 | 0.013336 | 2.212221 | 1.000000 | 0.994000 | 0.997000 |
| 1000 | ivfpq_nprobe8 | 0.019800 | 0.020505 | 0.021358 | 0.070546 | 0.078053 | 0.130000 | 0.056000 | 0.052000 |
| 10000 | flat | 0.616750 | 0.976610 | 1.045482 | 0.005819 | 19.531293 | 1.000000 | 1.000000 | 1.000000 |
| 10000 | hnsw_ef64 | 0.333450 | 0.460055 | 0.472427 | 1.786023 | 22.123201 | 1.000000 | 0.994000 | 0.987000 |
| 10000 | ivfpq_nprobe8 | 0.131150 | 0.148080 | 0.167981 | 0.174341 | 0.215382 | 0.030000 | 0.012000 | 0.009000 |

## Do Not Claim

- Real stable ONNX FAR, FRR, or EER values.
- Confirmed biometric accuracy for the stable MVP.
- A calibrated deployment threshold already exists.
- Validated custom PyTorch model quality.
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
