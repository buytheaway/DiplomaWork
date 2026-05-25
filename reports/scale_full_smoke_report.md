# Scale Full Smoke Report

Date: 2026-05-26

## Runtime

- Database: PostgreSQL `biometric_vggface2_2m_real` via Docker Compose service `db`
- Backend: local Python/FastAPI on `http://127.0.0.1:8000`
- Desktop: local PySide desktop client
- Launcher: `start_scale_full.bat`
- Default pipeline: `custom`
- Available pipelines: `pretrained`, `custom`

## Counts

| Metric | Value |
|---|---:|
| Active identities | 5,741 |
| Active templates | 2,015,076 |
| Custom templates (`torch_insightface_iresnet100`) | 2,015,037 |
| Pretrained templates (`onnx_w600k_r50`) | 39 |
| Custom indexed vectors | 2,015,037 |
| Pretrained indexed vectors | 39 |

## Labels

Dataset-imported identities were renamed from dataset-specific names to neutral gallery labels:

- `Gallery Person 000001` ... `Gallery Person 005736`

Manual profile labels currently present:

- `Abylai Erniyazov`
- `Arnur Shermatov`
- `Damir Zhumabekov`
- `Mukhan Daniyar`
- `Omirgaliyev Ruslan`

No active labels matching `VGG` were returned by `/v1/persons?q=VGG`.

## Clean Enroll Folders

Active folder: `data/new_custom_enroll`

| Label | Clean images kept |
|---|---:|
| Abylai Erniyazov | 10 |
| Arnur Shermatov | 5 |
| Damir Zhumabekov | 8 |
| Mukhan Daniyar | 10 |
| Omirgaliyev Ruslan | 6 |

Rejected/extra folders:

- `data/new_custom_enroll_rejected`
- `data/new_custom_enroll_rejected_pretrained`
- `data/new_custom_enroll_extra`

## Manual Profile Embeddings

| Label | Custom embeddings | Pretrained embeddings |
|---|---:|---:|
| Abylai Erniyazov | 11 | 10 |
| Arnur Shermatov | 19 | 5 |
| Damir Zhumabekov | 16 | 8 |
| Mukhan Daniyar | 12 | 10 |
| Omirgaliyev Ruslan | 6 | 6 |

## Same-Photo Smoke

Single-image smoke used the first clean image per label with `multi_face=false`, `k=3`.

| Label | Pipeline | Decision | Best label | Best score | Latency ms | detect ms | faiss/search ms | db ms |
|---|---|---|---|---:|---:|---:|---:|---:|
| Abylai Erniyazov | custom | match | Abylai Erniyazov | 1.0000 | 186.03 | 89.04 | 82.05 | 14.90 |
| Abylai Erniyazov | pretrained | match | Abylai Erniyazov | 1.0000 | 542.48 | 369.86 | 0.27 | 172.31 |
| Arnur Shermatov | custom | match | Arnur Shermatov | 1.0000 | 365.04 | 257.69 | 87.03 | 20.18 |
| Arnur Shermatov | pretrained | match | Arnur Shermatov | 1.0000 | 464.41 | 351.76 | 3.31 | 109.17 |
| Damir Zhumabekov | custom | match | Damir Zhumabekov | 1.0000 | 240.83 | 167.15 | 54.41 | 19.23 |
| Damir Zhumabekov | pretrained | match | Damir Zhumabekov | 1.0000 | 496.11 | 272.36 | 0.28 | 223.43 |
| Mukhan Daniyar | custom | match | Mukhan Daniyar | 1.0000 | 274.81 | 176.55 | 82.55 | 15.64 |
| Mukhan Daniyar | pretrained | match | Mukhan Daniyar | 1.0000 | 596.70 | 406.40 | 10.84 | 179.42 |
| Omirgaliyev Ruslan | custom | match | Omirgaliyev Ruslan | 1.0000 | 214.62 | 113.36 | 84.26 | 16.95 |
| Omirgaliyev Ruslan | pretrained | match | Omirgaliyev Ruslan | 1.0000 | 443.14 | 280.53 | 10.18 | 152.39 |

Note: this smoke used full-resolution stored photos. Live webcam sends resized JPEG frames, so live latency is expected to differ.

## Experiment Separation

- The current Desktop scale database uses real-image-derived embeddings plus manual enrolled profiles.
- LFW verification is the real biometric verification experiment.
- Real-image embedding benchmarks and scale database checks are reported separately from LFW biometric verification.
