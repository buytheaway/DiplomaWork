# Custom Runtime Migration Report

Date: 2026-05-24

## Goal

Integrate the new custom Torch embedding model into the current database without mixing embedding spaces.

Final runtime candidate:

- checkpoint: `custom_torch_candidate_bundle/model.pth`
- architecture: `insightface_iresnet100`
- runtime model name: `torch_insightface_iresnet100`
- preprocessing: `runtime_fallback_center_crop`
- color order: `RGB`
- normalization: `[-1, 1]`
- TTA: `hflip`
- threshold: `0.205047`

## Database Inspection

Current database:

`backend/backend_local_test3.db`

Tables:

- `persons`: `id`, `label`, `status`, `created_at`, `updated_at`
- `embeddings`: `id`, `person_id`, `model`, `dim`, `vector`, `created_at`, `is_active`, `pipeline`
- `index_snapshots`: `id`, `index_type`, `params`, `path`, `embeddings_count`, `created_at`
- `audit_logs`: `id`, `event_type`, `actor_role`, `route`, `status_code`, `details`, `created_at`

No table stores source image paths or raw uploaded image bytes.

Audit log path search:

- `.jpg`: 0 rows
- `.jpeg`: 0 rows
- `.png`: 0 rows
- `.webp`: 0 rows
- `image_path`: 0 rows
- `file_path`: 0 rows
- `photo_`: 0 rows
- `C:`: 0 rows
- `OneDrive`: 0 rows

Local project image artifacts outside datasets/cache were only smoke/test images:

- `custom_pipeline_bundle/smoke_inputs/single_face.jpg`
- `custom_pipeline_bundle/smoke_inputs/multiple_faces.jpg`
- `custom_pipeline_bundle/smoke_inputs/no_face.png`
- `test_face.jpg`

These files are not linked to existing `persons` or `embeddings` rows.

## Active Embeddings

Active embeddings by pipeline/model after candidate runtime smoke:

| Pipeline | Model | Active embeddings |
|---|---|---:|
| custom | `torch_insightface_iresnet100` | 1 |
| custom | `torch_ir50` | 9244 |
| pretrained | `onnx_w600k_r50` | 9293 |
| pretrained | `torch_ir50` | 17 |

The single `torch_insightface_iresnet100` row is the runtime smoke enrollment:

- label: `Candidate Runtime Smoke`
- embeddings: 1

## Migration Decision

Old `torch_ir50` embeddings cannot be migrated to `torch_insightface_iresnet100` because only embedding vectors are stored. A vector produced by one face model cannot be converted into the embedding space of another model.

Correct migration requires original images:

```text
person_id,label,image_path
...
```

or re-enrollment/import through the new runtime extractor.

## Index Separation

Old custom index:

- model: `torch_ir50`
- path family: `backend/data/index/custom.*.faiss`

New custom candidate index:

- model: `torch_insightface_iresnet100`
- path family: `backend/data/index/custom_candidate.*.faiss`
- latest vectors: 1
- latest `.faiss` file size: 2627 bytes
- latest `.faiss.map.json` file size: 77 bytes

The old and new vector spaces are not mixed in one FAISS index.

## Verified Runtime Smoke

Backend health:

- status: `ok`
- default pipeline: `custom`
- available pipelines: `custom`
- model: `torch_insightface_iresnet100`

Smoke checks:

| Check | Result |
|---|---|
| Custom enroll | 200 OK |
| Same-photo custom search | match, score `1.0` |
| Different-photo custom search | unknown, best score `-0.0275` |
| Live-style custom search | match, `search_mode=live_fast`, `fallback_reason=null` |

Latency:

| Request | Total ms | Detect/embed ms | FAISS ms | DB ms |
|---|---:|---:|---:|---:|
| same manual search | 48.92 | 40.10 | 4.61 | 4.17 |
| different manual search | 41.31 | 39.49 | 0.37 | 1.42 |
| same live-style search | 41.79 | 40.22 | 0.18 | 1.36 |

## Required Next Flow

To migrate real people to the new custom model:

1. Keep all old `torch_ir50` rows active for historical compatibility.
2. Prepare a CSV or folder import with source images:

   ```text
   person_id,label,image_path
   <existing person id>,Mukhan Daniyar,C:\path\to\photo1.jpg
   <existing person id>,Mukhan Daniyar,C:\path\to\photo2.jpg
   ```

3. Run a new batch re-embed importer that:
   - reuses existing `persons.id`;
   - creates new `embeddings` rows with:
     - `pipeline=custom`
     - `model=torch_insightface_iresnet100`
   - does not delete old embeddings;
   - does not touch old `custom.*.faiss` snapshots.
4. Rebuild only the new candidate custom index path:
   - `backend/data/index/custom_candidate.faiss`
5. Verify:
   - active `torch_insightface_iresnet100` count;
   - custom candidate index count;
   - same-photo match;
   - different-photo unknown;
   - live webcam behavior;
   - latency breakdown.

