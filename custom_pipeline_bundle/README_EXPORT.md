# Custom Pipeline Bundle

Portable model bundle for external backend integration with no code changes.

## Included
- .env.custom (+ .env.custom.example)
- requirements-ml-<backend>.txt (runtime dependencies)
- manifest.json (machine-readable model contract)
- checksums.sha256 (artifact integrity)
- validate_bundle.py (bundle integrity + config check)
- smoke_runtime.py (health + single/no_face/multiple/stability checks)
- smoke_inputs/ (ready test images for runtime smoke checks)
- hnsw/ (faiss.index + meta.json)
- models/ (backend model files)

## Preflight (1-2 minutes)
1) Install runtime deps:
   pip install -r model_bundle/requirements-ml-onnx.txt
2) Run validator (works inside target project too):
   python model_bundle/validate_bundle.py --bundle_dir model_bundle --require_manifest --check_onnx_output
3) Start backend and check health:
   curl http://127.0.0.1:8000/v1/health
4) Run runtime smoke tests (replace <EMBED_ENDPOINT>):
   python model_bundle/smoke_runtime.py --embed_path <EMBED_ENDPOINT> --single_face_image model_bundle/smoke_inputs/single_face.jpg --no_face_image model_bundle/smoke_inputs/no_face.png --multiple_faces_image model_bundle/smoke_inputs/multiple_faces.jpg

Expected in /v1/health:
- available_pipelines contains custom
- unavailable_pipelines is {}

Generated from: C:\Users\qwety\OneDrive\Рабочий стол\Diploma AI