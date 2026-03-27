# ONNX Preflight Checklist

1) Files exist:
- models/det_10g.onnx
- models/w600k_r50.onnx
- hnsw/faiss.index
- hnsw/meta.json

2) Config set:
- ENABLE_CUSTOM_PIPELINE=true
- CUSTOM_BACKEND=onnx
- CUSTOM_INDEX_PATH=/app/model_bundle/hnsw
- ONNX_DETECTOR_PATH=/app/model_bundle/models/det_10g.onnx
- ONNX_EMBEDDER_PATH=/app/model_bundle/models/w600k_r50.onnx
- EMBEDDING_DIM=512

3) Install runtime dependencies:
pip install -r model_bundle/requirements-ml-onnx.txt

4) Validate bundle:
python model_bundle/validate_bundle.py --bundle_dir model_bundle --require_manifest --check_onnx_output

5) Runtime smoke checks:
python model_bundle/smoke_runtime.py --embed_path <EMBED_ENDPOINT> --single_face_image model_bundle/smoke_inputs/single_face.jpg --no_face_image model_bundle/smoke_inputs/no_face.png --multiple_faces_image model_bundle/smoke_inputs/multiple_faces.jpg

6) Runtime check:
GET /v1/health => available_pipelines contains custom, unavailable_pipelines == {}