# Data and Models

## What exists in the project

The project includes several model and dataset directions.

## Runtime pipelines

The runtime logic exposes:

- `pretrained`
- `custom`
- `compare`

### Pretrained

The baseline pipeline is the stable operational reference.

### Custom

The custom branch is the comparative and research branch. It should be described as experimental until valid training logs and labeled verification results are available.

### Compare

Compare runs both pipelines on the same input.

## Training subsystem

The separate `training/` branch is used for:

- dataset preparation;
- custom model training;
- evaluation;
- export and benchmarking.

## `deploy/model_bundle`

`deploy/model_bundle` is a local external artifact bundle.

Typical contents:

- `train.csv`
- `faiss.index`
- `meta.json`
- `best.pt`

Important note:

- treat it as an external local artifact, not as the trusted runtime source of truth;
- do not keep real gallery and index artifacts in GitHub if they may expose biometric data or proprietary weights.

## Safe wording

Safe defense wording:

- the baseline pipeline is stable and used as the operational reference;
- the custom branch exists for comparative and research tasks;
- the architecture supports multiple embedding sources and multiple runtime paths;
- custom model quality and stable ONNX biometric accuracy require real labeled evaluation before numerical claims.
