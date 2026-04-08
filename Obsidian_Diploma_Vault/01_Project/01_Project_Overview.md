# Project Overview

## What this project is

DiplomaWork is an operator-oriented biometric face search system.
It combines a FastAPI backend, a PySide6 desktop client, FAISS vector search and a storage layer for metadata and embeddings.

## Practical workflow

1. The operator uploads an image or uses the webcam.
2. The backend detects one or more faces.
3. The system computes embeddings.
4. FAISS searches the nearest vectors.
5. The desktop shows summary, candidates and decision status.

## Main idea

The project deliberately separates:

- embedding extraction;
- ANN retrieval;
- operator UI;
- metadata and embedding storage;
- runtime security and audit.

Because of that, the system is not tied to one model implementation or one interface.

## Main capabilities

- image search;
- person enroll;
- multi-face search;
- strict single-face enroll;
- compare mode;
- live webcam mode;
- database view;
- logs and index maintenance;
- per-pipeline FAISS index management;
- encrypted embedding storage;
- encrypted index snapshots;
- audit logging and retention.

## Main defense point

This is not just a model and not just a script.
It is a working MVP system with backend, desktop client, index layer, storage, operator workflows and a basic security layer.
