# Threshold Calibration Methodology

This document describes the verification metrics used for biometric threshold
calibration. It does not report final numeric biometric results.

## FAR

False Acceptance Rate (FAR) is the fraction of negative pairs that are
incorrectly accepted as matches:

`FAR = false accepts / all negative pairs`

In this project, higher score means more similar, and `score >= threshold` means
match. Lowering the threshold usually increases FAR because more negative pairs
are accepted.

## FRR

False Rejection Rate (FRR) is the fraction of positive pairs that are
incorrectly rejected:

`FRR = false rejects / all positive pairs`

Raising the threshold usually increases FRR because fewer positive pairs are
accepted.

## EER

Equal Error Rate (EER) is the operating point where FAR and FRR are equal or as
close as possible on the evaluated threshold grid. When adjacent curve points
safely bracket the crossing, linear interpolation can be used to estimate the
EER threshold.

EER is useful for comparing verification behavior without choosing an
application-specific threshold in advance. It should not be treated as the final
deployment threshold by itself.

## Threshold Trade-Off

The threshold controls the balance between false accepts and false rejects:

- Lower threshold: more pairs are accepted, FAR may increase, FRR may decrease.
- Higher threshold: fewer pairs are accepted, FAR may decrease, FRR may increase.

The appropriate threshold depends on the application risk model. Security-first
systems usually prefer lower FAR, while convenience-oriented systems may accept
higher FAR to reduce FRR.

## Separation From Retrieval Measurements

Retrieval/index measurements describe vector index behavior: latency, memory
estimate, build time, and indexed-vector counts after embeddings already exist.

Threshold calibration is a separate biometric verification evaluation. It
requires labeled positive and negative biometric pairs. Retrieval latency alone
cannot produce biometric hit@K, FAR, FRR, or EER.

## Current Status

The project now includes metric helpers and unit tests for FAR, FRR, EER,
TAR@FAR, and threshold selection. Final numeric biometric results should be
added only after running the LFW or another labeled verification evaluation
script on a real labeled dataset.

No final FAR, FRR, EER, or biometric hit@K values are reported in this document.
