# Benchmarking

## Что именно можно измерять

- latency поиска;
- `top_k_overlap@K` against exact Flat search for synthetic retrieval;
- biometric verification metrics such as FAR, FRR, EER, and TAR@FAR after labeled pair evaluation;
- difference between pipelines;
- difference between index types.

## Что уже есть в проекте

- compare mode в backend и desktop;
- index stats;
- benchmark script для retrieval;
- tracked synthetic benchmark artifacts in `docs/benchmarks/retrieval_benchmark_pr2.md`;
- stable extractor pair evaluator in `scripts/evaluate_verification_pairs.py`;
- training/export tooling.

## Почему benchmark важен

Проект про быстрый поиск, а не только про сам факт распознавания. Поэтому надо сравнивать:

- retrieval overlap and latency for synthetic vectors;
- скорость;
- стоимость разных index configurations.

Important wording:

- `top_k_overlap@K` is not biometric identification hit@K.
- The synthetic benchmark is not biometric accuracy.
- FAR, FRR, and EER require labeled positive and negative biometric pairs.
- Stable ONNX/InsightFace accuracy should not be claimed until the pair evaluator is actually run.

## Что удобно говорить на защите

“Мы сравниваем не только модели, но и индексные конфигурации, потому что practical biometric search определяется и качеством embedding, и качеством approximate nearest-neighbor retrieval.”
