# Benchmarking

## Что именно можно измерять

- latency поиска;
- recall@k;
- top-k quality;
- difference between pipelines;
- difference between index types.

## Что уже есть в проекте

- compare mode в backend и desktop;
- index stats;
- benchmark script для retrieval;
- training/export tooling.

## Почему benchmark важен

Проект про быстрый поиск, а не только про сам факт распознавания. Поэтому надо сравнивать:

- точность;
- скорость;
- стоимость разных index configurations.

## Что удобно говорить на защите

“Мы сравниваем не только модели, но и индексные конфигурации, потому что practical biometric search определяется и качеством embedding, и качеством ANN retrieval.”
