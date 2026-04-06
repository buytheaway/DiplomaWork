# Risks and Limitations

## Текущие ограничения

- Не все custom model paths доведены до production-grade состояния.
- Live webcam mode работает как near real-time, а не как полный video streaming pipeline.
- Исследовательская training-ветка и runtime-ветка не всегда полностью совпадают по operational maturity.
- Качество comparative branch требует отдельной валидации и benchmark.

## Как это правильно озвучивать

Не нужно скрывать ограничения. Правильная формулировка:

“Рабочий baseline и системная архитектура уже реализованы и воспроизводимы. Часть comparative и research веток находится в исследовательском состоянии и служит для дальнейшего развития проекта.”

## Что не надо обещать

- не говорить, что все модели уже production-ready;
- не говорить, что webcam работает как полноценный industrial surveillance stream;
- не смешивать training experiments и stable runtime в одну и ту же формулировку.
