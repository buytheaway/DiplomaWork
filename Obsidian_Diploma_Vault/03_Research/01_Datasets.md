# Datasets

## Датасеты, которые фигурировали в проекте

- `FaceID-550`
- `Face-Detection-Dataset`
- `Face Dataset Of People That Don't Exist`
- `CelebA`
- `DigiFace-1M`

## Что важно говорить

В проекте датасеты нужны не только для иллюстрации, но и для:

- подготовки training data;
- smoke/eval сценариев;
- демонстрации reproducible dataset protocol.

## Формат данных

Поддерживаемый формат:

`dataset/<identity>/*.jpg`

или разложенный вариант:

`train/<identity>/*.jpg`
`val/<identity>/*.jpg`
`test/<identity>/*.jpg`

## CelebA

Использовался как удобный внешний датасет для training experiments.

## DigiFace-1M

Использовался через подготовленные локальные сценарии и subsets.

## Что важно не перепутать

Datasets и runtime — это разные части проекта:

- datasets нужны для исследований и подготовки;
- runtime pipeline нужен для рабочего поиска.
