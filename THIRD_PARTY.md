# Сторонние компоненты

## Модель YuNet — `data/models/face_detection_yunet.onnx`

Лёгкая модель детекции лиц YuNet (проект libfacedetection). Файл взят из репозитория depthai_yunet (`models/build/face_detection_yunet.onnx`).
Лицензия этого репозитория — MIT, copyright 2021 geax — см. `third_party/depthai_yunet-LICENSE.txt`.
Лицензию исходной модели смотрите в апстриме YuNet / libfacedetection.

Формат модели: три выхода `loc`, `conf`, `iou`. Декодирование якорей — в `src/yunet-detector.hpp`,
эталонная реализация на Python — в `tests/yunet_reference.py`.

## OpenCV

Используются модули `core`, `imgproc`, `dnn` (Apache-2.0).
