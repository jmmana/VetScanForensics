# Canine Thoracic Radiographs — Real vs AI-Generated

Dataset armado para investigar la deteccion de fraude por imagenes de
diagnostico veterinario generadas con IA (radiografias falsas usadas en
reclamos de seguros de mascotas).

## Contenido

- `real/` — 152 radiografias laterales de torax canino, reales, tomadas
  entre 2019 y 2020 en el Hospital Veterinario de la Universidad Autonoma
  de Baja California. Reempaquetadas desde Mendeley Data con permiso
  explicito de la licencia CC BY 4.0.
- `fake/` — Radiografias sinteticas generadas con un DCGAN entrenado sobre
  las imagenes reales de esta misma carpeta (ver `training/` en el repo
  del proyecto: https://github.com/jmmana/VetScanForensics).

## Fuente original y cita obligatoria

Flores Duenas, C.A.; Gaxiola Camacho, S.M.; Montano Gomez, M.F. (2022),
"Radiographic Dataset for VHS determination learning process", Mendeley
Data, V1, doi: 10.17632/ktx4cj55pn.1
https://data.mendeley.com/datasets/ktx4cj55pn/1

Licencia original: CC BY 4.0 (Creative Commons Attribution 4.0
International). Este dataset es una redistribucion con atribucion, tal
como la licencia permite, agregando la carpeta `fake/` como contribucion
nueva.

## Por que existe este dataset

No encontramos ningun dataset publico, en Kaggle o fuera de Kaggle, que
combine radiografias veterinarias reales y generadas por IA para entrenar
detectores de fraude. Toda la investigacion de "medical deepfake
detection" que existe (MedForensics/MICCAI 2025, M3Dsynth, etc.) es
exclusivamente de imagen medica humana. El fraude en seguros de mascotas
es un problema documentado y creciente en la industria aseguradora.
