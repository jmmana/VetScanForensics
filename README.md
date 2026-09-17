# VetScanForensics

**Deteccion de radiografias veterinarias generadas por IA para fraude en seguros de mascotas**
**AI-Generated Veterinary Radiograph Detection for Pet Insurance Fraud**

> **Articulo cientifico** — Maestria en Inteligencia Artificial · Universidad de La Salle, Bogota, Colombia
> **Autor:** Juan Manuel Castillo Pinto · jmmana@gmail.com
> **Paper (Grimorio):** pendiente de publicar en tesis.grimorio.dev
> **Dataset (Kaggle):** https://www.kaggle.com/datasets/jmmana/vet-radiographs-real-vs-ai-generated
> **Repo:** https://github.com/jmmana/VetScanForensics

---

## 1. El problema

El fraude en seguros de mascotas le cuesta a las aseguradoras entre 3 y 7
puntos de loss ratio (300 mil a 700 mil dolares por cada 10 millones en
primas), en buena parte porque las historias clinicas veterinarias no
viajan entre clinicas: no hay forma de verificar el historial de una
mascota entre distintos proveedores. A eso se suma un riesgo nuevo:
generadores de imagenes por IA (GAN, modelos de difusion) ya pueden crear
radiografias sinteticas suficientemente convincentes como para enganar a
radiologos humanos, segun el estudio de RSNA publicado en marzo de 2026.

Toda la investigacion existente sobre deteccion de "deepfakes medicos"
(MedForensics/MICCAI 2025, M3Dsynth, y el paper de deepfakes de GAN vs.
difusion en cancer de piel) es exclusivamente de imagen medica **humana**.
No existe, hasta donde revisamos, ningun dataset ni metodo publicado para
imagen de diagnostico **veterinaria**.

## 2. La idea

1. Partir de un dataset real y legalmente reutilizable (CC BY 4.0) de
   radiografias caninas de torax.
2. Entrenar un generador (DCGAN) sobre esas mismas radiografias para
   producir versiones sinteticas realistas ("fakes").
3. Entrenar un detector de dos ramas (espacial + frecuencia) que aprenda a
   distinguir reales de falsas, inspirado en la arquitectura de
   MedForensics/DSKI pero simplificado a dos senales.
4. Republicar el dataset resultante (reales + falsas) en Kaggle, con
   atribucion correcta a la fuente original, para que quede disponible
   para quien quiera seguir esta linea.
5. Documentar todo en un articulo cientifico.

## 3. Dataset base

Flores Duenas, C.A.; Gaxiola Camacho, S.M.; Montano Gomez, M.F. (2022),
"Radiographic Dataset for VHS determination learning process", Mendeley
Data, V1, doi: [10.17632/ktx4cj55pn.1](https://data.mendeley.com/datasets/ktx4cj55pn/1),
licencia CC BY 4.0. 152 radiografias laterales de torax canino, formato
PNG, recolectadas en 2019-2020 en el Hospital Veterinario de la
Universidad Autonoma de Baja California.

Mendeley sirve la descarga real por JavaScript (no hay endpoint REST
estable), asi que el primer paso del pipeline es manual (ver abajo).

## 4. Pipeline paso a paso

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Descarga manual (una sola vez):
#    abre https://data.mendeley.com/datasets/ktx4cj55pn/1
#    click en "Download all 152 files", mueve el zip a data/raw/
python training/01_prepare_real_data.py

# 2. Entrena el generador y crea las radiografias falsas
python training/02_train_generator_and_make_fakes.py --epochs 800 --n-fake 300

# 3. Entrena y evalua el detector real-vs-falsa
python training/03_train_and_evaluate_detector.py --epochs 25

# 4. Figura de espectro de frecuencia (evidencia visual del paper)
python training/04_frequency_analysis_figure.py

# 5. Sube el dataset (reales + falsas) a Kaggle
bash kaggle/upload.sh
```

## 5. Estructura del proyecto

```
VetScanForensics/
├── data/
│   ├── raw/          # zip descargado manualmente de Mendeley
│   ├── real/         # 152 radiografias reales, ya preparadas
│   └── fake/         # radiografias sinteticas generadas por el DCGAN
├── training/
│   ├── 01_prepare_real_data.py
│   ├── 02_train_generator_and_make_fakes.py
│   ├── 03_train_and_evaluate_detector.py
│   ├── 04_frequency_analysis_figure.py
│   └── metrics.json          # resultados finales (se genera al entrenar)
├── kaggle/
│   ├── dataset-metadata.json
│   ├── README.md
│   └── upload.sh
└── paper/
    ├── main.tex
    └── figures/
```

## 6. Limitaciones honestas

- 152 imagenes reales es un dataset chico. El generador y el detector son
  una prueba de concepto reproducible, no un sistema listo para produccion
  clinica o para litigio de seguros.
- El generador es un DCGAN, no un modelo de difusion. Con mas computo, el
  siguiente paso natural es fine-tunear un modelo de difusion (LoRA/
  DreamBooth) sobre las mismas 152 imagenes y comparar que tan bien
  generaliza el detector entre familias de generadores (GAN vs. difusion),
  que es exactamente la brecha que la literatura humana ya identifico como
  dificil.
- El dataset base cubre una sola vista radiografica (latero-lateral de
  torax) y una sola especie (canina). Extender a ecografia, otras vistas o
  gato queda como trabajo futuro.

## 7. Citas y referencias clave

Ver `paper/main.tex`, seccion de referencias. Incluye: Mendeley dataset
original, MedForensics/DSKI (MICCAI 2025, arXiv:2509.15711), estudio RSNA
de radiografias falsas (marzo 2026), y reportes de fraude en seguros de
mascotas.
