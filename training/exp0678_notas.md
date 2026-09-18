# EXP-06 / EXP-07 / EXP-08: implementación y verificación

Verificación local: 2026-09-18. No se modificó `paper/main.tex`; no se hizo
commit ni push. Registro de generadores: [GENERATORS.md](GENERATORS.md).
DCGAN se consume desde su alias histórico `data/fake/` (300 PNG).
StyleGAN2-ADA y difusión todavía no están disponibles.

## Entorno y protocolo

Python 3.14, torch 2.12.0, torchvision 0.27.0; CPU, 4 hilos.
Se creó `.venv` con `--system-site-packages` y se instaló PyWavelets 1.10.0.
`requirements.txt` declara scipy y PyWavelets. En otro equipo:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`experiment_common.py` importa `radial_power_spectrum`, `load_real_split`,
`load_dataset_splits` y `run_epoch` de 03. La única extensión en 03 es un
argumento opcional `fake_paths`; omitirlo conserva su comportamiento previo.
Se conserva la arquitectura exacta ResNet18+FFT, incluyendo los pesos ImageNet
V1, normalización [-1,1], resize aleatorio 16..128 ->128, flip en train,
BCEWithLogitsLoss, Adam lr=1e-4, batch=16, semilla 42 y selección por mayor
ROC-AUC de validación. El umbral fijo es 0,5; positiva = falsa.
Se entrenan todos los parámetros. No se elige checkpoint con test.

Estas ejecuciones usaron **2 épocas**, sin recortar el dataset:
train = 91 reales + 180 falsas, val = 31 + 61, test = 30 + 59.
Los reales de test son exactamente el holdout; no se entrenó sobre ellos.
Son verificaciones del pipeline, no resultados finales de 25 épocas ni
prueba de transferencia. FFT-only, DCT-only y wavelet-only aún clasifican
todo como falso a umbral 0,5: accuracy 59/89, especificidad 0, balanced
accuracy 0,5. Sus probabilidades y AUC sí difieren. Spatial-only alcanzó
100% en este test pequeño de DCGAN; no implica generalización fuera de él.

Cada JSON contiene configuración, versiones, manifiesto train/val/test,
hash del split real, historial y época seleccionada, matriz de confusión,
probabilidades y etiquetas. Se conserva en memoria el mejor checkpoint para
la evaluación; estos scripts no exportan pesos. Los JSON se escriben al
terminar exitosamente. `summary.json` consolida la última ejecución exitosa
por variante: comprobar generadores/protocolo antes de comparar filas.
Una repetición sobrescribe el resultado de esa variante.

## Backbones y representaciones

- ResNet18/50, EfficientNet-B0 y ConvNeXt-Tiny: pesos ImageNet-1K V1 de
  torchvision, se reemplaza el clasificador y se conserva el resto.
- `vit_small`: torchvision 0.27.0 no proporciona ViT-Small. Se usa
  **ViT-B/16**, con aviso en consola y `requested/actual/weights/note` en
  resultados. Solo esta rama espacial interpola 128 a 224, resolución de
  sus pesos. La frecuencia sigue calculándose en 128. No interpretar esa
  fila como una medición de ViT-Small. Ver [modelos oficiales de ViT](https://docs.pytorch.org/vision/2.0/models/vision_transformer.html).
  Los pesos que no estén en caché necesitan acceso a download.pytorch.org;
  nunca se sustituye silenciosamente por inicialización aleatoria.
- FFT: función original, 32 coronas radiales, log-potencia, normalización
  min-max por imagen.
- DCT: `scipy.fft.dctn(type=2, norm='ortho')`, log-potencia, radios enteros,
  32 intervalos lineales y min-max, igual que FFT. El origen DC es (0,0),
  no el centro: la DCT ocupa el cuadrante de frecuencias no negativas.
  Se conserva la convención original del límite superior exclusivo.
- Wavelet: `pywt.WaveletPacket2D`, **db2**, nivel 3, `periodization`.
  db2 ofrece soporte compacto y dos momentos nulos; tres niveles dan 64
  subbandas con suficientes coeficientes en imágenes de 128. Se ordenan
  por frecuencia (`order='freq'`), se aplanan por filas y se promedian las
  energías de parejas contiguas (32 valores), luego log1p y min-max.
  Se agrupan parejas para mantener 32 entradas y la misma capacidad de
  la rama. No se simula un radio Fourier: estas subbandas también codifican
  orientación. Ver [WaveletPacket2D](https://pywavelets.readthedocs.io/en/latest/ref/wavelet-packets.html).
- Cada frecuencia individual entra al mismo MLP 32->64->32. `all` usa
  espacial + FFT + DCT + wavelet, concatenando 96 entradas al MLP 96->64->32.
  Las variantes sin espacial no construyen ni descargan un backbone.
  DCT/wavelet se extraen de la imagen aumentada; FFT conserva el orden
  exacto del detector 03 (antes del flip). En evaluación no hay flip.

## Comandos ejecutados

Desde la raíz del repositorio, todos con salida exitosa:

```sh
.venv/bin/python training/07_baseline_architectures.py --backbone resnet18 --generators dcgan --epochs 2 --device cpu
.venv/bin/python training/08_frequency_ablation.py --features spatial --generators dcgan --epochs 2 --device cpu
.venv/bin/python training/08_frequency_ablation.py --features fft --generators dcgan --epochs 2 --device cpu
.venv/bin/python training/06_cross_generator_matrix.py --epochs 2 --device cpu
.venv/bin/python training/08_frequency_ablation.py --features dct --generators dcgan --epochs 2 --device cpu
.venv/bin/python training/08_frequency_ablation.py --features wavelet --generators dcgan --epochs 2 --device cpu
.venv/bin/python training/08_frequency_ablation.py --features all --generators dcgan --epochs 2 --device cpu
.venv/bin/python training/test_exp0678.py
```

Las pruebas verifican identidad de arquitectura/parámetros iniciales y
preprocesamiento con 03, splits idénticos y disjuntos, holdout, perfiles
finitos/distintos, gradientes de las ocho variantes, dimensiones de los cinco
backbones y nueve métricas con una matriz de confusión conocida. La prueba
de dimensiones usa tensores meta y pesos sin cargar: **no cuenta como
entrenamiento de esos backbones** ni produce resultados de rendimiento.

También se ejecutaron los tres CLI con `--generators stylegan2ada`:
salieron con código 2 y «Corre primero el generador stylegan2ada», sin
traceback. La prueba automatizada cubre directorios vacíos mediante mock.
EXP-06 con selección automática detectó solo DCGAN, produjo JSON y CSV 1x1
y documentó explícitamente que aún no mide transferencia. La diagonal
coincide con EXP-07 bajo la misma semilla y configuración.

ROC-AUC y PR-AUC usan probabilidades; PR-AUC es integración trapezoidal de
`precision_recall_curve`, **no average precision**. Especificidad =
TN/(TN+FP); balanced accuracy = (especificidad + TP/(TP+FN))/2.

## Pendiente y comandos cuando lleguen los generadores

No se evaluaron StyleGAN2-ADA/difusión ni la matriz 3x3: faltan sus imágenes.
No se entrenaron ResNet50, EfficientNet-B0, ConvNeXt-Tiny ni ViT-B/16;
solo se comprobaron sus dimensiones. Las fusiones `spatial+fft`,
`spatial+dct`, `spatial+wavelet` de EXP-08 tienen pruebas de gradientes;
la primera además usa el mismo modelo entrenado en EXP-06/07. No se
hicieron todavía corridas CLI individuales de esas tres fusiones en EXP-08.

Depositar imágenes según `GENERATORS.md`, entrenadas exclusivamente con
`gen_pool`. Estos comandos exigen que estén los tres y usan 25 épocas:

```sh
.venv/bin/python training/06_cross_generator_matrix.py --generators dcgan stylegan2ada diffusion --epochs 25

for backbone in resnet18 resnet50 efficientnet_b0 convnext_tiny vit_small; do
  .venv/bin/python training/07_baseline_architectures.py --backbone "$backbone" --generators dcgan stylegan2ada diffusion --epochs 25 || break
done

for features in spatial fft dct wavelet spatial+fft spatial+dct spatial+wavelet all; do
  .venv/bin/python training/08_frequency_ablation.py --features "$features" --generators dcgan stylegan2ada diffusion --epochs 25 || break
done
```

EXP-06 entrena tres modelos (uno por fila), no nueve; evalúa cada uno contra
los tres conjuntos test. Siempre usa la partición test de falsas del destino,
incluso fuera de diagonal, manteniendo igual el conjunto de cada columna.
EXP-07/08 mezclan los generadores seleccionados con el límite global original
3:1 y muestreo determinista, sin balancear artificialmente por generador.
Se puede omitir `--generators` para usar solo los disponibles; usar la lista
explícita anterior evita ejecutar una comparación incompleta por accidente.

## Resultados observados (2 épocas, solo DCGAN)

| Ejecución | Accuracy | ROC-AUC | PR-AUC | Balanced accuracy |
|---|---:|---:|---:|---:|
| exp07_baselines/resnet18 | 0.977528 | 0.997740 | 0.998831 | 0.966667 |
| exp08_frequency/all | 0.988764 | 1.000000 | 1.000000 | 0.983333 |
| exp08_frequency/dct | 0.662921 | 0.722034 | 0.792055 | 0.500000 |
| exp08_frequency/fft | 0.662921 | 0.383051 | 0.560042 | 0.500000 |
| exp08_frequency/spatial | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| exp08_frequency/wavelet | 0.662921 | 0.393220 | 0.592271 | 0.500000 |
| exp06 dcgan → dcgan | 0.977528 | 0.997740 | 0.998831 | 0.966667 |

Tabla extraída de los JSON generados, no estimada. Se verificaron de nuevo
las matrices de confusión, AUC y balanced accuracy contra las probabilidades
y etiquetas guardadas, la ausencia de solapamiento de splits, y la identidad
de las nueve métricas entre CSV, matriz 1x1 y baseline ResNet18.
