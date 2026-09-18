# Spec: EXP-06 (cross-generator), EXP-07 (baselines), EXP-08 (forense de frecuencia)

## Contexto

Proyecto de deteccion forense de radiografias veterinarias falsas. Ya existe
un detector dual-stream (ResNet18 + FFT radial de 32 bins) en
`training/03_train_and_evaluate_detector.py`, entrenado con un DCGAN sobre
un split con holdout real (`data/splits/real_split.json`: 122 gen_pool, 30
holdout). Actualmente solo hay UN generador (DCGAN, `data/fake/`).

En paralelo (fuera de esta tarea, corriendo en Kaggle) se estan entrenando
dos generadores mas: StyleGAN2-ADA (EXP-03) y un modelo de difusion (EXP-04).
Sus salidas AUN NO EXISTEN en este repo. Tu trabajo es dejar listo el CODIGO
que consume esas salidas en cuanto lleguen, no ejecutar nada que dependa de
ellas todavia.

## Objetivo

Preparar, sin poder correrlos aun contra datos reales de los 3 generadores,
los scripts para:

1. **EXP-07**: comparar 5 arquitecturas de backbone espacial (ResNet18 ya
   existente, ResNet50, EfficientNet-B0, ConvNeXt-Tiny, ViT-Small) usando la
   misma rama de frecuencia y el mismo protocolo de entrenamiento.
2. **EXP-08**: ablation de dominio de frecuencia (spatial-only, FFT-only,
   DCT-only, wavelet-only, y las combinaciones fusionadas) sobre el MISMO
   detector base.
3. **EXP-06**: dado un conjunto de generadores disponibles en `data/fake_*/`
   (uno por generador, ver convencion abajo), entrenar el detector con
   datos de un generador y evaluarlo contra los demas, construyendo la
   matriz completa train-generador x test-generador con las 9 metricas
   pedidas (accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC,
   MCC, balanced accuracy).

## Convencion de carpetas para multi-generador (IMPORTANTE, disenala tu)

Hoy solo existe `data/fake/` (DCGAN). Para que EXP-03/04 puedan simplemente
volcar su salida sin que tengas que tocar nada despues, define y documenta
una convencion clara, por ejemplo:

```
data/fake_dcgan/       # ya existe como data/fake/, puedes renombrar o symlink
data/fake_stylegan2ada/
data/fake_diffusion/
```

Escribe esa convencion en un archivo `training/GENERATORS.md` corto, y haz
que los 3 scripts nuevos (06, 07, 08) la lean desde ahi (una lista de
generadores disponibles, no hardcodeada en cada script).

## No-goals

- No entrenes nada contra generadores que no existen todavia (StyleGAN2-ADA,
  difusion). Si `data/fake_stylegan2ada/` o `data/fake_diffusion/` no
  existen o estan vacias, los scripts deben fallar con un mensaje claro
  ("Corre primero el generador X"), no con un traceback críptico.
- No toques `paper/main.tex`.
- No hagas commit ni push.
- No inventes ni asumas metricas para generadores que no has corrido. Todo
  numero debe salir de una ejecucion real.
- Puedes y debes probar el pipeline completo usando el UNICO generador que
  SI existe hoy (DCGAN, `data/fake/`), tratándolo como si fuera el unico
  generador disponible, para verificar que el codigo corre sin errores de
  principio a fin. Eso es exactamente lo que se espera que hagas para
  validar tu propio trabajo antes de terminar.

## Diseno

### `training/GENERATORS.md`
Documenta la convencion de carpetas y como agregar un generador nuevo.

### `training/07_baseline_architectures.py`
- Reutiliza `radial_power_spectrum`, `load_real_split`, y la logica de
  split de `03_train_and_evaluate_detector.py` (impórtalas, no las
  copies/dupliques).
- Backbone parametrizable: ResNet18 (ya existe), ResNet50, EfficientNet-B0,
  ConvNeXt-Tiny, ViT-Small (via `torchvision.models`, usa los pesos
  preentrenados en ImageNet disponibles en torchvision, si un modelo no
  esta disponible en la version de torchvision instalada, documentalo y
  usa el mas cercano disponible, no falles silenciosamente).
- Argumento `--backbone {resnet18,resnet50,efficientnet_b0,convnext_tiny,vit_small}`.
- Argumento `--generators` (lista de nombres de `training/GENERATORS.md`,
  default: todos los disponibles actualmente en disco).
- Salida: `training/exp07_baselines/<backbone>_metrics.json` por backbone
  corrido, y al final un `training/exp07_baselines/summary.json` que
  consolida todos los backbones corridos exitosamente en una tabla.

### `training/08_frequency_ablation.py`
- Mismo detector dual-stream, pero parametrizable en que features usa:
  `--features {spatial,fft,dct,wavelet,spatial+fft,spatial+dct,spatial+wavelet,all}`.
- Para DCT usa `scipy.fft.dctn` o equivalente sobre la imagen, con el mismo
  esquema de binning radial que ya usa `radial_power_spectrum` para FFT
  (adaptalo, no inventes un esquema totalmente distinto sin explicar por
  que).
- Para wavelet usa `pywt` (PyWavelets, instalalo si falta) con una wavelet
  packet decomposition razonable (documenta cual elegiste y por que).
- Salida: `training/exp08_frequency/<features>_metrics.json` por variante,
  y `training/exp08_frequency/summary.json` consolidado.

### `training/06_cross_generator_matrix.py`
- Lee generadores disponibles de `training/GENERATORS.md` / carpetas en
  disco.
- Para cada par (train_generator, test_generator) donde ambos existen en
  disco, entrena el detector (arquitectura base: ResNet18 + FFT, la actual)
  con real (gen_pool/holdout como ya esta) + fake del train_generator, y
  evalua contra real_holdout + fake del test_generator.
- Si solo hay 1 generador disponible (el caso de hoy), la matriz es 1x1;
  documenta esto en el output, no lo trates como error.
- Calcula las 9 metricas pedidas: accuracy, precision, recall, specificity,
  F1, ROC-AUC, PR-AUC, MCC, balanced accuracy (usa sklearn donde exista
  funcion directa; para specificity y balanced accuracy usa formulas
  correctas a partir de la matriz de confusion, no aproximaciones).
- Salida: `training/exp06_cross_generator/matrix.json` con la matriz
  completa (aunque sea 1x1 por ahora) y un CSV legible
  `training/exp06_cross_generator/matrix.csv`.

## Verificacion que debes hacer antes de terminar

1. Corre `07_baseline_architectures.py` con `--backbone resnet18` (el que
   ya sabemos que funciona) contra el unico generador disponible hoy
   (DCGAN) y confirma que produce metricas sensatas (no errores).
2. Corre al menos 2 variantes de `08_frequency_ablation.py`
   (`spatial`, `fft`) contra DCGAN y confirma que corren sin error y dan
   numeros distintos entre si (si dieran identicos, hay un bug, investigalo
   antes de terminar).
3. Corre `06_cross_generator_matrix.py` y confirma que produce la matriz
   1x1 correctamente con las 9 metricas, y que documenta explicitamente que
   solo hay 1 generador disponible por ahora.
4. Si algo falla, arreglalo tu mismo iterando, no me entregues codigo sin
   probar. Documenta en un `training/exp0678_notas.md` que corriste, que
   funciono, y que quedo sin poder probar por falta de los otros 2
   generadores (para que cuando esos existan, yo sepa exactamente que
   comando correr).
