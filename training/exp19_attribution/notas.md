# EXP-19: atribución parcial

Generadores evaluados: dcgan, diffusion. Ausentes: stylegan2ada.
Esta ejecución corresponde a 2 generadores. Con DCGAN y diffusion es una atribución parcial (2 de 3 generadores);
la conclusión completa depende de EXP-03 (StyleGAN2-ADA). No permite concluir que el fingerprint sobreviva o no sobreviva en general.
Estas métricas son multiclase, no accuracy binaria ni sustitutos de las otras secciones.

## Metodología

Se hereda DualStreamDetector de EXP-03: ResNet18 ImageNet + FFT radial 32 bins,
fusión 544→64, ReLU, dropout 0.3. Solo se reemplaza Linear(64,1) por
Linear(64,3). Cross-entropy sin pesos, logits en entrenamiento,
softmax y argmax en evaluación. Adam, lr=0.0001, batch=16,
25 épocas, semilla PyTorch/NumPy 42, dispositivo cpu.
Se selecciona la menor cross-entropy de validación (primer empate), sin usar test:
época 25.

load_real_split valida el JSON seed 42. common.split_for se llama por generador,
como EXP-06: máximo 3 falsas por real por generador, permutación NumPy 42 y
reparto proporcional por restos mayores. Se conservan exactamente esas falsas,
y una sola copia de los reales: train/val solo gen_pool (75/25), test solo holdout.
A diferencia de EXP-07/08, no se mezclan las falsas antes del límite global:
se usa la variante por generador de EXP-06 para representar cada clase.
Conteos: {'train': {'real': 91, 'dcgan': 180, 'diffusion': 180}, 'val': {'real': 31, 'dcgan': 61, 'diffusion': 61}, 'test': {'real': 30, 'dcgan': 59, 'diffusion': 59}}.
El JSON contiene el manifiesto completo y hashes de datos/split/checkpoint.
No se balancean clases; accuracy depende de estos soportes. Se incluye F1 por clase.

RadiographDataset de EXP-03 aplica a todas las clases escala de grises,
roundtrip bicúbico 16..128→128, FFT y normalización [-1,1] a tres canales.
Train: resize estocástico y flip horizontal. Val/test: resize con semilla 42+índice.
Índices del manifiesto multiclase fijos entre condiciones; no necesariamente coinciden
con los índices de test binario. Las 30 condiciones LEVELS y transform se importan
literalmente de EXP-09: roundtrip → postproceso → ambas ramas, sin reentrenar.
Ruido: 42000+índice (misma realización entre severidades); crop es fracción de área;
JPEG múltiple son 5 recodificaciones reales Q70. Resize no es severidad ordinal.

## Resultados

Accuracy limpia: 1.000000; n=148.
F1 por clase: {'real': 1.0, 'dcgan': 1.0, 'diffusion': 1.0}.
Matriz (filas verdaderas, columnas predichas; orden ['real', 'dcgan', 'diffusion']):
[[30, 0, 0], [0, 59, 0], [0, 0, 59]].

| Transformación | Nivel | Accuracy multiclase |
| --- | --- | --- |
| clean | 0 | 1.0000 |
| jpeg | 10 | 0.9324 |
| jpeg | 30 | 0.9797 |
| resize | 64 | 0.9932 |
| resize | 96 | 1.0000 |
| resize | 160 | 1.0000 |
| resize | 256 | 1.0000 |
| crop | 0.6 | 0.8311 |
| blur | 3.0 | 0.9527 |
| noise | 0.01 | 0.9459 |
| noise | 0.1 | 0.7230 |
| jpeg_multiple | 5 | 0.9932 |

Las matrices, F1, probabilidades y etiquetas de cada condición se guardan para
investigar pérdida de atribución y posibles colapsos a una clase. No se impone
monotonía; con test pequeño hay variación discreta. No se han ajustado
hiperparámetros a los resultados de test.
Descomposición aritmética de cambios respecto al test limpio (no prueba causal
de qué huella visual utiliza la red):

- jpeg 10: corrige 0 errores limpios, introduce 10; recall por clase en el orden declarado: 0.867, 0.932, 0.966.
- jpeg 30: corrige 0 errores limpios, introduce 3; recall por clase en el orden declarado: 0.900, 1.000, 1.000.
- resize 64: corrige 0 errores limpios, introduce 1; recall por clase en el orden declarado: 0.967, 1.000, 1.000.
- resize 96: corrige 0 errores limpios, introduce 0; recall por clase en el orden declarado: 1.000, 1.000, 1.000.
- resize 160: corrige 0 errores limpios, introduce 0; recall por clase en el orden declarado: 1.000, 1.000, 1.000.
- resize 256: corrige 0 errores limpios, introduce 0; recall por clase en el orden declarado: 1.000, 1.000, 1.000.
- crop 0.6: corrige 0 errores limpios, introduce 25; recall por clase en el orden declarado: 0.967, 0.610, 0.983.
- blur 3.0: corrige 0 errores limpios, introduce 7; recall por clase en el orden declarado: 0.833, 1.000, 0.966.
- noise 0.01: corrige 0 errores limpios, introduce 8; recall por clase en el orden declarado: 0.733, 1.000, 1.000.
- noise 0.1: corrige 0 errores limpios, introduce 41; recall por clase en el orden declarado: 0.000, 0.847, 0.966.
- jpeg_multiple 5: corrige 0 errores limpios, introduce 1; recall por clase en el orden declarado: 0.967, 1.000, 1.000.

EXP-13 reportó AUC=1 tanto sin como con la corrección de resize. La mitigación
compartida no demuestra neutralización del shortcut; una atribución alta aquí
no prueba huellas independientes del historial de construcción/resolución.
La figura usa escala secuencial blanco→azul #2a78d6 del proyecto,
normalizada por fila y anotada con conteos. El skill dataviz no está instalado;
la referencia de color es make_exp09_robustness_figure.py y specs/03.

## Reproducción

```bash
.venv/bin/python training/19_generator_attribution.py --generators dcgan diffusion --epochs 25 --batch-size 16 --lr 0.0001 --device auto --threads 4
.venv/bin/python -m unittest discover -s training -p test_exp19.py -v
# Cuando EXP-03 exporte imágenes entrenadas exclusivamente con gen_pool:
.venv/bin/python training/19_generator_attribution.py --generators dcgan stylegan2ada diffusion --output-dir training/exp19_attribution_three_generators
```

El checkpoint separado es training/checkpoints/attribution_multiclass.pth y se
sobrescribe al reentrenar; cada carpeta conserva resultados y manifiestos.
Una petición explícita de un generador ausente aborta antes de entrenar.

## Verificación de esta ejecución

Se completaron 25 épocas en CPU y las 30 condiciones de EXP-09. Las cuatro
pruebas de `test_exp19.py` pasaron, incluida la recomputación de todas las
métricas persistidas desde probabilidades y etiquetas. La petición explícita
`--generators stylegan2ada` devolvió el error esperado sin traceback.
Se abrió `confusion_matrix.png`: matriz 3×3 legible, con diagonal 30/59/59.
Los logs completos se conservan en `run.log` y `tests.log`.

Además de la separación por rutas, se verificaron los SHA-256 de contenido:
ningún archivo de train, val o test comparte contenido idéntico entre particiones.
Esto no descarta similitud visual ni memorización del generador.
Al recargar `attribution_multiclass.pth`, se reprodujeron todas las probabilidades
del test limpio (tolerancia absoluta 1e-7). El checkpoint binario y `paper/main.tex`
conservan exactamente sus SHA-256 previos a esta tarea.

El patrón de robustez no es degenerado: accuracy varía de 0.722973 a 1.000000.
JPEG mejora al aumentar la calidad, y blur y ruido pierden accuracy al aumentar
sigma (con mesetas). Crop 60% confunde 22 de las 59 DCGAN con diffusion.
Con ruido sigma 0.1, los 30 reales se atribuyen a DCGAN (8) o diffusion (22);
además hay 11 confusiones entre generadores. Explica los 41 errores de esa
condición, sin implicar un colapso de todo el test a una única clase.
La exactitud limpia perfecta describe este test pequeño y estos dos generadores;
no elimina la limitación de EXP-13 ni anticipa el resultado pendiente de EXP-03.
