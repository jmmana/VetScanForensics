# EXP-09 / EXP-12 — ejecución DCGAN, 2026-09-18

Se ejecutaron completos ambos experimentos en CPU, 4 hilos, con el entorno
`.venv` existente. Único generador disponible: DCGAN, 300 PNG en `data/fake/`.
Se reutilizó `training/checkpoints/detector_best.pth`; **no se entrenó nada**.
No se modificó `paper/main.tex` ni se hizo commit/push.

## Protocolo y checkpoint

Se leyeron specs/01 y specs/02. Se importan splits, arquitectura, FFT,
preprocesamiento y evaluación de `experiment_common.py` / experimento 03.
La única extensión de 03 es un callback opcional después del round-trip y
antes de FFT/tensor; sin callback el comportamiento es idéntico al anterior.
Split seed 42: train 91 reales + 180 falsas; val 31 + 61; test 30 + 59.
Los 30 reales de test son exactamente el holdout. Umbral 0,5; positiva=falsa.

El checkpoint histórico es un state_dict sin metadatos internos. Se comprobó
carga estricta de ResNet18+FFT32, el manifiesto del entrenamiento corregido
(25 épocas, split hash, seed 42, resize 16..128, 300 falsas), tamaños de los
splits y reproducción de accuracy/AUC y pérdida histórica (tolerancia 1e-4
para pérdida por diferencia CPU/MPS). No se seleccionó un nuevo modelo usando
test: este control verifica la identidad funcional del checkpoint histórico.
El registro antiguo no contiene hashes que vinculen los pesos a los archivos
de entrenamiento; esta limitación de procedencia queda explícita en metadata.

SHA256 del checkpoint reutilizado:
`9ae77cac53967ff77c235673fd17cb56a1ed8f7ecb2c9610d64d17ef5b4e6d8d`.
Cada experimento conserva hashes actuales de imágenes/split/checkpoint,
manifiesto train/val/test, dispositivo y evidencia de compatibilidad.
Si un checkpoint falta o no es compatible, `posthoc_common.py` llama a
`experiment_common.train` con los mismos defaults de 03: 25 épocas, Adam
1e-4, batch 16, semilla 42 y selección por AUC de validación. Guarda pesos
con firma del conjunto en `checkpoints/detector_posthoc_<firma>.pth` y un
JSON de procedencia; el segundo experimento los reutiliza. No sobrescribe
los pesos originales ni `metrics.json`. Esa rama no necesitó entrenarse en
esta ejecución porque existe un checkpoint compatible.

## EXP-09: resultados y lectura de curvas

[JSON](exp09_robustness/results.json), [CSV](exp09_robustness/results.csv),
[figura](exp09_robustness/robustness.png).
30 filas: referencia limpia + **29 condiciones de postproceso**. Todas usan
exactamente las mismas 89 imágenes y semillas de preprocesamiento.
Ataques independientes sobre la imagen de 128×128 producida por el protocolo,
antes de ambas ramas; no se acumulan ataques salvo las cinco recomprensiones
JPEG y las repeticiones de sharpening solicitadas. Esto mide postprocesamiento
a resolución de entrada del detector, no ataques sobre archivos reales a su
resolución nativa. Se mantienen el round-trip y su semilla 42+índice.

JPEG se codifica/decodifica realmente con PIL en BytesIO, incluyendo las
cinco pasadas Q70. Recorte usa sqrt(fracción de área), redondeado al píxel.
Ruido usa el mismo campo normal estándar por imagen entre severidades,
escalado por sigma, clip [0,1] y cuantización uint8. Contraste/brillo usan
ImageEnhance; gamma usa una LUT `salida=entrada**gamma` porque ImageEnhance
no tiene ajuste gamma. No se ajusta el umbral ni se reentrena por ataque.

| Condición | Accuracy | ROC-AUC | Balanced accuracy |
|---|---:|---:|---:|
| Limpia | 0.955056 | 1.000000 | 0.966102 |
| JPEG Q10 | 0.786517 | 0.987006 | 0.683333 |
| JPEG Q30 | 1.000000 | 1.000000 | 1.000000 |
| Resize, los cuatro tamaños | 0.955056 | 1.000000 | 0.966102 |
| Blur sigma 3 | 0.404494 | 0.894350 | 0.550847 |
| Ruido sigma 0.01 | 1.000000 | 1.000000 | 1.000000 |
| Ruido sigma 0.1 | 0.662921 | 0.798305 | 0.500000 |
| Recorte 60% del área | 0.910112 | 0.999435 | 0.932203 |
| JPEG Q70, cinco pasadas | 0.955056 | 1.000000 | 0.966102 |

Blur y ruido bajan ambas métricas al aumentar sus sigmas en los niveles
solicitados. Resize conserva accuracy/AUC, aunque los píxeles sí cambian;
64,96,160,256 no constituyen una escala de degradación monótona: mezclan
reducciones y ampliaciones. La figura lo indica y conserva las cuatro medidas.
JPEG tiene el eje de calidad invertido: más degradación hacia la derecha.

**Subidas comprobadas, sin suavizar ni corregir resultados:** JPEG Q30 y ruido
0.01 corrigen exactamente los cuatro falsos negativos originales, sin introducir
errores. AUC ya era 1 y sigue en 1: cambia el cruce del umbral, no la ordenación
perfecta entre clases. Por ejemplo `fake_0157.png` pasa de P(falsa)=0.3953 a
0.6351 con Q30 y 0.5655 con ruido 0.01; las otras tres falsas son
`fake_0129.png`, `fake_0110.png`, `fake_0113.png`. Los cuatro valores iniciales
están entre 0.3446 y 0.4186; todos superan 0.5 con ambos ataques. Esto explica
aritméticamente el aumento, sin atribuir una causa visual que no se midió.
Q10 produce 19 falsos positivos; ruido 0.1 clasifica todo como falso y por eso
su accuracy coincide con la prevalencia 59/89 y balanced accuracy es 0.5.
Blur 3 conserva 30/30 reales correctos pero solo detecta 6/59 falsas.

Sharpen x3 corrige los cuatro falsos negativos y añade un falso positivo:
accuracy 88/89. Contraste/brillo/gamma tampoco son escalas ordinales únicas
al mezclar factores menores y mayores a uno. El pequeño aumento de AUC entre
crop 75% (0.998870) y 60% (0.999435) corresponde a pasar de dos pares real/falsa
mal ordenados a uno, entre 30×59=1770 pares; accuracy sí cae de 82/89 a 81/89.
No hay garantía matemática de monotonía para estas métricas en un test pequeño.
Probabilidades y etiquetas de cada condición están en `metadata.json` para
reproducir todos estos controles.

## EXP-12: observación visual

Se abrieron [Grad-CAM](exp12_explainability/gradcam_examples.png) y
[saliencia FFT](exp12_explainability/frequency_saliency.png), además de la
figura de robustez y una figura existente en `paper/figures/` para comparar
estilo. Se corrigió una flecha que Helvetica no representaba; las figuras
finales usan DejaVu Sans, fondo blanco, azul #2a78d6 y naranja #eb6834.
El heatmap inferno mantiene la familia visual de las figuras espectrales.

Los cuatro mapas son no constantes: std del mapa nativo 0.0350, 0.0669,
0.0855 y 0.1038. Se verifican antes de normalizar para evitar amplificar un
mapa nulo o ruido numérico. Los dos reales muestran zonas amplias de mayor
intensidad sobre el interior del tórax: superior/central en real_0006 y
central/izquierda en real_0009. El fondo negro de la parte superior presenta
menor intensidad. En fake_0177 la región destacada también es central; en
fake_0224 es más extensa hacia la mitad derecha y el borde superior.

Los mapas se solapan con anatomía visible y zonas de textura, pero no permiten
atribuir la decisión a una estructura anatómica particular, bordes concretos
ni artefactos de generación específicos. El mapa nativo es solo 4×4: su
interpolación da regiones suaves y no demuestra reconocimiento anatómico.
No se observa una concentración exclusiva en el fondo negro. No se midió
causalidad por ablación/oclusión y no se concluye qué rama domina la decisión.

Frecuencia: media absoluta del gradiente del logit falsa respecto a los 32
bins normalizados en **todo el test**, no solo los cuatro ejemplos. Bins de
mayor sensibilidad local: 19,17,16,25,1 (índices desde cero). La mezcla de bins
medios/altos y el bin 1 no respalda una conclusión de «solo altas frecuencias».
Es sensibilidad local, no importancia causal ni porcentaje de contribución.
Método y límites detallados en [notas de EXP-12](exp12_explainability/notas.md).

## Verificaciones ejecutadas

- Ambos CLI terminaron con DCGAN y reutilizaron el mismo hash de checkpoint.
- Cuatro pruebas de `test_exp0912.py`: identidad del preprocesamiento limpio,
  ataque antes de ambas ramas para ambas clases, transformaciones/área/ruido,
  recomputación de métricas desde probabilidades y CSV, Grad-CAM no constante
  y rechazo de mapa nulo, gradientes de los 32 bins contra diferencias finitas
  en doble precisión. Todas pasan.
- Seis pruebas existentes de `test_exp0678.py` pasan tras extender el dataset.
- Ambos scripts se invocaron con cada generador ausente (`stylegan2ada`,
  `diffusion`): cuatro salidas con código 2 y «Corre primero el generador X»,
  sin traceback y antes de cargar/entrenar detector.
- Figuras finales abiertas: no vacías, ejes completos, textos legibles.

Comandos desde la raíz (MPLCONFIGDIR evita caché fuera del sandbox):

```sh
MPLCONFIGDIR=/tmp/vetscan-mpl .venv/bin/python training/09_robustness.py --generators dcgan --device cpu
MPLCONFIGDIR=/tmp/vetscan-mpl .venv/bin/python training/make_exp09_robustness_figure.py
MPLCONFIGDIR=/tmp/vetscan-mpl .venv/bin/python training/12_explainability.py --generators dcgan --device cpu
MPLCONFIGDIR=/tmp/vetscan-mpl .venv/bin/python training/test_exp0912.py
MPLCONFIGDIR=/tmp/vetscan-mpl .venv/bin/python training/test_exp0678.py
```

## Cuando lleguen StyleGAN2-ADA / difusión

No se evaluaron ni se fabricaron resultados de esos generadores. Depositar
las imágenes según `GENERATORS.md`, generadas solo desde gen_pool. Ambos CLI
usan `resolve_generators`; sin `--generators` descubren todos los disponibles.
Con varios generadores se mezclan sus falsas y se usa el split original de 03,
con máximo global 3:1, como EXP-07/08. El test no se balancea artificialmente
por generador. EXP-12 muestra hasta dos falsas por generador representado en
ese test; puede no haber muestras de un generador muy pequeño tras el muestreo.

Estos comandos conservan los resultados DCGAN en sus carpetas actuales:

```sh
MPLCONFIGDIR=/tmp/vetscan-mpl .venv/bin/python training/09_robustness.py --generators dcgan stylegan2ada diffusion --device cpu --output-dir training/exp09_robustness/all_generators
MPLCONFIGDIR=/tmp/vetscan-mpl .venv/bin/python training/make_exp09_robustness_figure.py --results training/exp09_robustness/all_generators/results.json --output training/exp09_robustness/all_generators/robustness.png
MPLCONFIGDIR=/tmp/vetscan-mpl .venv/bin/python training/12_explainability.py --generators dcgan stylegan2ada diffusion --device cpu --output-dir training/exp12_explainability/all_generators
```

El checkpoint entrenado solo con DCGAN no representa ese nuevo protocolo
mezclado: si no hay uno compatible, EXP-09 entrenará una vez con los defaults
de 03 y EXP-12 lo reutilizará. Dentro de cada ejecución de robustez nunca se
reentrena entre ataques. Para una evaluación separada por generador se puede
pasar un solo nombre y una carpeta de salida distinta. Esto no sustituye la
matriz de transferencia de EXP-06. `--device auto` permite MPS/CUDA si están
disponibles; la verificación actual se hizo íntegramente en CPU.

El confound de resolución sigue siendo una limitación: EXP-13 obtuvo AUC 1
incluso con la corrección. Estos experimentos no demuestran haberlo eliminado
ni justifican generalización a otros generadores o a radiografías clínicas.
