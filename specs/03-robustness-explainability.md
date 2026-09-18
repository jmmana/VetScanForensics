# Spec: EXP-09 (robustez/anti-forensics) y EXP-12 (explicabilidad)

## Contexto

Mismo proyecto que specs/01 y specs/02, léelos primero para el contexto
completo del audit, el confound de resolucion, y la convencion de
`experiment_common.py` / `training/03_train_and_evaluate_detector.py` que
ya existe y debes reutilizar, no reimplementar.

Hoy solo existe el generador DCGAN (`data/fake/`). StyleGAN2-ADA y difusion
se estan entrenando en Kaggle, en paralelo, fuera de esta tarea. Diseña el
codigo para que funcione con cualquier numero de generadores disponibles
(usa `experiment_common.resolve_generators` / la convencion de
`training/GENERATORS.md` ya existente), pero VERIFICA todo con el unico
generador que existe hoy.

## Objetivo

### EXP-09: robustez ante post-procesamiento (`training/09_robustness.py`)

Un atacante no va a enviar el PNG del generador tal cual. Evalua el
detector YA ENTRENADO (usa el checkpoint de `training/checkpoints/
detector_best.pth` si existe y es compatible, o entrena uno nuevo con el
mismo protocolo de `03_train_and_evaluate_detector.py` si no) contra el
conjunto de test, aplicando cada una de estas transformaciones ANTES de
extraer features (tanto a reales como a falsas del set de test, no solo a
las falsas):

- Compresion JPEG en calidades {10, 30, 50, 70, 90, 95} (guarda y vuelve a
  cargar la imagen realmente como JPEG con esa calidad usando PIL, no
  simules el artefacto)
- Resize a {64, 96, 160, 256} y de vuelta a 128 (bicubico)
- Recorte central al {90%, 75%, 60%} del area y luego resize a 128
- Blur Gaussiano con sigma en {0.5, 1.0, 2.0, 3.0}
- Ruido Gaussiano con sigma en {0.01, 0.05, 0.1} (en escala [0,1])
- Ajuste de contraste/brillo/gamma (usa PIL.ImageEnhance, factores {0.7,
  1.3} para contraste y brillo; gamma {0.7, 1.5})
- Sharpening (PIL.ImageFilter.SHARPEN, aplicado 1 y 3 veces)
- Recompresion JPEG multiple (5 pasadas a calidad 70)

Para cada transformacion y nivel de severidad, recalcula accuracy, ROC-AUC,
y balanced accuracy sobre el mismo conjunto de test (holdout real + falsas
de test, sin reentrenar el detector). Guarda:
- `training/exp09_robustness/results.json`: lista de
  `{transform, severity, accuracy, roc_auc, balanced_accuracy, n}`.
- `training/exp09_robustness/results.csv`: mismo contenido, tabular.
- Un script de figura `training/make_exp09_robustness_figure.py` que
  produzca UNA figura con subplots (una curva por familia de
  transformacion: JPEG, resize, blur, noise — las 4 mas importantes segun
  el plan), eje X = severidad, eje Y = accuracy y AUC, usando la paleta
  validada del skill de dataviz ya usada en el resto del proyecto (azul
  #2a78d6 para accuracy, naranja #eb6834 para AUC, o similar, revisa las
  figuras ya generadas en `paper/figures/` para consistencia de estilo).

### EXP-12: explicabilidad (`training/12_explainability.py`)

Aplica Grad-CAM (Selvaraju et al. 2017) a la RAMA ESPACIAL del detector
(el backbone ResNet, no toques la rama de frecuencia con Grad-CAM, no
tiene sentido espacialmente) para un conjunto de imagenes de ejemplo (usa
`captum` o implementa Grad-CAM a mano con hooks de PyTorch, tu decides,
pero verifica que el mapa de calor resultante tenga sentido, es decir, que
no sea uniforme/constante, eso indicaria un bug).

Ademas, genera un mapa de "saliencia de frecuencia": para cada uno de los
32 bins radiales de la rama de frecuencia, calcula el gradiente de la
salida del modelo respecto a ese bin (o usa integrated gradients si
prefieres, documenta cual metodo usaste y por que), para ver que bandas de
frecuencia pesan mas en la decision.

Salida:
- `training/exp12_explainability/gradcam_examples.png`: grid con al menos
  4 ejemplos (2 reales, 2 falsas), cada uno con la imagen original al lado
  de su mapa de calor Grad-CAM superpuesto.
- `training/exp12_explainability/frequency_saliency.png`: grafico de barras
  de importancia por bin de frecuencia (32 barras), con la paleta del
  proyecto.
- `training/exp12_explainability/notas.md`: que observaste (¿el modelo
  mira anatomia, bordes, background, artefactos de generacion?), sin
  sobre-interpretar mas alla de lo que el mapa realmente muestra.

## No-goals

- No toques `paper/main.tex`, no hagas commit ni push.
- No entrenes nada nuevo si ya existe un checkpoint valido y compatible
  con el protocolo actual (holdout real, split seed 42) — reutilizalo. Si
  no existe o es incompatible, entrena uno nuevo siguiendo exactamente el
  protocolo de `03_train_and_evaluate_detector.py` (mismo split, misma
  arquitectura) y dejalo documentado.
- No fabriques resultados para StyleGAN2-ADA/difusion, ambos scripts deben
  fallar con mensaje claro si se les pide un generador que no existe en
  disco, igual que EXP-06/07/08.
- No sobre-interpretes los mapas de Grad-CAM en las notas ("esto prueba
  que..."), describe lo que se ve, nada mas.

## Verificacion antes de terminar

1. Corre EXP-09 completo contra DCGAN, confirma que las curvas tienen
   sentido (a mayor severidad de degradacion, accuracy/AUC deberian bajar
   o mantenerse, no subir sin explicacion).
2. Genera la figura de robustez y verifica visualmente (abrela) que no
   este vacia ni con ejes rotos.
3. Corre EXP-12, genera los mapas Grad-CAM, verifica que no sean
   constantes/uniformes (eso seria un bug de implementacion).
4. Escribe un `training/exp0912_notas.md` con lo que corriste, que
   observaste, y los comandos exactos para correr esto de nuevo cuando
   StyleGAN2-ADA/difusion esten listos.
