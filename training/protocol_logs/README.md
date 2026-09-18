# EXP-01 / EXP-02 / EXP-13 / EXP-15

## Protocolo ejecutado

Los comandos y semillas estan en `run_manifest.json`; las salidas completas se
conservan en `generator.log`, `detector.log` y `shortcut_audit.log`.
El entrenamiento usa MPS fuera del sandbox, donde PyTorch detecta el dispositivo.

- Split persistente: 122 reales en gen_pool y 30 en holdout, semilla 42.
- GAN desde cero: 800 epocas, exclusivamente gen_pool.
- Se eliminan las 2300 falsas anteriores antes de generar las 300 nuevas
  del valor por defecto de `--n-fake` en el comando solicitado. No se mezclan
  muestras del GAN antiguo con las del GAN nuevo.
- Detector: 25 epocas; reales train/val/test = 91/31/30. Las falsas se
  distribuyen proporcionalmente con redondeo por restos mayores.
- Round-trip bicubico uniforme entero 16..128, salida 128, para ambas clases.
  Train obtiene semillas del RNG de PyTorch por acceso; val/test usa 42+indice.
  FFT y rama espacial reciben la misma imagen corregida antes del flip de train.
- EXP-13 reutiliza DualStreamDetector, radial_power_spectrum, run_epoch y las
  rutas de resize de EXP-00. Ejecuta 20 epocas por brazo con identicos grupos,
  splits, inicializacion y seleccion por mejor AUC de validacion.
- En EXP-13 primero se construye el historial A (directo) o B (8x8->128), luego
  se aplica el round-trip comun, antes de FFT/tensor. Aplicarlo antes de crear
  el historial no evaluaria si borra una huella ya presente.

## Limites de interpretacion

El rango prescrito 16..128 no garantiza eliminar la perdida de informacion
causada por la semilla 8x8. La auditoria mide esa limitacion sin retocar el rango
ni los hiperparametros para favorecer un resultado. Un AUC alto con correccion
impide declarar neutralizado el shortcut, aunque el detector real/falsa mejore.
El test de la auditoria contiene solo 31 imagenes; no demuestra equivalencia
con el azar mediante una sola estimacion puntual.

El protocolo viejo se conserva tal como fue medido, sin reentrenarlo. La
comparacion no aisla el efecto causal de una unica modificacion: cambian el
holdout, el entrenamiento del GAN, el preprocesamiento, los splits y el numero
de falsas disponibles (300 nuevas vs. 456 muestreadas anteriormente).
El GAN conserva el comportamiento previo sin semilla fija de entrenamiento;
la semilla 42 fija el split, y se fija tambien para detector/auditoria.

## Verificaciones de implementacion

Se comprobaron cobertura total y ausencia de overlap del split, idempotencia,
filtrado de 122 reales frente a 152 sin --split-file, disjuncion de train/val/test,
preprocesamiento repetible en validacion y variable en entrenamiento para
ambos datasets. Arquitectura y FFT del detector se compararon por AST con HEAD
sin cambios. Los hashes de las metricas antiguas y paper/main.tex estan en el
manifiesto para verificar que no se alteraron.

## Resultados finales

Todos los comandos del pipeline finalizaron con codigo 0 en este Mac, usando MPS.

| Medida | Protocolo viejo | Protocolo nuevo |
| --- | ---: | ---: |
| AUC de test | 1.000000 | 1.000000 |
| Accuracy | 1.000000 | 0.955056 |
| Precision | 1.000000 | 1.000000 |
| Recall | 1.000000 | 0.932203 |
| F1 | 1.000000 | 0.964912 |
| N train / val / test | 388 / 98 / 122 | 271 / 92 / 89 |

Nuevo test: 30 reales del holdout y 59 falsas nuevas. Reales correctamente
clasificadas: 30/30; falsas correctamente clasificadas: 55/59.

EXP-13: **AUC sin correccion = 1.0; AUC con correccion = 1.0** (accuracy 1.0
en ambos brazos). Se replica EXP-00, pero el round-trip prescrito no neutraliza
el shortcut en este experimento. No debe presentarse el nuevo AUC 1.0 como
libre del confound de resolucion. No se modificaron rango ni hiperparametros
para bajar las metricas. El holdout esta implementado; la neutralizacion no
esta demostrada y la auditoria aporta evidencia en contra de su eficacia.

La comparacion legible por maquina esta en `../exp15_protocol_comparison.json`.
Se verificaron los hashes originales del split y de las metricas antiguas.
Las figuras ROC/matriz de confusion fueron actualizadas por el detector.

## Cambio externo concurrente

La verificacion final detecto que el hash de `paper/main.tex` habia cambiado.
Se confirmo que el hash inicial coincide con el archivo del commit `871ac1e`
y el archivo actual coincide exactamente con el commit externo `3b501fc`
(2026-09-18 09:03:07 -0500, reestructuracion del paper). Esta tarea no edito
`paper/main.tex`, no ejecuto commit ni push y no revirtio ese trabajo externo.
El manifiesto conserva ambos hashes y la explicacion.
