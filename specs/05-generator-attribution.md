# Spec: EXP-19 (atribucion de generador, clasificacion multiclase)

## Contexto

Mismo proyecto que specs/01-04, leelos primero. Hoy hay dos generadores
con imagenes en disco: DCGAN (`data/fake_dcgan`, alias `data/fake`) y
diffusion (`data/fake_diffusion`, 300 imagenes). StyleGAN2-ADA
(`data/fake_stylegan2ada`) todavia no existe — sigue entrenando en
Kaggle. Usa `experiment_common.py` (`resolve_generators`,
`load_real_split`, la arquitectura ResNet18+FFT32, el split seed 42) y
sigue la convencion de `training/GENERATORS.md`. No reimplementes nada
que ya exista ahi.

## Objetivo

EXP-19 del plan maestro: ¿son identificables huellas especificas de cada
generador, mas alla de la deteccion binaria real/falsa? Crea
`training/19_generator_attribution.py` que:

1. Entrena un clasificador **multiclase** (una clase por generador
   disponible, mas la clase `real`) usando la misma arquitectura dual
   stream (ResNet18 backbone + FFT radial de 32 bins) que
   `03_train_and_evaluate_detector.py`, cambiando solo la cabeza final a
   `num_classes = len(generadores_disponibles) + 1` con softmax/
   cross-entropy en vez de la cabeza binaria. Reals de train/val vienen
   de `real_gen_pool`; reals de test vienen de `real_holdout` (mismo
   split seed 42 que el resto del proyecto, nunca mezclar).
2. Con solo 2 generadores disponibles hoy (dcgan, diffusion), esto
   produce una tarea de 3 clases (real/dcgan/diffusion). Escribe el
   codigo para que funcione con cualquier numero de generadores
   (igual que EXP-06/07/08/09/12), pero verifica y reporta con los que
   existen hoy. Si se pide `--generators` con alguno ausente, aborta con
   el mismo patron de mensaje claro usado en los scripts anteriores («
   Corre primero el generador X»), sin fabricar resultados.
3. Reporta accuracy multiclase, matriz de confusion NxN, y F1 por clase
   sobre el test set (holdout real + falsas de test de cada generador,
   siguiendo el mismo split que usan EXP-06/07/08).
4. Evalua **degradacion bajo post-procesamiento**: reutiliza las mismas
   transformaciones y severidades de EXP-09 (`09_robustness.py` /
   `posthoc_common.py`, no las reimplementes, importalas) sobre este
   modelo multiclase, para ver si la atribucion de generador sobrevive
   JPEG, resize, blur y ruido igual de bien/mal que la deteccion binaria.
5. Guarda:
   - `training/exp19_attribution/results.json`: accuracy multiclase,
     matriz de confusion, F1 por clase, y la tabla de degradacion bajo
     post-procesamiento (accuracy multiclase vs. severidad, mismas
     condiciones representativas que la Tabla de EXP-09 del paper: JPEG
     Q10/Q30/Q70x5, resize, crop 60%, blur sigma 3, ruido sigma 0.01 y
     0.1).
   - Una figura `training/exp19_attribution/confusion_matrix.png| con
     la matriz de confusion NxN, paleta validada del skill de dataviz
     (secuencial, no arcoiris).
   - `training/exp19_attribution/notas.md`: metodologia exacta,
     resultados, y comandos para re-correr con los 3 generadores cuando
     StyleGAN2-ADA este disponible.

## No-goals

- No toques `paper/main.tex`, no hagas commit ni push.
- No fabriques una fila para StyleGAN2-ADA.
- No entrenes el detector binario de nuevo ni modifiques
  `training/checkpoints/detector_best.pth` — este es un modelo nuevo y
  separado (multiclase), guardalo en
  `training/checkpoints/attribution_multiclass.pth` (o similar, fuera
  del path del detector binario) si decides persistirlo; recuerda que
  `*.pth` ya esta en `.gitignore`.
- No concluyas "el fingerprint de generador sobrevive/no sobrevive en
  general" a partir de solo 2 de 3 generadores — deja esa conclusion
  explicitamente condicionada y pendiente de EXP-03.
- No mezcles el conjunto de test de este experimento con el usado para
  reportar accuracy binaria en otras secciones; son tareas distintas
  (multiclase vs. binaria) aunque compartan el split de reales.

## Verificacion antes de terminar

1. Corre el script completo con los generadores disponibles hoy
   (dcgan, diffusion), confirma que entrena sin errores y produce una
   matriz de confusion 3x3 coherente (diagonal alta si el modelo
   realmente distingue generadores).
2. Corre la evaluacion de robustez multiclase y confirma que los
   numeros cambian de forma explicable con la severidad (no exijas
   monotonia perfecta, igual que EXP-09, pero si un patron degenerado
   como accuracy=0 o accuracy=1 en todas las condiciones, investigalo
   antes de reportarlo).
3. Abre la figura de matriz de confusion y confirma que no este vacia.
4. Escribe pruebas basicas (`training/test_exp19.py`) que verifiquen al
   menos: forma de salida del modelo multiclase, que el split de reales
   holdout nunca aparece en train/val, y que un generador ausente
   produce el mensaje de error esperado sin traceback.
5. Documenta en las notas que esta es una atribucion parcial (2 de 3
   generadores) y que la conclusion completa depende de EXP-03.
