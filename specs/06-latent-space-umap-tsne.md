# Spec: EXP-16 (visualizacion del espacio latente: UMAP + t-SNE)

## Contexto

Mismo proyecto que specs/01-05, leelos primero, en particular
specs/05-generator-attribution.md (EXP-19), que ya entreno un
clasificador multiclase real/dcgan/diffusion cuyo penultimo embedding
(antes de la capa lineal final, 64 dimensiones tras la fusion
espacial+FFT segun `training/19_generator_attribution.py`) es
exactamente lo que este experimento necesita visualizar. No entrenes un
modelo nuevo: reutiliza el checkpoint
`training/checkpoints/attribution_multiclass.pth` de EXP-19 si es
compatible (mismo split, misma arquitectura); si no existe o no es
compatible, entrenalo de nuevo siguiendo el protocolo de
`19_generator_attribution.py` exactamente, documentandolo.

Generadores con imagenes en disco hoy: dcgan, diffusion. StyleGAN2-ADA
sigue sin estar disponible.

## Objetivo

EXP-16 del plan maestro. Crea `training/16_latent_space.py` que:

1. Extrae el embedding de 64 dimensiones (salida de la capa de fusion,
   antes del clasificador final) del modelo de EXP-19 para **todas** las
   imagenes disponibles: reales (gen_pool + holdout, marcadas con su
   split de origen) y falsas de cada generador disponible. Usa el mismo
   preprocesamiento determinista (roundtrip semilla 42+indice) que
   `19_generator_attribution.py`/`03_train_and_evaluate_detector.py`
   para test/val; no necesitas el aumento de datos de entrenamiento aqui,
   esto es solo extraccion de features para visualizar, no una metrica
   de evaluacion con fugas de informacion (documenta explicitamente que
   split usaste para cada punto, para poder filtrar despues si hace
   falta).
2. Proyecta esos embeddings a 2D con **UMAP** (McInnes et al. 2018,
   arXiv:1802.03426) y con **t-SNE** (van der Maaten & Hinton 2008).
   Usa las librerias estandar (`umap-learn`, `scikit-learn.manifold.TSNE`)
   si estan disponibles o instalalas (`pip install -q
   --break-system-packages umap-learn scikit-learn` si scikit-learn no
   esta ya en el entorno). Documenta version exacta y semilla usada en
   ambas proyecciones (no son deterministas sin semilla fija).
3. Genera DOS figuras separadas (una por metodo de proyeccion),
   coloreando cada punto por clase (`real`, `dcgan`, `diffusion`), con
   la paleta categorica validada del skill de dataviz ya usada en el
   resto del proyecto (referencia: `make_exp09_robustness_figure.py`,
   `19_generator_attribution.py` si genero su figura con esa paleta).
   Guarda:
   - `training/exp16_latent_space/umap_projection.png`
   - `training/exp16_latent_space/tsne_projection.png`
   - `training/exp16_latent_space/embeddings.npz` (embeddings crudos +
     labels + generador + split, para poder re-proyectar sin
     re-extraer features)
   - `training/exp16_latent_space/notas.md`: metodologia exacta,
     observaciones honestas (¿se separan los clusters por generador?
     ¿se solapan reales y alguna clase sintetica? ¿hay sub-clusters
     dentro de una clase que sugieran algo, sin sobre-interpretar?),
     limitaciones (UMAP/t-SNE distorsionan distancias globales, no son
     pruebas cuantitativas de separabilidad — esa pregunta ya la
     responde EXP-19 cuantitativamente, esto es un complemento visual,
     dilo explicitamente), y comandos exactos para re-correr con
     StyleGAN2-ADA cuando este disponible.

## No-goals

- No toques `paper/main.tex`, no hagas commit ni push.
- No entrenes un modelo nuevo si el checkpoint de EXP-19 ya sirve.
- No fabriques puntos para StyleGAN2-ADA; si se pide explicitamente con
  `--generators` y falta alguno, aborta con el mismo patron de mensaje
  claro de los scripts anteriores.
- No concluyas en las notas que la separacion visual "prueba" que el
  detector es forense y no un shortcut — sigue siendo el mismo confound
  de resolucion de EXP-13 sin resolver, dejalo dicho explicitamente
  igual que en EXP-19.
- No interpretes distancias absolutas entre clusters en UMAP/t-SNE como
  significativas (limitacion conocida de ambos metodos) — solo agrupamiento
  relativo/separabilidad cualitativa.

## Verificacion antes de terminar

1. Corre el script completo, confirma que produce embeddings para todas
   las imagenes disponibles (reales gen_pool+holdout, dcgan, diffusion).
2. Abre ambas figuras y confirma que no esten vacias, que los ejes
   tengan sentido, y que la leyenda distinga las 3 clases claramente.
3. Escribe pruebas basicas (`training/test_exp16.py`) que verifiquen al
   menos: forma del embedding extraido (64 dims), que el numero de
   puntos por clase coincide con el numero de imagenes en disco, y que
   pedir un generador ausente produce el error esperado sin traceback.
4. Documenta en las notas si los clusters de reales vs. sinteticos se
   separan claramente o se mezclan, sin especular sobre causas mas alla
   de lo que el plot muestra.
