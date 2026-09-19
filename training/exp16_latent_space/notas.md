# EXP-16: espacio latente

## Ejecución y datos

Conteos completos, sin submuestreo: {'real': 152, 'dcgan': 300, 'diffusion': 300}. Origen: {'gen_pool': 122, 'holdout': 30, 'synthetic': 600}.
Ausentes: ['stylegan2ada']. No se fabricaron puntos.
Checkpoint: `training/checkpoints/attribution_multiclass.pth`; SHA256 `9366d5c8597661a570c4638b38ab4e2eacbd64396e9d83c210b05d528fcae5ce`.
Reutilizado sin entrenar: True.
Motivo de fallback: None.
Comando de entrenamiento original, si hizo falta: None.
Compatibilidad comprobada contra identidad EXP-19: arquitectura, clases en orden,
split seed 42/hash, manifiesto train/val/test y hashes de imágenes; carga estricta de pesos.
Dispositivo de extracción: cpu.

## Metodología

Se importa AttributionModel de 19_generator_attribution.py. Se captura con un pre-hook
la entrada a la última Linear(64,C): fusión 544→64, ReLU, dropout desactivado por eval().
Sin gradientes, sin augmentación de entrenamiento, sin modificar los pesos.
Se usa literalmente RadiographDataset de EXP-03 mediante loader de EXP-19:
grises, roundtrip bicúbico 16..128→128, FFT radial 32 bins, tres canales normalizados [-1,1].
Semilla de preprocesamiento por punto = 42 + índice global del manifiesto (base cero).
Orden: reales gen_pool ordenados por ruta, holdout ordenados, después cada generador
(en orden de classes) con rutas ordenadas. Los índices no son los de los loaders de
val/test de EXP-19: se reutiliza el mismo algoritmo determinista, con otro universo.
El NPZ conserva `preprocessing_seed` para reproducir cada roundtrip.

`embeddings.npz` contiene embeddings float32 N×64, labels enteros, classes,
generator (real para imágenes reales), split (gen_pool/holdout/synthetic),
attribution_split (train/val/test/unused), paths y preprocessing_seed.
`unused` identifica falsas fuera del límite de entrenamiento de EXP-19, si las hay.
metadata_json incrusta procedencia completa; results.json también la conserva.
Las dos coordenadas 2D se guardan en projections.npz. No hay pickle en los NPZ.

Proyecciones ajustadas a TODOS los embeddings crudos, sin estandarizar ni PCA previo;
PCA solo inicializa t-SNE. Las etiquetas solo colorean, no se pasan a los reductores.
Semilla de ambas proyecciones: 42; parámetros completos:

```json
{
  "umap": {
    "a": null,
    "angular_rp_forest": false,
    "b": null,
    "dens_frac": 0.3,
    "dens_lambda": 2.0,
    "dens_var_shift": 0.1,
    "densmap": false,
    "disconnection_distance": null,
    "force_approximation_algorithm": false,
    "init": "spectral",
    "learning_rate": 1.0,
    "local_connectivity": 1.0,
    "low_memory": true,
    "metric": "euclidean",
    "metric_kwds": null,
    "min_dist": 0.1,
    "n_components": 2,
    "n_epochs": 500,
    "n_jobs": 1,
    "n_neighbors": 15,
    "negative_sample_rate": 5,
    "output_dens": false,
    "output_metric": "euclidean",
    "output_metric_kwds": null,
    "precomputed_knn": [
      null,
      null,
      null
    ],
    "random_state": 42,
    "repulsion_strength": 1.0,
    "set_op_mix_ratio": 1.0,
    "spread": 1.0,
    "target_metric": "categorical",
    "target_metric_kwds": null,
    "target_n_neighbors": -1,
    "target_weight": 0.5,
    "tqdm_kwds": {
      "desc": "Epochs completed",
      "bar_format": "{desc}: {percentage:3.0f}%| {bar} {n_fmt}/{total_fmt} [{elapsed}]",
      "disable": true
    },
    "transform_mode": "embedding",
    "transform_queue_size": 4.0,
    "transform_seed": 42,
    "unique": false,
    "verbose": false
  },
  "tsne": {
    "angle": 0.5,
    "early_exaggeration": 12.0,
    "init": "pca",
    "learning_rate": "auto",
    "max_iter": 1000,
    "method": "barnes_hut",
    "metric": "euclidean",
    "metric_params": null,
    "min_grad_norm": 1e-07,
    "n_components": 2,
    "n_iter_without_progress": 300,
    "n_jobs": 1,
    "perplexity": 30.0,
    "random_state": 42,
    "verbose": 0
  }
}
```

Versiones exactas: {'umap-learn': '0.5.12', 'scikit-learn': '1.9.0', 'numpy': '2.5.2', 'numba': '0.67.0', 'scipy': '1.17.1', 'torch': '2.12.0', 'torchvision': '0.27.0', 'matplotlib': '3.10.9', 'pillow': '12.2.0'}. Python: 3.14.7.
UMAP usa un hilo; t-SNE n_jobs=1; PyTorch threads=4.
Semillas fijas permiten repetir bajo el mismo entorno; distintas versiones o dispositivos
pueden producir diferencias numéricas. No se ajustaron parámetros buscando separación.
Paleta categórica del proyecto: real azul #2a78d6, dcgan naranja #eb6834,
diffusion aqua #1baf7a (referencias make_exp09_robustness_figure.py y specs/04).
El skill dataviz no está instalado; se reutiliza su paleta documentada en el proyecto.
StyleGAN2-ADA futuro usa violeta #8064a2; no aparece si está ausente.

## Observaciones visuales

Se abrieron e inspeccionaron ambas figuras de la ejecución completa. No están vacías;
los ejes indican coordenadas arbitrarias de cada método y la leyenda distingue
real (152), dcgan (300) y diffusion (300).

En UMAP aparecen tres agrupamientos mayormente separados: reales abajo, DCGAN a
la derecha y diffusion arriba a la izquierda. No hay una mezcla extensa entre clases,
pero se observa un punto azul real junto al extremo del grupo naranja DCGAN.
No se observa mezcla de reales con el grupo diffusion en esta vista.

En t-SNE vuelve a verse un grupo principal real separado de los dos grupos
sintéticos y un punto real junto a una prolongación del grupo DCGAN. No debe
reportarse separación perfecta: esa excepción aparece en las dos figuras.
Los grupos DCGAN y diffusion también se distinguen entre sí en ambas vistas.

Dentro de diffusion se ven varios lóbulos o concentraciones conectadas; DCGAN
presenta prolongaciones y agrupamientos locales. La nube real principal también
muestra estructura interna. No se infiere un número de sub-clusters estable,
anatomía, modo de generación ni causas del punto aislado a partir de estos mapas.
La forma y la disposición cambian entre métodos; no se comparan sus distancias.
Estas observaciones corresponden exclusivamente a las figuras seed=42 revisadas;
al re-correr, el script regenera esta sección como pendiente para evitar arrastrar
conclusiones visuales a datos o proyecciones diferentes.

## Limitaciones

Se incluyen train, val y test para exploración, no para una métrica de evaluación.
Es un espacio aprendido con supervisión multiclase: algunos puntos ya fueron vistos
por el clasificador. El origen y la partición guardados permiten filtrar después.
UMAP y t-SNE distorsionan distancias globales; los ejes tienen unidades arbitrarias.
No interpretar distancias absolutas entre clusters, tamaños ni áreas como medidas
comparables entre métodos. Agrupamiento relativo y separabilidad solo cualitativos.
No son pruebas cuantitativas de separabilidad: EXP-19 responde esa pregunta con
métricas de test; EXP-16 es un complemento visual. Un sub-cluster no identifica su causa.
El confound de resolución de EXP-13 sigue sin resolver (AUC=1 con y sin mitigación).
La separación visual NO prueba que el detector sea forense y no un shortcut.
Con dcgan/diffusion la atribución sigue siendo parcial; StyleGAN2-ADA está pendiente.

## Reproducción

```bash
.venv/bin/python -m pip install umap-learn scikit-learn
.venv/bin/python training/16_latent_space.py --generators dcgan diffusion --device auto --threads 4 --seed 42
.venv/bin/python -m unittest discover -s training -p test_exp16.py -v
# Reproyectar sin cargar modelo ni extraer features (también permite cambiar --seed):
.venv/bin/python training/16_latent_space.py --embeddings-input training/exp16_latent_space/embeddings.npz --output-dir training/exp16_reprojection --seed 42
# Cuando data/fake_stylegan2ada contenga imágenes de EXP-03:
.venv/bin/python training/16_latent_space.py --generators dcgan stylegan2ada diffusion --output-dir training/exp16_latent_space_three_generators
```

La última orden valida los datos antes de entrenar. Si el checkpoint aún tiene tres
clases, ejecuta el script original EXP-19 con sus defaults (25 épocas, Adam 1e-4,
batch 16, seed 42, selección por menor pérdida de validación) para cuatro clases;
conserva ese informe en exp19_retraining dentro de output-dir. El script EXP-19
sobrescribe attribution_multiclass.pth; los hashes de esta ejecución quedan en el NPZ.
Si ya existe un checkpoint compatible de cuatro clases, se reutiliza sin entrenar.
Si falta un generador explícitamente solicitado, aborta con «Corre primero el generador X».

Referencias: [McInnes et al. (2018), UMAP](https://arxiv.org/abs/1802.03426);
[van der Maaten y Hinton (2008), t-SNE](https://www.jmlr.org/papers/v9/vandermaaten08a.html).

## Verificación realizada

Ejecución completa: 752×64 features finitos, con todas las imágenes disponibles:
122 reales gen_pool + 30 holdout, 300 DCGAN y 300 diffusion.
Particiones EXP-19: 451 train, 153 val, 148 test; ninguna imagen unused en esta ejecución.
Las cuatro pruebas de test_exp16.py pasaron: entrada de 64 dimensiones que reconstruye
los logits con la capa final, extracción determinista, cobertura y procedencia exactas
de archivos en disco, generador ausente sin traceback y artefactos guardados válidos.
El checkpoint se reutilizó, con SHA256 comprobado; no se entrenó ningún modelo.

Se ejecutó también el modo --embeddings-input en /tmp/vetscan_exp16_reprojection,
sin reextraer features. Con semilla 42, ambas proyecciones coinciden exactamente
punto por punto con la ejecución completa (np.testing.assert_array_equal).
