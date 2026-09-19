# EXP-17 — calidad de imagen

Ejecución UTC: 2026-09-19T01:45:28.637665+00:00. Finalizada UTC: 2026-09-19T01:50:25.107928+00:00.
Bibliotecas: torchmetrics 1.9.0, torch-fidelity 0.4.0, torch 2.12.0, torchvision 0.27.0, numpy 2.5.2, Pillow 12.2.0. Python: 3.14.7 (main, Aug  5 2026, 10:29:49) [Clang 21.0.0 (clang-2100.1.1.101)]. Extracción: cpu;
cálculo FID/KID en CPU float64, features extraídas en float32. Hilos: 4.

## Resultados de esta ejecución

| Generador | Reales | Falsas | FID | KID media | KID std | FID bootstrap percentiles 2.5–97.5% |
|---|---:|---:|---:|---:|---:|---|
| dcgan | 122 | 300 | 326.381176 | 0.35892338 | 0.00597098 | [332.093214, 344.957431] |
| diffusion | 122 | 300 | 336.399032 | 0.40337579 | 0.00871308 | [338.638337, 357.037317] |

Generadores ausentes (sin métricas): stylegan2ada.
KID se muestra en unidades crudas, sin multiplicar por 100/1000.

En esta corrida, el intervalo bootstrap no contiene el FID puntual para: dcgan, diffusion. No interpretarlo como intervalo calibrado ni usarlo para probar diferencias entre generadores.

## Metodología exacta

- `experiment_common.resolve_generators()` lee el JSON de `training/GENERATORS.md`.
  Se incluyen todas las imágenes PNG/JPG/JPEG del directorio resuelto, sin
  limitar ni equilibrar el número de falsas. DCGAN acepta el alias `data/fake`.
- Referencia exclusiva: `gen_pool` de `/Users/juancastillo/Documents/WarlockCode/VetScanForensics/data/splits/real_split.json`, semilla del split
  42. `load_real_split` valida exhaustividad, duplicados y ausencia
  de overlap con holdout. Las imágenes del holdout no se cargan ni entran en
  features, FID, KID o bootstrap. Su uso se reserva a evaluación del detector.
  El entrenamiento solo con gen_pool es la condición del registro de generadores;
  las imágenes exportadas por sí solas no prueban la procedencia del entrenamiento.
- PIL `convert('L')` a gris de 8 bits, resize bilinear determinista a 128×128
  para todos los conjuntos (mismo tamaño objetivo que el generador), sin crop,
  augmentación ni round-trip aleatorio del detector. Repetir el canal gris tres
  veces produce tensores RGB uint8 [0,255]. Se conserva la deformación de aspecto
  del resize cuadrado; esto y el historial de resolución condicionan las métricas.
- `torch_fidelity.FeatureExtractorInceptionV3('inception-v3-compat', ['2048'])`
  en modo eval: pesos oficiales convertidos de TensorFlow, pool final de 2048
  dimensiones; resize interno bilinear compatible con TensorFlow a 299×299,
  normalización interna `(x-128)/128`. Sin antialias en el resize interno,
  ni normalización torchvision adicional; equivale al extractor de TorchMetrics
  FID configurado con `antialias=False` (su default en 1.9.0 es True).
  Se extrae una sola vez por imagen y se reutiliza el mismo tensor para ambas
  métricas. Pesos y listas de imágenes tienen SHA-256 en `results.json`.
- FID: `torchmetrics.image.fid.FrechetInceptionDistance` con adaptador identidad
  mediante la API pública `feature=Module` sobre esos features de Inception.
  La biblioteca calcula medias, covarianzas y distancia; no se implementa FID
  desde cero ni se reduce la dimensión para ocultar la escasez de datos.
- KID: `torchmetrics.image.kid.KernelInceptionDistance`, mismos features;
  kernel polinómico `(x·y/2048 + 1)^3`, estimador MMD² sin sesgo, 100
  submuestras de 100 reales y 100 falsas.
  Tamaño común a todos los generadores: mínimo del solicitado (100),
  n_real y el menor n_fake. Muestreo sin reemplazo dentro de cada submuestra;
  se permiten repeticiones entre submuestras. Semilla 42 reiniciada por
  generador. `kid_std` es la desviación poblacional entre estas submuestras
  (implementación de TorchMetrics), **no un intervalo de confianza ni un error
  estándar**; las submuestras se solapan. El estimador KID puede ser negativo.
- FID bootstrap: 100 réplicas, semilla 42, remuestreo
  independiente con reemplazo de features reales y falsas, manteniendo los dos
  tamaños originales. Cada réplica usa la misma clase de TorchMetrics. Intervalo
  percentil exploratorio del 95%, cuantiles NumPy con interpolación lineal;
  las réplicas completas quedan en JSON. No es un intervalo corregido por sesgo;
  puede desplazarse respecto al FID original y no contenerlo. Tampoco representa
  variabilidad entre entrenamientos/semillas del generador.

## Limitaciones e interpretación

**No hay pares correspondientes real–sintética.** SSIM (Wang et al., 2004),
PSNR y LPIPS (Zhang et al., 2018) requieren pares para la interpretación estándar.
No se inventan correspondencias por orden, azar o vecino más cercano:
`paired_metrics: null` y `ssim`, `psnr`, `lpips: null`, con razón explícita en JSON.

**Muestra muy pequeña.** Los conteos efectivos están en la tabla. El umbral de
~2000 por conjunto es una advertencia práctica, no una garantía ni un corte
teórico de validez. Con 122 reales y 300 falsas, FID tiene fuerte sesgo de muestra
finita e inestabilidad; las covarianzas de dimensión 2048 son de rango deficiente
(a lo sumo n−1). Calcular en float64 ayuda numéricamente, pero no resuelve la
limitación estadística. El bootstrap tampoco corrige el sesgo ni crea nueva
información; su cobertura con esta muestra no está garantizada. KID aporta un
estimador sin sesgo de MMD², pero puede tener alta varianza. No establecer un
ranking concluyente por diferencias pequeñas ni comparar ejecuciones con
distintos tamaños/pipelines como si fueran equivalentes.

**Dominio de Inception.** Inception fue entrenado con imágenes naturales de
ImageNet, no radiografías veterinarias. Repetir gris como RGB satisface el formato,
pero no elimina ese desajuste. FID/KID son señales relativas bajo este protocolo;
sus valores absolutos no son comparables a benchmarks de imágenes naturales.
Comparar contra imágenes de entrenamiento no mide generalización generativa,
no descarta memorización y no demuestra diversidad ni validez clínica/anatómica.

Un FID menor o mayor no prueba que un generador sea indistinguible ni fácil de
detectar: esa pregunta corresponde al detector y a EXP-05/06, bajo su protocolo
de holdout. No se infiere rendimiento forense a partir de estas métricas.

## Reproducción (desde la raíz del repositorio)

Instalación utilizada, con versiones fijadas para repetirla:

```bash
.venv/bin/python -m pip install -q --break-system-packages torchmetrics==1.9.0 torch-fidelity==0.4.0
```

Comando ejecutado:

```bash
.venv/bin/python training/17_image_quality.py --device cpu --seed 42 --kid-subsets 100 --kid-subset-size 100 --fid-bootstraps 100
```

Detección automática de todos los generadores disponibles (también al llegar StyleGAN2-ADA):

```bash
.venv/bin/python training/17_image_quality.py --device cpu --seed 42 --kid-subsets 100 --kid-subset-size 100 --fid-bootstraps 100
```

Depositar las imágenes de StyleGAN2-ADA entrenado solo con gen_pool en
`data/fake_stylegan2ada/`. Para exigir los tres y abortar con mensaje legible si falta alguno:

```bash
.venv/bin/python training/17_image_quality.py --generators dcgan stylegan2ada diffusion --device cpu --seed 42 --kid-subsets 100 --kid-subset-size 100 --fid-bootstraps 100
```

Cada ejecución exitosa reemplaza `results.json` y estas notas en `--output-dir`
(por defecto `training/exp17_image_quality`). Los pesos se descargan la primera
vez a `training/checkpoints/exp17/checkpoints/`; se necesita red solo si faltan.
No se genera figura; la tabla evita añadir una comparación visual que sugiera
precisión injustificada. No se modifica el manuscrito ni se entrena un detector.

## Verificación

Después de la corrida completa:

```bash
.venv/bin/python training/test_exp17.py
```

La prueba contrasta el adaptador de features con las APIs directas de imágenes
de TorchMetrics usando el mismo resize interno (FID `antialias=False` y KID con
ese extractor); tolerancias absolutas 2e-4 para FID y 2e-5 para KID por el cálculo
float64 frente a la salida float32 directa. También audita el JSON: referencia
exacta gen_pool sin holdout, conteos reales, métricas finitas, percentiles de las
réplicas y métricas pareadas nulas. Si existe un generador ausente, comprueba el
mensaje de error sin traceback al solicitarlo explícitamente.

## Referencias y código de las bibliotecas

- Heusel et al. (2017), FID: https://arxiv.org/abs/1706.08500
- Chong y Forsyth (2020), sesgo de muestra finita de FID: https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.html
- Binkowski et al. (2018), KID: https://arxiv.org/abs/1801.01401
- Wang et al. (2004), SSIM: https://doi.org/10.1109/TIP.2003.819861
- Zhang et al. (2018), LPIPS: https://arxiv.org/abs/1801.03924
- TorchMetrics (FID/KID): https://github.com/Lightning-AI/torchmetrics/tree/v1.9.0/src/torchmetrics/image
- Extractor: https://github.com/toshas/torch-fidelity/blob/v0.4.0/torch_fidelity/feature_extractor_inceptionv3.py
