# Spec: EXP-01 / EXP-02 / EXP-13 — holdout real y corrección del confound de resolución

## Contexto (léelo completo antes de tocar código)

Este es un proyecto de detección forense de radiografías veterinarias
falsas generadas por IA. El pipeline actual tiene un DCGAN entrenado
sobre 152 radiografías reales (`data/real/*.png`, resoluciones nativas
grandes y heterogéneas, ~1500-2800px, no cuadradas) que genera imágenes
falsas nativamente en 128x128 (`data/fake/*.png`, ya hay 2300 generadas).
El detector actual (`training/03_train_and_evaluate_detector.py`) llega a
100% de accuracy/AUC, pero una auditoría encontró la causa probable: las
reales se reducen de ~2000px a 128px (un filtro pasa-bajos fuerte), las
falsas nacen nativamente en 128px, y esa diferencia de "historial de
resize" es aprendible por sí sola.

**Ya se demostró esto empíricamente** en
`training/00_resolution_confound_test.py`: un detector entrenado para
distinguir dos grupos de imágenes **ambas 100% reales** (grupo A: reducidas
directo a 128px; grupo B: reducidas a 8x8 y luego escaladas de vuelta a
128px, imitando cómo un GAN construye una imagen desde una semilla
diminuta) llega a 100% de AUC. Léelo para entender el patrón de
`radial_power_spectrum` y `DualStreamDetector` que ya existe — reutiliza
esas mismas funciones, no las reinventes.

## Objetivo de este spec

1. Separar un holdout real que el GAN nunca vea durante su entrenamiento
   (EXP-01).
2. Corregir el pipeline de preprocesamiento del detector para neutralizar
   el confound de resolución demostrado en EXP-00 (EXP-13).
3. Reentrenar todo con el pipeline corregido y reportar las métricas
   honestas, sin forzar ningún resultado (EXP-02).
4. Generar una comparación explícita: protocolo viejo (con el confound,
   sin holdout) vs. protocolo nuevo (corregido) — ambos números deben
   quedar documentados, no se oculta el resultado anterior (EXP-15).

## No-goals (no hagas esto)

- No toques `paper/main.tex` — eso lo hace Claude después con los
  números que tú produzcas.
- No entrenes StyleGAN2-ADA, difusión, ni descargues VetXRay — eso es
  trabajo de otra fase (Kaggle GPU), fuera de alcance aquí.
- No cambies la arquitectura del `DualStreamDetector` (ResNet18 + rama de
  frecuencia FFT de 32 bins) — solo el pipeline de datos/preprocesamiento
  y el split.
- No borres ni sobreescribas `training/metrics.json` (las métricas viejas)
  — cópialas primero a `training/metrics_old_protocol.json` antes de que
  cualquier script las sobreescriba, las necesitamos para la comparación
  del punto 4.
- No subas nada a Kaggle ni hagas push/commit — eso lo hace Claude
  después de revisar el diff.

## Diseño exacto

### 1. `training/01b_split_holdout.py` (script nuevo)

- Lee todos los archivos en `data/real/` (glob `*.png`, `*.jpg`).
- Con `numpy.random.default_rng(seed=42)`, separa aleatoriamente en:
  - `real_gen_pool`: ~80% (ej. 122 de 152)
  - `real_holdout`: ~20% restante (ej. 30 de 152)
- Guarda el resultado en `data/splits/real_split.json` con esta forma
  exacta:
  ```json
  {
    "seed": 42,
    "gen_pool": ["real_0000.png", "real_0003.png", ...],
    "holdout": ["real_0001.png", "real_0002.png", ...]
  }
  ```
- Imprime en stdout: cuántas imágenes en cada grupo, y confirma que
  `len(gen_pool) + len(holdout) == len(total)` y que no hay overlap
  (assert explícito, que falle ruidosamente si algo está mal).
- Idempotente: si `data/splits/real_split.json` ya existe, no lo
  regeneres (para que reproducir el experimento completo dé siempre el
  mismo split); solo bórralo manualmente si se quiere un split nuevo.

### 2. Modifica `training/02_train_generator_and_make_fakes.py`

- Agrega un argumento opcional `--split-file` (default: `None`).
- Si se pasa `--split-file data/splits/real_split.json`, la clase
  `RealRadiographs` debe filtrar y usar SOLO los archivos listados en
  `gen_pool` de ese JSON (no todos los de `data/real/`).
- Si no se pasa `--split-file`, el comportamiento debe ser exactamente
  igual al actual (usa todos los archivos de `data/real/`) — esto es
  para no romper la reproducibilidad de lo que ya se corrió antes.
- Imprime cuántas imágenes reales está usando para entrenar el generador,
  para que quede claro en el log cuál modo se usó.

### 3. Corrección del confound — módulo compartido de preprocesamiento

Crea una función reutilizable (puede vivir en un nuevo archivo
`training/preprocessing_common.py`, importado tanto por
`03_train_and_evaluate_detector.py` como por el script de validación del
punto 5):

```python
def random_resize_round_trip(img: PIL.Image.Image, rng: np.random.Generator,
                              size_range=(16, 128), final_size=128) -> PIL.Image.Image:
    """Aplica un round-trip de resize aleatorio ANTES del resize final a
    128x128, con el mismo rango de tamanos intermedios para reales y
    falsas, para que ninguna clase tenga una firma de 'historial de
    resize' mas limpia que la otra (ver EXP-00)."""
    intermediate = int(rng.integers(size_range[0], size_range[1] + 1))
    img = img.resize((intermediate, intermediate), Image.BICUBIC)
    return img.resize((final_size, final_size), Image.BICUBIC)
```

Reglas de aplicación (esto es lo más importante del spec, no te lo
saltes):

- Se aplica a **ambas clases por igual** (reales Y falsas), nunca solo a
  una.
- Durante **entrenamiento** (train split): `intermediate` se sortea de
  nuevo en cada época por cada imagen (es una augmentación estocástica
  normal, usa el `rng` global de PyTorch/numpy de la época, no necesita
  ser determinista).
- Durante **validación y test**: `intermediate` debe ser determinista y
  reproducible — usa `np.random.default_rng(seed=42 + index_de_la_imagen)`
  (o cualquier esquema que fije semilla por imagen) para que correr el
  script dos veces dé exactamente el mismo resultado en val/test.
- Ajusta `RadiographDataset.__getitem__` en
  `03_train_and_evaluate_detector.py` para llamar a esta función antes de
  calcular `radial_power_spectrum` y antes de construir el tensor para la
  ResNet, tanto para imágenes reales como falsas.

### 4. Corrige `load_dataset_splits()` en `03_train_and_evaluate_detector.py`

- El **test set de reales** debe salir EXCLUSIVAMENTE de
  `data/splits/real_split.json` → `holdout` (nunca de `gen_pool`).
- El **train/val de reales** sale de `gen_pool`.
- Las falsas se siguen muestreando de `data/fake/` con el mismo límite de
  3x el número de reales que ya existe (`max_fake_ratio`), pero ahora
  divide las falsas en train/val/test de forma proporcional a como
  quedaron divididas las reales (si reales quedó 60% train / 20% val /
  20% test aprox, las falsas deben seguir una proporción similar, para no
  desbalancear el split final).
- Agrega un `--split-file` (mismo default `data/splits/real_split.json`)
  como argumento del script principal.

### 5. Script de validación: `training/13_shortcut_audit.py`

Reutiliza la lógica de `00_resolution_confound_test.py` pero:
- Corre el mismo experimento de "grupo A vs grupo B" (reales divididas
  en dos, una mitad con resize directo y otra con el patrón de semilla
  8x8→128) DOS VECES:
  1. Sin aplicar `random_resize_round_trip` (esto debería reproducir el
     ~100% de AUC ya visto en EXP-00, como control negativo).
  2. Aplicando `random_resize_round_trip` a AMBOS grupos antes de todo lo
     demás (esto debería bajar el AUC hacia ~0.5 si la corrección
     funciona).
- Guarda ambos resultados en
  `training/exp13_shortcut_audit/exp13_result.json` con esta forma:
  ```json
  {
    "auc_without_fix": <float>,
    "auc_with_fix": <float>,
    "interpretation": "..."
  }
  ```
- Esto es la evidencia cuantitativa de que la corrección realmente
  neutraliza el shortcut, no solo una promesa.

### 6. Vuelve a correr el pipeline completo con la corrección

En este orden:
```bash
cp training/metrics.json training/metrics_old_protocol.json
python3 training/01b_split_holdout.py
python3 training/02_train_generator_and_make_fakes.py --epochs 800 --split-file data/splits/real_split.json
python3 training/03_train_and_evaluate_detector.py --epochs 25 --split-file data/splits/real_split.json
python3 training/13_shortcut_audit.py
```

Nota: el paso de `02_...` reentrena el GAN desde cero SOLO sobre
`gen_pool` (~122 imágenes en vez de 152) — esto sobreescribe
`training/checkpoints/generator.pth` y todo `data/fake/`. Eso es
intencional y correcto (las falsas viejas se generaron con el GAN viejo
que sí vio el holdout, hay que regenerarlas). Documenta en el log cuántas
imágenes falsas nuevas se generaron.

### 7. `training/exp15_protocol_comparison.json` (script o generado a mano al final)

Un archivo simple comparando lado a lado:
```json
{
  "old_protocol": {"...métricas de metrics_old_protocol.json..."},
  "new_protocol": {"...métricas nuevas de metrics.json..."},
  "shortcut_audit": {"...contenido de exp13_result.json..."}
}
```

## Qué NO reportar como éxito

Si el nuevo AUC/accuracy sigue siendo muy alto (ej. >95%) incluso después
de la corrección, **eso no es un bug tuyo a ocultar ni a "arreglar"
ajustando hiperparámetros hasta que baje** — repórtalo tal cual sale.
Puede ser un resultado real (el DCGAN es genuinamente fácil de detectar
por otras razones legítimas). Lo único que no es aceptable es que el
número salga alto porque la corrección no se aplicó bien — para
descartar eso, el resultado de `13_shortcut_audit.py` (`auc_with_fix`)
es el que prueba si la corrección en sí funciona, independiente de si el
resultado final del detector es alto o bajo.

## Criterios de aceptación

1. `data/splits/real_split.json` existe, con ~80/20 sin overlap, suma
   152.
2. `02_train_generator_and_make_fakes.py --split-file ...` imprime que
   usó ~122 imágenes (no 152).
3. `training/exp13_shortcut_audit/exp13_result.json` existe y muestra
   `auc_without_fix` cercano a 1.0 (replica EXP-00) y `auc_with_fix`
   medible y reportado (sin forzar que sea "bueno").
4. `training/metrics_old_protocol.json` (copia del resultado viejo) y
   `training/metrics.json` (resultado nuevo, sobreescrito por el paso 6)
   ambos existen y son diferentes.
5. `training/exp15_protocol_comparison.json` existe y compila ambos.
6. Todos los scripts corren sin excepciones de principio a fin en este
   Mac (Apple M4 Pro, MPS, sin CUDA) — pruébalo tú mismo antes de
   terminar, no asumas que compila.
7. No se tocó `paper/main.tex`, no se hizo commit, no se hizo push.
