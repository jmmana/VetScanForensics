"""EXP-17: FID/KID sobre gen_pool; sin métricas pareadas ficticias."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import shlex
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vetscan-matplotlib"))

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from experiment_common import ROOT, HERE, cli, load_real_split, resolve_generators, write_json


class Images(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Mismo tamaño objetivo y bilinear PIL que el entrenamiento del DCGAN.
        # Sin augmentación/round-trip del detector: se evalúa la salida generativa.
        with Image.open(self.paths[index]) as image:
            gray = np.array(image.convert("L").resize((128, 128), Image.Resampling.BILINEAR))
        return torch.from_numpy(gray).unsqueeze(0).repeat(3, 1, 1)


class PrecomputedInception(torch.nn.Identity):
    """Adaptador público feature=Module; no implementa ninguna métrica."""
    num_features = 2048


@torch.inference_mode()
def extract(paths, model, batch_size, device, label):
    chunks = []
    for batch in DataLoader(Images(paths), batch_size=batch_size, shuffle=False, num_workers=0):
        chunks.append(model(batch.to(device))[0].cpu().double())
        print(f"{label}: Inception {sum(len(x) for x in chunks)}/{len(paths)}", flush=True)
    features = torch.cat(chunks)
    if features.shape != (len(paths), 2048) or not torch.isfinite(features).all():
        raise ValueError(f"Features Inception inválidas: {label}")
    return features


def calculate_fid(real, fake):
    from torchmetrics.image.fid import FrechetInceptionDistance
    metric = FrechetInceptionDistance(feature=PrecomputedInception()).set_dtype(torch.float64)
    metric.update(real, real=True)
    metric.update(fake, real=False)
    return float(metric.compute())


def calculate_metrics(real, fake, args, subset_size, name):
    from torchmetrics.image.kid import KernelInceptionDistance
    fid = calculate_fid(real, fake)
    kid = KernelInceptionDistance(feature=PrecomputedInception(), subsets=args.kid_subsets,
                                 subset_size=subset_size, degree=3, gamma=1 / 2048,
                                 coef=1.0).set_dtype(torch.float64)
    kid.update(real, real=True)
    kid.update(fake, real=False)
    # Reiniciar por generador: resultados independientes del orden/selección.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        kid_mean, kid_std = map(float, kid.compute())
    rng = np.random.default_rng(args.seed)
    bootstrap = []
    for index in range(args.fid_bootstraps):
        # Remuestreo independiente con reemplazo; tamaños originales conservados.
        r = real[rng.integers(0, len(real), size=len(real))]
        f = fake[rng.integers(0, len(fake), size=len(fake))]
        bootstrap.append(calculate_fid(r, f))
        print(f"{name}: bootstrap FID {index + 1}/{args.fid_bootstraps}", flush=True)
    return {"fid": fid, "kid_mean": kid_mean, "kid_std": kid_std,
            "fid_bootstrap": {"replicates": args.fid_bootstraps, "seed": args.seed,
                              "method": "independent percentile bootstrap, replacement, original sample sizes",
                              "confidence_level": 0.95,
                              "percentile_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                              "values": bootstrap,
                              "interpretation": "Exploratorio; no corrige sesgo FID ni garantiza cobertura con n pequeño."}}


def manifest(paths):
    return [{"path": str(path.relative_to(ROOT)),
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]


def write_notes(result, output):
    rows = "\n".join(
        f"| {r['generator']} | {r['n_real']} | {r['n_fake']} | {r['fid']:.6f} | "
        f"{r['kid_mean']:.8f} | {r['kid_std']:.8f} | "
        f"[{r['fid_bootstrap']['percentile_interval'][0]:.6f}, "
        f"{r['fid_bootstrap']['percentile_interval'][1]:.6f}] |" for r in result["results"])
    p = result["protocol"]
    outside = [r["generator"] for r in result["results"]
               if not r["fid_bootstrap"]["percentile_interval"][0] <= r["fid"]
               <= r["fid_bootstrap"]["percentile_interval"][1]]
    bootstrap_observation = (
        f"En esta corrida, el intervalo bootstrap no contiene el FID puntual para: {', '.join(outside)}. "
        "No interpretarlo como intervalo calibrado ni usarlo para probar diferencias entre generadores."
        if outside else "Los intervalos bootstrap son exploratorios, no una prueba de diferencias entre generadores.")
    versions = ", ".join(f"{key} {value}" for key, value in result["versions"].items())
    text = f"""# EXP-17 — calidad de imagen

Ejecución UTC: {result['created_utc']}. Finalizada UTC: {result['completed_utc']}.
Bibliotecas: {versions}. Python: {result['python']}. Extracción: {p['device']};
cálculo FID/KID en CPU float64, features extraídas en float32. Hilos: {p['threads']}.

## Resultados de esta ejecución

| Generador | Reales | Falsas | FID | KID media | KID std | FID bootstrap percentiles 2.5–97.5% |
|---|---:|---:|---:|---:|---:|---|
{rows}

Generadores ausentes (sin métricas): {', '.join(result['unavailable_generators']) or 'ninguno'}.
KID se muestra en unidades crudas, sin multiplicar por 100/1000.

{bootstrap_observation}

## Metodología exacta

- `experiment_common.resolve_generators()` lee el JSON de `training/GENERATORS.md`.
  Se incluyen todas las imágenes PNG/JPG/JPEG del directorio resuelto, sin
  limitar ni equilibrar el número de falsas. DCGAN acepta el alias `data/fake`.
- Referencia exclusiva: `gen_pool` de `{p['split_file']}`, semilla del split
  {p['split_seed']}. `load_real_split` valida exhaustividad, duplicados y ausencia
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
  kernel polinómico `(x·y/2048 + 1)^3`, estimador MMD² sin sesgo, {p['kid_subsets']}
  submuestras de {p['kid_subset_size']} reales y {p['kid_subset_size']} falsas.
  Tamaño común a todos los generadores: mínimo del solicitado ({p['kid_subset_size_requested']}),
  n_real y el menor n_fake. Muestreo sin reemplazo dentro de cada submuestra;
  se permiten repeticiones entre submuestras. Semilla {p['seed']} reiniciada por
  generador. `kid_std` es la desviación poblacional entre estas submuestras
  (implementación de TorchMetrics), **no un intervalo de confianza ni un error
  estándar**; las submuestras se solapan. El estimador KID puede ser negativo.
- FID bootstrap: {p['fid_bootstraps']} réplicas, semilla {p['seed']}, remuestreo
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
.venv/bin/python -m pip install -q --break-system-packages torchmetrics=={result['versions']['torchmetrics']} torch-fidelity=={result['versions']['torch-fidelity']}
```

Comando ejecutado:

```bash
{result['command']}
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
- TorchMetrics (FID/KID): https://github.com/Lightning-AI/torchmetrics/tree/v{result['versions']['torchmetrics']}/src/torchmetrics/image
- Extractor: https://github.com/toshas/torch-fidelity/blob/v{result['versions']['torch-fidelity']}/torch_fidelity/feature_extractor_inceptionv3.py
"""
    (output / "notas.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generators", nargs="+")
    parser.add_argument("--split-file", type=Path, default=ROOT / "data/splits/real_split.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "exp17_image_quality")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--kid-subset-size", type=int, default=100)
    parser.add_argument("--fid-bootstraps", type=int, default=100)
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1 or min(args.kid_subsets, args.kid_subset_size, args.fid_bootstraps) < 2:
        raise ValueError("batch-size/threads >= 1; kid-subsets/kid-subset-size/fid-bootstraps >= 2")
    if args.seed < 0:
        raise ValueError("seed debe ser no negativa")
    generators, missing = resolve_generators(args.generators)
    for name in missing:
        print(f"AUSENTE: {name}. Corre primero el generador {name}; no se calculan métricas.", flush=True)
    split = load_real_split(args.split_file, ROOT / "data/real")
    real_paths = [ROOT / "data/real" / name for name in sorted(split["gen_pool"])]
    subset_size = min(args.kid_subset_size, len(real_paths), *(len(x) for x in generators.values()))
    if subset_size < 2:
        raise ValueError("FID/KID requieren al menos dos imágenes por conjunto")
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3, URL_INCEPTION_V3
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.hub.set_dir(str(HERE / "checkpoints/exp17"))
    extractor = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).eval().to(args.device)
    weights_path = Path(torch.hub.get_dir()) / "checkpoints" / URL_INCEPTION_V3.rsplit("/", 1)[-1]
    result = {
        "experiment": "EXP-17", "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([".venv/bin/python", *sys.argv]), "python": sys.version,
        "versions": {p: version(p) for p in ("torchmetrics", "torch-fidelity", "torch", "torchvision", "numpy", "Pillow")},
        "unavailable_generators": missing,
        "paired_metrics": None,
        "paired_metrics_reason": "Sin pares correspondientes real–sintética; SSIM/PSNR/LPIPS estándar no aplicables.",
        "protocol": {"reference": "real_gen_pool", "holdout_used": False,
                     "split_file": str(args.split_file), "split_seed": split["seed"],
                     "split_sha256": hashlib.sha256(args.split_file.read_bytes()).hexdigest(),
                     "device": args.device, "threads": args.threads, "batch_size": args.batch_size,
                     "seed": args.seed, "kid_subsets": args.kid_subsets, "kid_subset_size": subset_size,
                     "kid_subset_size_requested": args.kid_subset_size,
                     "kid_kernel": {"degree": 3, "gamma": 1 / 2048, "coef": 1.0},
                     "kid_std_definition": "Population standard deviation across overlapping subsets; not CI or SE",
                     "fid_bootstraps": args.fid_bootstraps, "feature_dimension": 2048,
                     "preprocessing": "PIL L uint8 -> bilinear 128x128 -> repeat gray channel 3 times -> torch-fidelity TF-compatible bilinear 299x299 -> (x-128)/128",
                     "weights_url": URL_INCEPTION_V3,
                     "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest()},
        "real_manifest": manifest(real_paths), "results": []}
    real = extract(real_paths, extractor, args.batch_size, args.device, "real_gen_pool")
    for name, paths in generators.items():
        notes = [result["paired_metrics_reason"],
                 "Inception ImageNet fuera de dominio; gris repetido en RGB. Comparación relativa, no benchmark natural.",
                 "FID/KID no prueban indistinguibilidad, detectabilidad ni validez clínica; ver EXP-05/06.",
                 "Referencia de entrenamiento: no mide generalización ni descarta memorización.",
                 "KID std entre submuestras no es IC; bootstrap FID exploratorio no corrige sesgo."]
        if min(len(real_paths), len(paths)) < 2000:
            notes.append("Muestra <~2000 en al menos un conjunto: FID sesgado/inestable; KID de alta varianza.")
        print(f"{name}: n_real={len(real_paths)}, n_fake={len(paths)}. {' '.join(notes)}", flush=True)
        fake = extract(paths, extractor, args.batch_size, args.device, name)
        metrics = calculate_metrics(real, fake, args, subset_size, name)
        row = {"generator": name, "n_real": len(real_paths), "n_fake": len(paths), **metrics,
               "paired_metrics": None, "ssim": None, "psnr": None, "lpips": None,
               "notes": notes, "fake_manifest": manifest(paths)}
        result["results"].append(row)
        print(f"{name}: FID={row['fid']:.6f}; KID={row['kid_mean']:.8f} ± {row['kid_std']:.8f}", flush=True)
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output_dir / "results.json", result)
    write_notes(result, args.output_dir)
    print(f"Resultados y notas guardados en {args.output_dir}", flush=True)


if __name__ == "__main__":
    cli(main)
