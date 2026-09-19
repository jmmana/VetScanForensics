# EXP-04 — Diffusion model, entrenamiento en Kaggle, 2026-09-18

Kernel `maktub83/vetscan-exp-04-diffusion` (GPU, dataset de entrada
`vet-radiographs-genpool-122`, 122 imagenes de `real_gen_pool` unicamente,
nunca `real_holdout`). Script: `kaggle_kernels/exp04_diffusion/train_diffusion.py`.
Estado final: `KernelWorkerStatus.COMPLETE`, sin errores en el log.

## Arquitectura y protocolo

DDPM (Ho et al. 2020, arXiv:2006.11239) entrenado **desde cero**, no
fine-tuning de un prior generico de Stable Diffusion (no conoce anatomia
radiografica). `UNet2DModel` de `diffusers` (HuggingFace), 1 canal
(escala de grises), resolucion 128x128, `block_out_channels`
[64,128,128,256], atencion en el segundo down-block / segundo up-block.
200 epocas sobre las 122 imagenes de gen_pool. `DDPMScheduler` estandar,
`get_cosine_schedule_with_warmup`.

Curva de perdida real (extraida del log del kernel):

| Epoca | Loss |
|---|---|
| 0 | 1.0272 |
| 20 | 0.0439 |
| 40 | 0.0457 |
| 60 | 0.0236 |
| 80 | 0.0321 |
| 100 | 0.0246 |
| 120 | 0.0217 |
| 140 | 0.0366 |
| 160 | 0.0241 |
| 180 | 0.0261 |
| 199 | 0.0156 |

La perdida cae de ~1.0 a un rango estable de 0.02-0.04 tras la epoca 20 y
no muestra divergencia ni colapso hasta el final del entrenamiento.

## Salida

300 imagenes generadas (`diffusion_0000.png`...`diffusion_0299.png`),
128x128, escala de grises, copiadas a `data/fake_diffusion/` (300
archivos, ~3.5 MB) siguiendo la convencion de `training/GENERATORS.md`.
Inspeccion visual de una muestra (epocas de generacion 0, 50, 150, 299):
las cuatro imagenes muestran siluetas toracicas plausibles con textura
interna, no ruido uniforme ni un modo colapsado repetido. No se hizo
FID/KID todavia (EXP-17, pendiente).

Checkpoint (`diffusion_pytorch_model.safetensors`, config.json) guardado
localmente en `training/checkpoints/diffusion_model/` junto con el log
crudo del kernel (`training_log.jsonl`). **No se commitea a git** (mismo
criterio que `detector_best.pth`/`generator.pth`: `*.safetensors` y
`training/checkpoints/` estan en `.gitignore`); tampoco se commitean las
300 imagenes generadas (`data/fake_diffusion/*.png` gitignoreado, mismo
criterio que `data/fake/*.png` para DCGAN — son reproducibles desde el
checkpoint y el script, no se versionan como datos binarios).

## Pendiente

- EXP-03 (StyleGAN2-ADA) sigue `RUNNING` en Kaggle al momento de escribir
  esto.
- Con ambos generadores nuevos disponibles: re-correr EXP-06 (matriz
  3x3), EXP-07/08 (25 epocas, los 3 generadores), EXP-09/12 (robustez y
  explicabilidad multi-generador), EXP-17 (FID/KID/SSIM/LPIPS vs. real)
  por generador.
