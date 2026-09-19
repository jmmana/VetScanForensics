# Spec: EXP-17 (calidad de imagen: FID/KID/SSIM/PSNR/LPIPS)

## Contexto

Mismo proyecto que specs/01, specs/02 y specs/03, leelos primero. Existe
`training/experiment_common.py` con `resolve_generators` y la convencion
de `training/GENERATORS.md` (bloque JSON `dcgan`/`stylegan2ada`/`diffusion`,
cada uno con su `directory`). Hoy hay imagenes reales en `data/real/`
(pool completo) y `data/splits/real_split.json` (gen_pool/holdout, semilla
42). Dos generadores tienen salida en disco: DCGAN (`data/fake_dcgan`,
alias historico `data/fake`) y diffusion (`data/fake_diffusion`, 300
imagenes 128x128 en escala de grises, generadas hoy). StyleGAN2-ADA
(`data/fake_stylegan2ada`) todavia no existe.

## Objetivo

Cuantificar que tan realistas son las imagenes sinteticas comparadas con
las reales, generador por generador, usando metricas estandar de
generative modeling. Esto es EXP-17 del plan maestro
(`~/.claude/plans/ahora-para-el-paper-cryptic-cookie.md`, buscalo si
existe para mas contexto, si no existe usa este spec como fuente de
verdad).

Crea `training/17_image_quality.py` que:

1. Usa `experiment_common.resolve_generators()` para descubrir que
   generadores tienen imagenes en disco. Para cada generador disponible,
   compara sus imagenes contra el **`real_gen_pool`** (las mismas
   imagenes reales usadas para entrenar ese generador, nunca el
   holdout: seria una comparacion invalida, el holdout es para evaluar
   el detector, no la calidad generativa).
2. Calcula, para cada generador disponible:
   - **FID** (Frechet Inception Distance, Heusel et al. 2017,
     arXiv:1706.08500). Usa `pytorch-fid` o `torchmetrics.image.fid.FrechetInceptionDistance`
     si ya esta disponible en el entorno (revisa `requirements.txt` /
     `.venv`), documenta cual libreria usaste y su version exacta.
     Convierte las imagenes en escala de grises a 3 canales (repetir el
     canal) antes de pasarlas a Inception, que espera RGB — documentalo,
     es una limitacion conocida (Inception fue entrenado en ImageNet
     natural, no en radiografias, FID en este dominio es una senal
     relativa entre generadores, no un valor absoluto comparable a
     benchmarks de imagenes naturales).
   - **KID** (Kernel Inception Distance, Binkowski et al. 2018,
     arXiv:1801.01401), misma libreria/pipeline que FID si es posible.
   - **SSIM** (Wang et al. 2004) y **PSNR**: estos requieren pares
     imagen-a-imagen, no hay pares reales-falsas correspondientes aqui
     (son datasets no pareados). Documenta esta limitacion explicitamente
     en las notas y en el JSON de salida (`"paired_metrics": null` o
     similar) en vez de inventar un pareo arbitrario. Si decides calcular
     una variante no pareada razonable (ej. SSIM del real mas cercano en
     espacio de pixeles, o promedio contra un real aleatorio con semilla
     fija), documentalo como tal, con su limitacion, no lo presentes como
     SSIM pareado estandar.
   - **LPIPS** (Zhang et al. 2018, arXiv:1801.03924): misma limitacion de
     no-pareo que SSIM/PSNR, mismo tratamiento honesto.
3. Con el numero de imagenes disponible (DCGAN: 300 falsas vs. 122
   reales de gen_pool; diffusion: 300 falsas vs. 122 reales de gen_pool),
   verifica que el tamano de muestra sea razonable para FID/KID (FID es
   conocido por ser sesgado con pocas muestras, menos de ~2000 suele
   dar valores inestables). Documenta esto como limitacion explicita en
   vez de ocultarlo, y si la libreria que uses reporta un intervalo o
   permite bootstrap, usalo.
4. Guarda:
   - `training/exp17_image_quality/results.json`: por generador,
     `{generator, n_real, n_fake, fid, kid_mean, kid_std, notes}`.
   - `training/exp17_image_quality/notas.md`: metodologia exacta,
     limitaciones (no-pareo, tamano de muestra, dominio fuera de
     distribucion de Inception), y los comandos exactos para re-correr
     esto cuando StyleGAN2-ADA este disponible.
   - Si quieres agregar una figura comparativa (barras de FID/KID por
     generador), usa la paleta validada del skill de dataviz ya usada en
     el resto del proyecto (azul #2a78d6, naranja #eb6834, aqua #1baf7a
     si necesitas un tercer color).

## No-goals

- No toques `paper/main.tex`, no hagas commit ni push.
- No fabriques resultados para StyleGAN2-ADA: el script debe reportar
  explicitamente que generador esta ausente, igual que EXP-06/07/08/09/12.
- No instales ni reinventes una implementacion de FID desde cero si una
  libreria mantenida (`pytorch-fid`, `torchmetrics`, `clean-fid`) ya esta
  disponible o se puede instalar con pip; si necesitas instalar algo,
  usa `pip install -q --break-system-packages <paquete>` dentro del
  entorno que ya usan los scripts anteriores (revisa como EXP-08 manejo
  la dependencia de PyWavelets como referencia) o el `.venv` del repo si
  existe.
- No sobre-interpretes un FID bajo o alto como prueba de "el generador
  es indistinguible" o "el generador es facil de detectar" — esa
  pregunta la responde el detector (EXP-05/06), no esta metrica. Deja
  esa distincion explicita en las notas.

## Verificacion antes de terminar

1. Corre el script completo, confirma que produce resultados para DCGAN
   y diffusion, y que reporta explicitamente StyleGAN2-ADA como ausente
   sin fallar con traceback.
2. Si generas una figura, abrela y confirma que no este vacia.
3. Escribe `training/exp17_image_quality/notas.md` con lo que corriste,
   las limitaciones metodologicas explicitas mencionadas arriba, y los
   comandos exactos para re-correr con los 3 generadores disponibles.
