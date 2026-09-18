# Registro de generadores (EXP-06/07/08)

Los tres scripts leen **este bloque JSON** mediante `experiment_common.py`.
Rutas relativas a la raíz del repositorio. Depositar PNG/JPG/JPEG directamente
(sin subdirectorios). No mover `data/fake/`: se acepta como alias histórico de
DCGAN. Si existen ambas rutas con imágenes se usa exclusivamente la canónica.

```json
{
  "dcgan": {"directory": "data/fake_dcgan", "aliases": ["data/fake"]},
  "stylegan2ada": {"directory": "data/fake_stylegan2ada", "aliases": []},
  "diffusion": {"directory": "data/fake_diffusion", "aliases": []}
}
```

Agregar un generador = agregar una entrada al bloque y llenar su carpeta.
Sin `--generators` se usan todas las entradas que tienen imágenes en disco;
las ausentes se registran en el resultado. Para exigir un conjunto usar
`--generators dcgan stylegan2ada diffusion`: cualquier ausencia/vacío aborta
con «Corre primero el generador X». No se fabrican resultados ausentes.
No cambiar nombres ni contenido durante una ejecución. Exportar imágenes de
cada generador entrenado **solo con gen_pool**, nunca con real_holdout.

EXP-07/08 mezclan las falsas seleccionadas y aplican el split original (semilla
42, máximo tres falsas por real en total). EXP-06 divide cada generador por
separado; entrena una vez por fila y evalúa siempre la partición test del
cada generador, incluso fuera de la diagonal. Ninguna falsa de train/val del
mismo generador entra en test. Se guardan las listas exactas en los resultados.
Con un generador, EXP-06 produce una matriz 1x1, sin medir transferencia todavía.
