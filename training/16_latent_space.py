"""EXP-16: visualización de la fusión de 64 dimensiones del clasificador EXP-19."""
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import experiment_common as common
from posthoc_common import digest, signature

_spec = importlib.util.spec_from_file_location('attribution19', common.HERE / '19_generator_attribution.py')
attribution = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(attribution)
COLORS = {'real': '#2a78d6', 'dcgan': '#eb6834', 'diffusion': '#1baf7a', 'stylegan2ada': '#8064a2'}


def all_points(generators, args, splits):
    """Sin límite de falsas; preservar origen real y pertenencia al entrenamiento."""
    real = common.load_real_split(args.split_file, common.base.REAL_DIR)
    membership = {p.resolve(): name for name, (paths, _) in
                  zip(('train', 'val', 'test'), splits) for p in paths}
    rows = []
    groups = [('real', origin, sorted(common.base.REAL_DIR / p for p in real[origin]))
              for origin in ('gen_pool', 'holdout')]
    groups += [(name, 'synthetic', sorted(paths)) for name, paths in generators.items()]
    classes = ['real', *generators]
    for name, origin, paths in groups:
        for path in paths:
            rows.append(dict(path=str(path.relative_to(common.ROOT)), label=classes.index(name),
                             generator=name, split=origin,
                             attribution_split=membership.get(path.resolve(), 'unused'),
                             preprocessing_seed=42 + len(rows), sha256=digest(path)))
    if len({(common.ROOT / r['path']).resolve() for r in rows}) != len(rows):
        raise ValueError('Imágenes duplicadas en el manifiesto EXP-16')
    return rows


def expected_identity(generators, args, splits):
    identity = signature(splits, args, generators)
    identity.update(architecture='resnet18+fft32-multiclass', classes=['real', *generators])
    return identity


def load_model(generators, args, splits):
    checkpoint = common.HERE / 'checkpoints/attribution_multiclass.pth'
    identity = expected_identity(generators, args, splits)
    device = common.base.get_device() if args.device == 'auto' else torch.device(args.device)
    model = attribution.AttributionModel(len(identity['classes'])).to(device)

    def attempt():
        if not checkpoint.exists():
            return 'checkpoint ausente'
        try:
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if saved.get('identity') != identity or saved.get('classes') != identity['classes']:
                return 'identidad incompatible (clases, arquitectura, split, manifiesto o hashes)'
            model.load_state_dict(saved['state_dict'], strict=True)
        except (RuntimeError, ValueError, KeyError, OSError) as exc:
            return f'pesos incompatibles: {exc}'
        return None

    reason = attempt()
    command = None
    if reason:
        print(f'EXP-19: {reason}; ejecutando su protocolo original completo.', flush=True)
        command = [sys.executable, str(common.HERE / '19_generator_attribution.py'),
                   '--generators', *generators, '--split-file', str(args.split_file.resolve()),
                   '--epochs', str(args.epochs), '--batch-size', str(args.batch_size),
                   '--lr', str(args.lr), '--device', args.device, '--threads', str(args.threads),
                   '--output-dir', str(args.output_dir.resolve() / 'exp19_retraining')]
        if subprocess.run(command, cwd=common.ROOT).returncode:
            raise ValueError('Falló el protocolo original EXP-19; no se generan proyecciones')
        failure = attempt()
        if failure:
            raise ValueError(f'Checkpoint tras entrenamiento inválido: {failure}')
    model.eval()
    print(f'Checkpoint EXP-19 {"reutilizado sin entrenamiento" if reason is None else "reentrenado"}', flush=True)
    return model, dict(path=str(checkpoint.relative_to(common.ROOT)), sha256=digest(checkpoint),
                       reused=reason is None, fallback_reason=reason, training_command=command,
                       identity=identity, device=str(device))


def extract_embeddings(model, rows, args):
    """Captura la entrada REAL a Linear(64,C), tras ReLU y dropout desactivado."""
    blocks = []
    handle = model.head[-1].register_forward_pre_hook(
        lambda module, inputs: blocks.append(inputs[0].detach().cpu().numpy().copy()))
    group = ([common.ROOT / r['path'] for r in rows], [r['label'] for r in rows])
    model.eval()
    device = next(model.parameters()).device
    try:
        with torch.inference_mode():
            for image, frequency, _ in attribution.loader(group, args):
                model(image.to(device), frequency.to(device))
    finally:
        handle.remove()
    embeddings = np.concatenate(blocks)
    if embeddings.shape != (len(rows), 64) or not np.isfinite(embeddings).all():
        raise ValueError('Embedding inválido: se esperaba N×64 finito')
    return embeddings


def project(embeddings, seed):
    from umap import UMAP
    from sklearn.manifold import TSNE
    if len(embeddings) < 4:
        raise ValueError('Se requieren al menos 4 imágenes para las proyecciones')
    umap_params = dict(n_components=2, n_neighbors=min(15, len(embeddings)-1), min_dist=0.1,
                       metric='euclidean', init='spectral', random_state=seed,
                       transform_seed=seed, n_jobs=1, n_epochs=500)
    tsne_params = dict(n_components=2, perplexity=min(30., len(embeddings)-1.),
                       metric='euclidean', init='pca', learning_rate='auto', max_iter=1000,
                       early_exaggeration=12., method='barnes_hut', angle=0.5,
                       random_state=seed, n_jobs=1)
    projections, settings = {}, {}
    for name, reducer in [('umap', UMAP(**umap_params)), ('tsne', TSNE(**tsne_params))]:
        print(f'Proyectando {name}: {len(embeddings)} puntos', flush=True)
        projections[name] = reducer.fit_transform(embeddings)
        if not np.isfinite(projections[name]).all():
            raise ValueError(f'Proyección {name} no finita')
        settings[name] = reducer.get_params()
    return projections, settings


def plot_projection(coordinates, labels, classes, method, output, seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
    # Orden aleatorio fijo para no ocultar sistemáticamente una clase con otra.
    order = np.random.default_rng(seed).permutation(len(labels))
    ax.scatter(*coordinates[order].T, c=[COLORS[classes[y]] for y in labels[order]],
               s=23, alpha=.7, edgecolors='none')
    for i, name in enumerate(classes):
        ax.scatter([], [], color=COLORS[name], s=40, label=f'{name} (n={int((labels == i).sum())})')
    ax.legend(frameon=True, loc='best')
    title = 'UMAP' if method == 'umap' else 't-SNE'
    ax.set(title=f'EXP-16 · {title} · fusión EXP-19 (64D) · n={len(labels)}',
           xlabel=f'{title} 1 (unidades arbitrarias)', ylabel=f'{title} 2 (unidades arbitrarias)')
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(output, dpi=180, facecolor='white')
    plt.close(fig)


def write_notes(metadata, args):
    counts = {c: sum(r['generator'] == c for r in metadata['manifest']) for c in metadata['classes']}
    splits = {s: sum(r['split'] == s for r in metadata['manifest'])
              for s in ('gen_pool', 'holdout', 'synthetic')}
    notes = f'''# EXP-16: espacio latente

## Ejecución y datos

Conteos completos, sin submuestreo: {counts}. Origen: {splits}.
Ausentes: {metadata['unavailable_generators']}. No se fabricaron puntos.
Checkpoint: `{metadata['checkpoint']['path']}`; SHA256 `{metadata['checkpoint']['sha256']}`.
Reutilizado sin entrenar: {metadata['checkpoint']['reused']}.
Motivo de fallback: {metadata['checkpoint']['fallback_reason']}.
Comando de entrenamiento original, si hizo falta: {metadata['checkpoint']['training_command']}.
Compatibilidad comprobada contra identidad EXP-19: arquitectura, clases en orden,
split seed 42/hash, manifiesto train/val/test y hashes de imágenes; carga estricta de pesos.
Dispositivo de extracción: {metadata['checkpoint']['device']}.

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
Semilla de ambas proyecciones: {args.seed}; parámetros completos:

```json
{json.dumps(metadata['projection_parameters'], indent=2)}
```

Versiones exactas: {metadata['versions']}. Python: {metadata['python']}.
UMAP usa un hilo; t-SNE n_jobs=1; PyTorch threads={args.threads}.
Semillas fijas permiten repetir bajo el mismo entorno; distintas versiones o dispositivos
pueden producir diferencias numéricas. No se ajustaron parámetros buscando separación.
Paleta categórica del proyecto: real azul #2a78d6, dcgan naranja #eb6834,
diffusion aqua #1baf7a (referencias make_exp09_robustness_figure.py y specs/04).
El skill dataviz no está instalado; se reutiliza su paleta documentada en el proyecto.
StyleGAN2-ADA futuro usa violeta #8064a2; no aparece si está ausente.

## Observaciones visuales

Pendientes de inspección de las figuras de esta ejecución. No se generan conclusiones
automáticas a partir de las etiquetas. Al re-correr se debe revisar esta sección.

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
.venv/bin/python training/16_latent_space.py --generators {' '.join(metadata['classes'][1:])} --device {args.device} --threads {args.threads} --seed {args.seed}
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
'''
    (args.output_dir / 'notas.md').write_text(notes)


def main():
    parser = common.parser(__doc__)
    parser.add_argument('--output-dir', type=Path, default=common.HERE / 'exp16_latent_space')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embeddings-input', type=Path, help='Reproyectar un NPZ sin reextraer features')
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32 or args.threads < 1:
        raise ValueError('seed debe estar en [0, 2**32) y threads ser positivo')
    torch.set_num_threads(args.threads)
    if args.embeddings_input:
        if args.generators:
            common.resolve_generators(args.generators)
        with np.load(args.embeddings_input, allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files}
        metadata = json.loads(str(arrays.pop('metadata_json')))
        if args.generators and args.generators != metadata['classes'][1:]:
            raise ValueError('--generators no coincide con las clases del NPZ')
        embeddings = arrays['embeddings']
        if embeddings.shape != (len(arrays['labels']), 64) or not np.isfinite(embeddings).all():
            raise ValueError('El archivo requiere embeddings N×64 finitos')
        metadata['embeddings_input'] = str(args.embeddings_input.resolve())
    else:
        generators, missing = common.prepare(args)
        print(f'Generadores: {list(generators)}; ausentes: {missing}', flush=True)
        splits = attribution.attribution_splits(generators, args)
        model, checkpoint = load_model(generators, args, splits)
        rows = all_points(generators, args, splits)
        embeddings = extract_embeddings(model, rows, args)
        metadata = dict(classes=['real', *generators], manifest=rows, checkpoint=checkpoint,
                        unavailable_generators=missing)
        arrays = dict(embeddings=embeddings, classes=np.array(metadata['classes']),
                      labels=np.array([r['label'] for r in rows], dtype=np.int64),
                      **{key: np.array([r[source] for r in rows]) for key, source in
                         [('generator', 'generator'), ('split', 'split'),
                          ('attribution_split', 'attribution_split'), ('paths', 'path'),
                          ('preprocessing_seed', 'preprocessing_seed')]})
        print(f'Embeddings extraídos: {embeddings.shape}', flush=True)
    projections, parameters = project(embeddings, args.seed)
    metadata.update(projection_parameters=parameters, seed=args.seed,
                    versions={p: importlib.metadata.version(p) for p in
                              ('umap-learn', 'scikit-learn', 'numpy', 'numba', 'scipy',
                               'torch', 'torchvision', 'matplotlib', 'pillow')},
                    python=platform.python_version())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / 'embeddings.npz', **arrays,
                        metadata_json=np.array(json.dumps(metadata)))
    np.savez_compressed(args.output_dir / 'projections.npz', **projections)
    for method, coordinates in projections.items():
        plot_projection(coordinates, arrays['labels'], metadata['classes'], method,
                        args.output_dir / f'{method}_projection.png', args.seed)
    common.write_json(args.output_dir / 'results.json', metadata)
    write_notes(metadata, args)
    print(f'Resultados completos en {args.output_dir}', flush=True)


if __name__ == '__main__':
    common.cli(main)
