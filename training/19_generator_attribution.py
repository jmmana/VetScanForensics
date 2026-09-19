"""EXP-19: atribución multiclase con arquitectura y preprocesamiento de EXP-03."""
import importlib.util
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import experiment_common as common
from posthoc_common import signature, digest

_spec = importlib.util.spec_from_file_location('robustness09', common.HERE / '09_robustness.py')
robustness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(robustness)


class AttributionModel(common.base.DualStreamDetector):
    def __init__(self, num_classes):
        if num_classes < 2:
            raise ValueError('Se requiere real y al menos un generador')
        super().__init__()
        self.head[-1] = nn.Linear(self.head[-1].in_features, num_classes)
        # El forward heredado conserva [batch, clases], pues clases >= 2.


def attribution_splits(generators, args):
    if common.load_real_split(args.split_file, common.base.REAL_DIR)['seed'] != 42:
        raise ValueError('El protocolo requiere split real con seed=42')
    groups = [([], []) for _ in range(3)]
    for class_id, files in enumerate(generators.values(), start=1):
        # Exactamente las falsas de EXP-06; límite de 3x reales POR generador.
        for destination, (paths, labels) in zip(groups, common.split_for(files, args)):
            for path, label in zip(paths, labels):
                if label or class_id == 1:
                    destination[0].append(path)
                    destination[1].append(class_id if label else 0)
    path_sets = [{p.resolve() for p in paths} for paths, _ in groups]
    assert all(not path_sets[i] & path_sets[j] for i, j in ((0, 1), (0, 2), (1, 2)))
    return tuple(groups)


def loader(group, args, train=False, postprocess=None):
    return DataLoader(common.base.RadiographDataset(*group, train=train, postprocess=postprocess),
                      batch_size=args.batch_size, shuffle=train)


def run_epoch(model, batches, optimizer=None):
    model.train(optimizer is not None)
    device = next(model.parameters()).device
    total, probabilities, labels = 0., [], []
    with torch.set_grad_enabled(optimizer is not None):
        for image, frequency, target in batches:
            target = target.to(device=device, dtype=torch.long)
            logits = model(image.to(device), frequency.to(device))
            loss = nn.functional.cross_entropy(logits, target)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(target)
            probabilities.extend(logits.softmax(1).detach().cpu().tolist())
            labels.extend(target.cpu().tolist())
    return dict(loss=total / len(labels), probabilities=probabilities, labels=labels)


def metrics(details, classes):
    predicted = np.asarray(details['probabilities']).argmax(1)
    labels = details['labels']
    ids = list(range(len(classes)))
    matrix = confusion_matrix(labels, predicted, labels=ids)
    return dict(accuracy=float(accuracy_score(labels, predicted)),
                confusion_matrix=matrix.tolist(),
                f1_per_class=dict(zip(classes, f1_score(labels, predicted, labels=ids,
                                                       average=None, zero_division=0).tolist())),
                support=dict(zip(classes, matrix.sum(1).tolist())), n=len(labels))


def representative(row):
    return row['severity'] in {
        'clean': [0], 'jpeg': [10, 30], 'jpeg_multiple': [5],
        'resize': robustness.LEVELS['resize'], 'crop': [.6], 'blur': [3.],
        'noise': [.01, .1],
    }.get(row['transform'], [])


def plot_confusion(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    matrix = np.asarray(result['evaluation']['confusion_matrix'])
    fractions = matrix / matrix.sum(1, keepdims=True)
    palette = LinearSegmentedColormap.from_list('project_blue', ['#ffffff', '#2a78d6'])
    n = len(matrix)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.4), max(5, n * 1.2)), layout='constrained')
    heat = ax.imshow(fractions, cmap=palette, vmin=0, vmax=1)
    for (i, j), count in np.ndenumerate(matrix):
        ax.text(j, i, f'{count}\n{fractions[i, j]:.1%}', ha='center', va='center',
                color='white' if fractions[i, j] > .65 else '#172b4d')
    ax.set(xticks=range(n), yticks=range(n), xticklabels=result['classes'],
           yticklabels=result['classes'], xlabel='Clase predicha', ylabel='Clase real',
           title=f'EXP-19 · atribución multiclase · n={matrix.sum()}')
    fig.colorbar(heat, ax=ax, label='Fracción por clase real')
    fig.savefig(output, dpi=180, facecolor='white')
    plt.close(fig)


def write_notes(result, args):
    e = result['evaluation']
    counts = {name: {c: sum(r['label'] == i for r in rows) for i, c in enumerate(result['classes'])}
              for name, rows in result['identity']['manifest'].items()}
    table = '\n'.join(f"| {r['transform']} | {r['severity']} | {r['accuracy']:.4f} |"
                      for r in result['representative_degradation'])
    diagnostics = []
    clean_predictions = np.asarray(result['predictions'][0]['probabilities']).argmax(1)
    labels = np.asarray(result['predictions'][0]['labels'])
    for row, detail in zip(result['degradation'], result['predictions']):
        if not representative(row) or row['transform'] == 'clean':
            continue
        predicted = np.asarray(detail['probabilities']).argmax(1)
        corrected = int(((clean_predictions != labels) & (predicted == labels)).sum())
        introduced = int(((clean_predictions == labels) & (predicted != labels)).sum())
        matrix = np.asarray(row['confusion_matrix'])
        recalls = np.diag(matrix) / matrix.sum(1)
        diagnostics.append(f"- {row['transform']} {row['severity']}: corrige {corrected} errores limpios, "
                           f"introduce {introduced}; recall por clase en el orden declarado: "
                           + ', '.join(f'{value:.3f}' for value in recalls) + '.')
    diagnostic_text = '\n'.join(diagnostics)
    notes = f'''# EXP-19: atribución parcial

Generadores evaluados: {', '.join(result['generators'])}. Ausentes: {', '.join(result['unavailable_generators']) or 'ninguno'}.
Esta ejecución corresponde a {len(result['generators'])} generadores. Con DCGAN y diffusion es una atribución parcial (2 de 3 generadores);
la conclusión completa depende de EXP-03 (StyleGAN2-ADA). No permite concluir que el fingerprint sobreviva o no sobreviva en general.
Estas métricas son multiclase, no accuracy binaria ni sustitutos de las otras secciones.

## Metodología

Se hereda DualStreamDetector de EXP-03: ResNet18 ImageNet + FFT radial 32 bins,
fusión 544→64, ReLU, dropout 0.3. Solo se reemplaza Linear(64,1) por
Linear(64,{len(result['classes'])}). Cross-entropy sin pesos, logits en entrenamiento,
softmax y argmax en evaluación. Adam, lr={args.lr}, batch={args.batch_size},
{args.epochs} épocas, semilla PyTorch/NumPy 42, dispositivo {result['training']['device']}.
Se selecciona la menor cross-entropy de validación (primer empate), sin usar test:
época {result['training']['best_epoch']}.

load_real_split valida el JSON seed 42. common.split_for se llama por generador,
como EXP-06: máximo 3 falsas por real por generador, permutación NumPy 42 y
reparto proporcional por restos mayores. Se conservan exactamente esas falsas,
y una sola copia de los reales: train/val solo gen_pool (75/25), test solo holdout.
A diferencia de EXP-07/08, no se mezclan las falsas antes del límite global:
se usa la variante por generador de EXP-06 para representar cada clase.
Conteos: {counts}.
El JSON contiene el manifiesto completo y hashes de datos/split/checkpoint.
No se balancean clases; accuracy depende de estos soportes. Se incluye F1 por clase.

RadiographDataset de EXP-03 aplica a todas las clases escala de grises,
roundtrip bicúbico 16..128→128, FFT y normalización [-1,1] a tres canales.
Train: resize estocástico y flip horizontal. Val/test: resize con semilla 42+índice.
Índices del manifiesto multiclase fijos entre condiciones; no necesariamente coinciden
con los índices de test binario. Las 30 condiciones LEVELS y transform se importan
literalmente de EXP-09: roundtrip → postproceso → ambas ramas, sin reentrenar.
Ruido: 42000+índice (misma realización entre severidades); crop es fracción de área;
JPEG múltiple son 5 recodificaciones reales Q70. Resize no es severidad ordinal.

## Resultados

Accuracy limpia: {e['accuracy']:.6f}; n={e['n']}.
F1 por clase: {e['f1_per_class']}.
Matriz (filas verdaderas, columnas predichas; orden {result['classes']}):
{e['confusion_matrix']}.

| Transformación | Nivel | Accuracy multiclase |
| --- | --- | --- |
{table}

Las matrices, F1, probabilidades y etiquetas de cada condición se guardan para
investigar pérdida de atribución y posibles colapsos a una clase. No se impone
monotonía; con test pequeño hay variación discreta. No se han ajustado
hiperparámetros a los resultados de test.
Descomposición aritmética de cambios respecto al test limpio (no prueba causal
de qué huella visual utiliza la red):

{diagnostic_text}

EXP-13 reportó AUC=1 tanto sin como con la corrección de resize. La mitigación
compartida no demuestra neutralización del shortcut; una atribución alta aquí
no prueba huellas independientes del historial de construcción/resolución.
La figura usa escala secuencial blanco→azul #2a78d6 del proyecto,
normalizada por fila y anotada con conteos. El skill dataviz no está instalado;
la referencia de color es make_exp09_robustness_figure.py y specs/03.

## Reproducción

```bash
.venv/bin/python training/19_generator_attribution.py --generators {' '.join(result['generators'])} --epochs {args.epochs} --batch-size {args.batch_size} --lr {args.lr} --device {args.device} --threads {args.threads}
.venv/bin/python -m unittest discover -s training -p test_exp19.py -v
# Cuando EXP-03 exporte imágenes entrenadas exclusivamente con gen_pool:
.venv/bin/python training/19_generator_attribution.py --generators dcgan stylegan2ada diffusion --output-dir training/exp19_attribution_three_generators
```

El checkpoint separado es training/checkpoints/attribution_multiclass.pth y se
sobrescribe al reentrenar; cada carpeta conserva resultados y manifiestos.
Una petición explícita de un generador ausente aborta antes de entrenar.
'''
    (args.output_dir / 'notas.md').write_text(notes)


def main():
    parser = common.parser(__doc__)
    parser.add_argument('--output-dir', type=Path, default=common.HERE / 'exp19_attribution')
    args = parser.parse_args()
    generators, missing = common.prepare(args)
    print(f'Generadores: {list(generators)}; ausentes: {missing}', flush=True)
    classes = ['real', *generators]
    splits = attribution_splits(generators, args)
    identity = signature(splits, args, generators)
    identity.update(architecture='resnet18+fft32-multiclass', classes=classes)
    torch.manual_seed(42)
    np.random.seed(42)
    device = common.base.get_device() if args.device == 'auto' else torch.device(args.device)
    model = AttributionModel(len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader, val_loader = loader(splits[0], args, True), loader(splits[1], args)
    history, best_loss, best_state, best_epoch = [], float('inf'), None, None
    print(f'Dispositivo: {device}', flush=True)
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(model, train_loader, optimizer)
        validation = run_epoch(model, val_loader)
        row = dict(epoch=epoch, train_loss=training['loss'], val_loss=validation['loss'],
                   val_accuracy=metrics(validation, classes)['accuracy'])
        history.append(row)
        print(row, flush=True)
        if validation['loss'] < best_loss:
            best_loss, best_epoch = validation['loss'], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise ValueError('No hay checkpoint válido: pérdida de validación no finita')
    model.load_state_dict(best_state)
    checkpoint = common.HERE / 'checkpoints/attribution_multiclass.pth'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=best_state, classes=classes, identity=identity), checkpoint)
    rows, predictions = [], []
    for family, levels in robustness.LEVELS.items():
        for severity in levels:
            details = run_epoch(model, loader(splits[2], args,
                postprocess=partial(robustness.transform, family=family, severity=severity)))
            row = dict(transform=family, severity=severity, **metrics(details, classes))
            rows.append(row)
            predictions.append(dict(transform=family, severity=severity, **details))
            print(f'{family} {severity}: accuracy={row["accuracy"]:.6f}', flush=True)
    result = dict(task='multiclass_generator_attribution', classes=classes, generators=list(generators),
                  unavailable_generators=missing, identity=identity,
                  protocol=dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=42, threads=args.threads,
                                selection='minimum validation cross-entropy', max_fake_ratio_per_generator=3),
                  versions=common.metadata(args, generators, missing)['versions'],
                  training=dict(history=history, best_epoch=best_epoch, best_val_loss=best_loss, device=str(device)),
                  checkpoint=str(checkpoint.relative_to(common.ROOT)), checkpoint_sha256=digest(checkpoint),
                  evaluation={k: v for k, v in rows[0].items() if k not in ('transform', 'severity')},
                  degradation=rows, representative_degradation=[r for r in rows if representative(r)],
                  predictions=predictions,
                  limitation='Atribución parcial con los generadores disponibles; conclusión completa pendiente de EXP-03.')
    common.write_json(args.output_dir / 'results.json', result)
    plot_confusion(result, args.output_dir / 'confusion_matrix.png')
    write_notes(result, args)


if __name__ == '__main__':
    common.cli(main)
