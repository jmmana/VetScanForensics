"""Una figura de cuatro familias; métricas y referencia limpia sin imponer monotonía."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BLUE, ORANGE = '#2a78d6', '#eb6834'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE / 'exp09_robustness/results.json')
    parser.add_argument('--output', type=Path, default=HERE / 'exp09_robustness/robustness.png')
    args = parser.parse_args()
    rows = json.loads(args.results.read_text())
    clean = next(r for r in rows if r['transform'] == 'clean')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    panels = [('jpeg', 'JPEG', 'Calidad JPEG (menor = más degradación)'),
              ('resize', 'Resize bicúbico → 128', 'Tamaño intermedio (px; no es una escala ordinal)'),
              ('blur', 'Blur gaussiano', 'Sigma (px)'), ('noise', 'Ruido gaussiano', 'Sigma (escala [0, 1])')]
    for ax, (family, title, xlabel) in zip(axes.flat, panels):
        selected = sorted((r for r in rows if r['transform'] == family), key=lambda r: r['severity'])
        if not selected:
            raise ValueError(f'Faltan resultados de {family}')
        x = [r['severity'] for r in selected]
        for metric, label, color in [('accuracy', 'Accuracy', BLUE), ('roc_auc', 'ROC-AUC', ORANGE)]:
            ax.plot(x, [r[metric] for r in selected], 'o-', color=color, label=label, linewidth=2)
            ax.axhline(clean[metric], color=color, linestyle=':', alpha=.65, linewidth=1)
        ax.set(title=title, xlabel=xlabel, ylabel='Métrica', ylim=(0, 1.04), xticks=x)
        ax.grid(axis='y', alpha=.18)
        if family == 'jpeg':
            ax.invert_xaxis()
        ax.legend(loc='lower left', frameon=False)
    fig.suptitle(f'EXP-09 · test fijo n={clean["n"]} · puntos: postproceso; líneas punteadas: sin ataque')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor='white')
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
