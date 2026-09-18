"""Grad-CAM espacial y sensibilidad local de los 32 bins FFT, sin reentrenar."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import experiment_common as common
from posthoc_common import load_experiment

BLUE = '#2a78d6'


def gradcam(model, image, frequency):
    """Grad-CAM de layer4: ReLU(sum_k mean_ij(d score/d A_kij) A_k)).

    score = logit para predicción falsa, -logit para predicción real.
    Se usa autograd.grad sobre el tensor capturado por forward hook para evitar
    conflictos de backward hooks con los ReLU inplace de torchvision.
    """
    activations = []
    hook = model.spatial.layer4.register_forward_hook(lambda module, inputs, output: activations.append(output))
    try:
        logit = model(image, frequency)
        target = 1 if logit.item() >= 0 else 0
        score = logit if target else -logit
        activation = activations[0]
        gradient, = torch.autograd.grad(score.sum(), activation)
        raw = torch.relu((gradient.mean((2, 3), keepdim=True) * activation).sum(1, keepdim=True))
        raw_array = raw.detach().cpu().numpy()[0, 0]
        if not np.isfinite(raw_array).all() or np.ptp(raw_array) <= 1e-10:
            raise ValueError('Grad-CAM constante/no finito: revisar gradientes/objetivo; no se sustituye ni se descarta la imagen')
        heat = F.interpolate(raw, size=(128, 128), mode='bilinear', align_corners=False)[0, 0]
        heat = (heat - heat.min()) / (heat.max() - heat.min())
        return heat.detach().cpu().numpy(), raw_array, dict(
            probability_fake=torch.sigmoid(logit).item(), target_class=target,
            raw_min=float(raw_array.min()), raw_max=float(raw_array.max()), raw_std=float(raw_array.std()),
            activation_gradient_norm=gradient.norm().item())
    finally:
        hook.remove()


def frequency_gradients(model, dataset, device, batch_size):
    gradients = []
    for start in range(0, len(dataset), batch_size):
        batch = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        images = torch.stack([b[0] for b in batch]).to(device)
        frequencies = torch.stack([b[1] for b in batch]).to(device).requires_grad_(True)
        logit = model(images, frequencies)
        gradient, = torch.autograd.grad(logit.sum(), frequencies)
        gradients.append(gradient.detach().cpu().numpy())
    values = np.concatenate(gradients)
    if values.shape != (len(dataset), 32) or not np.isfinite(values).all() or np.max(np.abs(values)) == 0:
        raise ValueError('Gradientes de frecuencia inválidos')
    return values


def main():
    parser = common.parser(__doc__)
    parser.add_argument('--output-dir', type=Path, default=common.HERE / 'exp12_explainability')
    args = parser.parse_args()
    model, splits, provenance = load_experiment(args)
    device = next(model.parameters()).device
    dataset = common.base.RadiographDataset(*splits[2], train=False)
    paths, labels = splits[2]
    # Selección por orden de split, nunca por confianza o apariencia del mapa.
    groups = {'real': [i for i, label in enumerate(labels) if label == 0]}
    generators, _ = common.resolve_generators(args.generators)
    for name, files in generators.items():
        file_set = set(files)
        groups[name] = [i for i, path in enumerate(paths) if path in file_set]
    if len(groups['real']) < 2 or sum(len(indices) for name, indices in groups.items() if name != 'real') < 2:
        raise ValueError('Se necesitan al menos dos reales y dos falsas en test para mostrar ejemplos')
    selected = [(name, i) for name, indices in groups.items() for i in indices[:2]]
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica']})
    fig, axes = plt.subplots(len(selected), 3, figsize=(8, 2.5 * len(selected)), layout='constrained', squeeze=False)
    records, arrays = [], {}
    for row, (name, idx) in enumerate(selected):
        image, frequency, _ = dataset[idx]
        heat, raw, record = gradcam(model, image[None].to(device), frequency[None].to(device))
        processed = (image[0].numpy() + 1) / 2
        with Image.open(paths[idx]) as source:
            original = np.asarray(source.convert('L').resize((128, 128), Image.Resampling.BICUBIC))
        axes[row, 0].imshow(original, cmap='gray', vmin=0, vmax=255)
        axes[row, 1].imshow(processed, cmap='gray', vmin=0, vmax=1)
        axes[row, 2].imshow(processed, cmap='gray', vmin=0, vmax=1)
        overlay = axes[row, 2].imshow(heat, cmap='inferno', vmin=0, vmax=1, alpha=.55)
        axes[row, 0].set_title(f'{name}: {paths[idx].name}\nOriginal (vista a 128 px)', fontsize=9)
        axes[row, 1].set_title('Entrada del detector\nResize del protocolo', fontsize=9)
        target_name = 'falsa' if record['target_class'] else 'real'
        axes[row, 2].set_title(f'Grad-CAM: objetivo {target_name}\nP(falsa)={record["probability_fake"]:.3f}', fontsize=9)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
        records.append(dict(index=idx, group=name, path=str(paths[idx].relative_to(common.ROOT)), label=labels[idx], **record))
        arrays[f'heat_{idx}'], arrays[f'raw_{idx}'] = heat, raw
    fig.colorbar(overlay, ax=axes[:, 2], shrink=.55, label='Grad-CAM normalizado por imagen')
    fig.suptitle('EXP-12 · rama espacial · layer4 (4×4 → 128×128)')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / 'gradcam_examples.png', dpi=180, facecolor='white')
    plt.close(fig)
    gradients = frequency_gradients(model, dataset, device, args.batch_size)
    importance = np.abs(gradients).mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
    ax.bar(np.arange(32), importance, color=BLUE)
    ax.set(xlabel='Bin radial FFT (bajas → altas frecuencias)', ylabel='Media |∂logit(falsa) / ∂bin|',
           title=f'Sensibilidad local de frecuencia · test completo (n={len(dataset)})', xticks=np.arange(0, 32, 2))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.2)
    fig.savefig(args.output_dir / 'frequency_saliency.png', dpi=180, facecolor='white')
    plt.close(fig)
    np.savez_compressed(args.output_dir / 'attributions.npz', frequency_gradients=gradients, **arrays)
    common.write_json(args.output_dir / 'metadata.json', dict(**provenance, examples=records,
        gradcam_method='ReLU of activation weighted by spatial mean gradient; layer4; predicted-class signed logit',
        frequency_method='mean absolute gradient of fake logit w.r.t. normalized FFT bins over full test',
        frequency_importance=importance.tolist(), frequency_signed_mean=gradients.mean(0).tolist()))
    top = np.argsort(-importance)[:5].tolist()
    notes = f'''# EXP-12: método y límites

Checkpoint: `{provenance['checkpoint']}`; reutilizado: {provenance['reused']}.
Los ejemplos son los primeros dos reales y los hasta dos de cada generador
presentes en el test fijo, sin selección por confianza o por aspecto del mapa.
Original = vista del archivo a 128 px; entrada = imagen tras el resize compartido.
La superposición se alinea con esta última, que es la imagen evaluada.

Grad-CAM manual con forward hook en `spatial.layer4` y `autograd.grad`:
promedio espacial del gradiente, suma ponderada de activaciones y ReLU.
Objetivo: logit de la clase predicha (z para falsa, -z para real), sin sigmoid
para evitar saturación. La frecuencia permanece fija. No se aplica Grad-CAM
a la rama FFT. Método de [Selvaraju et al., ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html).

Mapas nativos de 4×4, interpolados a 128×128 y normalizados individualmente:
no permiten delimitar estructuras anatómicas finas ni comparar intensidad
absoluta entre imágenes. Todos los mapas elegidos son finitos y no constantes;
sus rangos, desviaciones y normas del gradiente están en `metadata.json`.
Esto verifica variación numérica, no validez causal de la explicación.

Frecuencia: gradiente local del **logit falsa** respecto a cada uno de los 32
bins normalizados que recibe el MLP. Media del valor absoluto en las {len(dataset)}
imágenes de test; también se conservan gradientes individuales y medias con signo.
Se elige gradiente directo para medir sensibilidad local sin introducir una
referencia arbitraria para integrated gradients. No mide contribución acumulada,
ni fracción de la decisión, ni el gradiente respecto a píxeles/FFT sin normalizar.
Los bins con mayor media absoluta en esta ejecución (índices 0–31) son {top}.

## Observaciones automáticas reproducibles

'''
    for record in records:
        heat = arrays[f'heat_{record["index"]}']
        y, x = np.unravel_index(heat.argmax(), heat.shape)
        notes += f'- `{record["path"]}`: P(falsa)={record["probability_fake"]:.4f}; máximo interpolado en (x={x}, y={y}); std del mapa nativo={record["raw_std"]:.6g}.\n'
    notes += '\nLa revisión visual específica de esta corrida se documenta en `../exp0912_notas.md`.\n'
    (args.output_dir / 'notas.md').write_text(notes)
    print(f'Figuras y atribuciones guardadas en {args.output_dir}', flush=True)


if __name__ == '__main__':
    common.cli(main)
